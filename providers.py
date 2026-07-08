"""Provider-specific model shims, quarantined away from the pipeline logic.

Everything here exists because of how specific serving stacks misbehave, not
because of what the product does:

  * MiniMax via some OpenRouter providers emits tool calls as XML in message
    content instead of OpenAI tool_calls — in two dialects: the clean
    ``<minimax:tool_call>`` form and a corrupted ``]<]minimax[>[`` delimiter
    with direct ``<param>value</param>`` tags (seen when a no-tools provider
    serves the model; kept as a safety net for self-hosted vLLM too).
  * Some stacks return empty tool_call ids, which breaks the agent loop.
  * Text-only models can't ingest binary content blocks (base64 PDFs) that
    deepagents' read_file emits for files under /input/.

FixedToolIdModel also enforces the per-run token budget: the "Budget: max N
tokens" prompt line is advisory, this is the actual stop.
"""
import json
import re
import threading
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr


class BudgetExceeded(RuntimeError):
    """Per-run LLM token budget is spent; the agent loop must stop.

    Raised BEFORE the next model call starts (never mid-call), so the run
    aborts at a step boundary and the checkpointer holds every completed
    step for salvage."""


# Content-block types a text-only model can't ingest (PDF/image/audio uploads).
_BINARY_BLOCK_TYPES = {"file", "image", "audio", "image_url", "input_file"}

# Clean dialect: <minimax:tool_call> … </minimax:tool_call>
# Corrupted dialect (broken serving stack): ]<]minimax[>[ … ]<]/minimax[>[
_MINIMAX_OPEN_RE = re.compile(r"<minimax:tool_call>|\]<\]minimax\[>\[", re.IGNORECASE)
_MINIMAX_CLOSE_RE = re.compile(r"</minimax:tool_call>|\]<\]/minimax\[>\[", re.IGNORECASE)
_MINIMAX_INVOKE_OPEN_RE = re.compile(r'<invoke\s+name="([^"]+)"\s*>', re.IGNORECASE)
_MINIMAX_INVOKE_CLOSE_RE = re.compile(r"</invoke>", re.IGNORECASE)
_MINIMAX_PARAM_OPEN_RE = re.compile(r'<parameter\s+name="([^"]+)"\s*>', re.IGNORECASE)
_MINIMAX_PARAM_CLOSE_RE = re.compile(r"</parameter>", re.IGNORECASE)
# Direct-tag params of the corrupted dialect: <file_path>…</file_path>
_MINIMAX_DIRECT_TAG_RE = re.compile(r"<(\w+)>(.*?)(?:</\1>|$)", re.DOTALL)
# MiniMax reasoning markers that leak into content on some stacks.
_MM_THINK_RE = re.compile(r"</?mm:think>", re.IGNORECASE)


def _parse_param_value(raw: str):
    """Param values arrive as JSON when structured; keep raw string otherwise.
    Tolerates a trailing partial close-tag left by max_tokens truncation."""
    raw = re.sub(r"</\w*$", "", raw).strip()
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


class FixedToolIdModel(ChatOpenAI):
    """Workarounds for Minimax/OpenRouter:
    1. Fills in empty tool_call IDs.
    2. Converts minimax tool-call XML (clean or corrupted dialect) emitted in
       message content into real OpenAI-format tool_calls so the agent loop
       can dispatch them. The parser tolerates truncated/unclosed tags so a
       response cut by max_tokens still yields a (possibly partial) tool call
       instead of a dead agent loop.
    Sanitizes before sending and after receiving; retries once on empty-id errors.

    Also enforces the per-run token budget: every generate() first checks the
    running total and raises BudgetExceeded once `token_budget` is spent.
    Shared across the orchestrator + all subagents, since they use the same
    model instance.
    """

    token_budget: int = 0  # total tokens per model instance; 0 = unlimited
    _tokens_used: int = PrivateAttr(default=0)
    _usage_lock: object = PrivateAttr(default_factory=threading.Lock)

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    def _check_budget(self) -> None:
        if self.token_budget and self._tokens_used >= self.token_budget:
            raise BudgetExceeded(
                f"LLM token budget spent: {self._tokens_used}/{self.token_budget}"
            )

    def _register_usage(self, result) -> None:
        total = 0
        for gen_list in result.generations:
            for gen in gen_list:
                usage = getattr(gen.message, "usage_metadata", None) or {}
                total += int(usage.get("total_tokens") or 0)
        if not total:  # some providers only report an aggregate
            usage = (result.llm_output or {}).get("token_usage") or {}
            total = int(usage.get("total_tokens") or 0)
        if total:
            with self._usage_lock:
                self._tokens_used += total

    @staticmethod
    def _slice_until(text: str, close_re: re.Pattern[str], next_open: int) -> tuple[str, int]:
        """Return (value, abs_end) up to either the first close-tag match within
        [0:next_open] or up to next_open itself."""
        sub = text[:next_open]
        m = close_re.search(sub)
        if m:
            return sub[:m.start()], m.end()
        return sub, next_open

    @classmethod
    def _parse_invoke_args(cls, invoke_body: str) -> dict:
        """Params in either dialect: <parameter name="x">v</parameter> or the
        corrupted direct tags <x>v</x>."""
        args: dict = {}
        params = list(_MINIMAX_PARAM_OPEN_RE.finditer(invoke_body))
        if params:
            for j, pm in enumerate(params):
                pname = pm.group(1)
                val_start = pm.end()
                next_p_start = params[j + 1].start() if j + 1 < len(params) else len(invoke_body)
                raw, _ = cls._slice_until(
                    invoke_body[val_start:], _MINIMAX_PARAM_CLOSE_RE,
                    next_p_start - val_start,
                )
                args[pname] = _parse_param_value(raw)
            return args
        for tm in _MINIMAX_DIRECT_TAG_RE.finditer(invoke_body):
            args[tm.group(1)] = _parse_param_value(tm.group(2))
        return args

    @classmethod
    def _parse_minimax_xml(cls, msg: AIMessage) -> None:
        """Extract minimax tool-call XML from message content into tool_calls.

        Handles both delimiter dialects; tolerates unclosed </parameter>,
        </invoke> and block closers so that responses cut off by max_tokens
        still produce a tool call.
        """
        content = msg.content if isinstance(msg.content, str) else ""
        if not content or not _MINIMAX_OPEN_RE.search(content):
            return

        new_calls: list[dict] = []
        ak_new: list[dict] = []
        blocks_to_strip: list[tuple[int, int]] = []

        pos = 0
        while True:
            m_open = _MINIMAX_OPEN_RE.search(content, pos)
            if not m_open:
                break
            body_start = m_open.end()
            m_close = _MINIMAX_CLOSE_RE.search(content, body_start)
            if m_close:
                body_end = m_close.start()
                block_end = m_close.end()
            else:
                body_end = len(content)
                block_end = len(content)
            block = content[body_start:body_end]
            blocks_to_strip.append((m_open.start(), block_end))

            invokes = list(_MINIMAX_INVOKE_OPEN_RE.finditer(block))
            for i, im in enumerate(invokes):
                name = im.group(1)
                inv_body_start = im.end()
                next_inv_start = invokes[i + 1].start() if i + 1 < len(invokes) else len(block)
                invoke_body, _ = cls._slice_until(
                    block[inv_body_start:], _MINIMAX_INVOKE_CLOSE_RE,
                    next_inv_start - inv_body_start,
                )

                args = cls._parse_invoke_args(invoke_body)

                tid = f"call_{uuid.uuid4().hex[:16]}"
                new_calls.append({"name": name, "args": args, "id": tid, "type": "tool_call"})
                ak_new.append({
                    "id": tid,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                })

            pos = block_end

        if not new_calls and not blocks_to_strip:
            return
        if new_calls:
            msg.tool_calls = list(msg.tool_calls or []) + new_calls
            ak = list(msg.additional_kwargs.get("tool_calls") or [])
            ak.extend(ak_new)
            msg.additional_kwargs["tool_calls"] = ak
        if blocks_to_strip:
            chunks = []
            prev = 0
            for s, e in blocks_to_strip:
                chunks.append(content[prev:s])
                prev = e
            chunks.append(content[prev:])
            msg.content = _MM_THINK_RE.sub("", "".join(chunks)).strip()

    @staticmethod
    def _as_conversations(messages):
        """generate() receives a batch: list[list[BaseMessage]]. A few call paths
        pass a flat list[BaseMessage]. Normalize to a list of conversations so the
        in-place patches below operate on individual messages, not the batch list."""
        if messages and isinstance(messages[0], (list, tuple)):
            return messages
        return [messages]

    @classmethod
    def _strip_binary_blocks(cls, messages) -> None:
        """Remove non-text (PDF/image) content blocks before sending to a
        text-only model.

        deepagents' built-in read_file returns a binary file (e.g. a PDF under
        /input/) as a base64 content block. A text-only model (deepseek-v4-flash,
        minimax-m3) can't ingest it: the host tries to parse the upload and times
        out, returning {'error': {'message': 'Timed out parsing LC_AUTOGENERATED',
        'code': 504}}. We drop those blocks and leave a hint so the agent
        re-routes to read_pdf (docling → /scratch/<file>.md) for text extraction.
        """
        hint = ("[binary file content removed — this model cannot read raw files. "
                "Use read_pdf('/input/<file>.pdf') to extract text into "
                "/scratch/<file>.md, then read_file that .md.]")
        for convo in cls._as_conversations(messages):
            for msg in convo:
                content = getattr(msg, "content", None)
                if not isinstance(content, list):
                    continue
                kept, stripped = [], False
                for block in content:
                    if isinstance(block, dict) and (
                        block.get("type") in _BINARY_BLOCK_TYPES
                        or "base64" in block or "file_data" in block or "image_url" in block
                    ):
                        stripped = True
                        continue
                    kept.append(block)
                if stripped:
                    if not any(isinstance(b, dict) and b.get("type") == "text" for b in kept):
                        kept.append({"type": "text", "text": hint})
                    msg.content = kept

    @staticmethod
    def _patch_ai(msg: AIMessage) -> None:
        for tc in (msg.tool_calls or []):
            if not tc.get("id"):
                tc["id"] = f"call_{uuid.uuid4().hex[:16]}"
        for tc in (getattr(msg, "invalid_tool_calls", None) or []):
            if not tc.get("id"):
                tc["id"] = f"call_{uuid.uuid4().hex[:16]}"
        ak = msg.additional_kwargs.get("tool_calls") or []
        for i, tc_raw in enumerate(ak):
            if i < len(msg.tool_calls or []) and msg.tool_calls[i].get("id"):
                tc_raw["id"] = msg.tool_calls[i]["id"]
            elif not tc_raw.get("id"):
                tc_raw["id"] = f"call_{uuid.uuid4().hex[:16]}"

    @classmethod
    def _sanitize(cls, messages) -> None:
        """Walk message history and patch any empty tool_call_id in-place.
        ToolMessages with empty ids are matched FIFO to the most recent AIMessage."""
        for convo in cls._as_conversations(messages):
            pending: list[str] = []
            for msg in convo:
                if isinstance(msg, AIMessage) and (msg.tool_calls or msg.additional_kwargs.get("tool_calls")):
                    cls._patch_ai(msg)
                    pending = [tc["id"] for tc in (msg.tool_calls or [])]
                elif isinstance(msg, ToolMessage) and not getattr(msg, "tool_call_id", None):
                    if pending:
                        msg.tool_call_id = pending.pop(0)

    def generate(self, messages, stop=None, callbacks=None, **kwargs):
        self._check_budget()
        self._strip_binary_blocks(messages)
        self._sanitize(messages)
        try:
            result = super().generate(messages, stop=stop, callbacks=callbacks, **kwargs)
        except ValueError as e:
            if "tool call id" not in str(e).lower():
                raise
            self._sanitize(messages)
            result = super().generate(messages, stop=stop, callbacks=callbacks, **kwargs)
        self._register_usage(result)
        for gen_list in result.generations:
            for gen in gen_list:
                if isinstance(gen.message, AIMessage):
                    self._parse_minimax_xml(gen.message)
                    self._patch_ai(gen.message)
        return result
