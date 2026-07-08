import importlib.util
import json
import os
import re
import sys
import shutil
import argparse
import sqlite3
import uuid
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Egress guard first: with NETWORK_GUARD=strict nothing below — including the
# tracing setup — can open a connection to a host that isn't configured.
from netguard import install_guard
install_guard()

from tracing import setup_tracing, get_run_config, flush_tracing
setup_tracing()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import (
    CompositeBackend, FilesystemBackend,
    StateBackend, StoreBackend,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from tools import (
    read_pdf, extract_tables, search_examples,
    search_bm25, search_vector, list_sections, read_section,
)
from schemas import ScoringResult, CharterStructuredOutput
from pydantic import BaseModel


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EXTRACTION_TOOLS = [read_pdf, extract_tables, search_bm25, search_vector,
                     list_sections, read_section, search_examples]
_ORCHESTRATOR_TOOLS = [search_examples]

WORK_ROOT = "/workspace/work/current"
AGENTS_ROOT = Path(os.environ.get("AGENTS_ROOT", "/workspace/agents"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/workspace/output"))
DEFAULT_BUSINESS_RULES = Path(__file__).parent / "agent_init" / "buisness_rules.md"

MAX_TOKENS = int(os.getenv("MAX_TOKENS_PER_RUN", "200000"))
MAX_SUBAGENTS = int(os.getenv("MAX_SUBAGENTS_PER_RUN", "15"))
HITL_ENABLED = os.getenv("HITL_ENABLED", "false").lower() in ("1", "true", "yes")


def prepare_run_workspace(src: str | Path) -> Path:
    """Stage input under WORK_ROOT/input/ for each run.

    Both /input/ and /scratch/ are wiped and recreated per run so a run never sees
    another document's leftovers (stale PDFs, cached markdown, CSVs). This keeps
    every consumer — the agent, retrieval, and finalization — scoped strictly to
    the current input. Relative ``src`` is resolved against ``WORKSPACE_ROOT``
    (default ``/workspace``)."""
    work_root = Path(WORK_ROOT)
    workspace_root = Path(os.environ.get("WORKSPACE_ROOT", "/workspace"))

    inp = work_root / "input"
    scratch = work_root / "scratch"
    for d in (inp, scratch):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    raw = Path(src)
    src_path = raw.resolve() if raw.is_absolute() else (workspace_root / raw).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Input path not found: {raw} → {src_path}")

    if src_path.is_file():
        shutil.copy2(src_path, inp / src_path.name)
    elif src_path.is_dir():
        for f in sorted(src_path.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                shutil.copy2(f, inp / f.name)
    else:
        raise FileNotFoundError(f"Not a file or directory: {src_path}")

    if not any(inp.iterdir()):
        raise FileNotFoundError(f"No files staged from {src_path}")

    os.environ["WORK_ROOT"] = WORK_ROOT
    return inp


_OUTPUT_SCHEMA = """\
=== REQUIRED OUTPUT SCHEMA ===
Your FINAL REPORT must populate ALL fields below (use exact field names as section headers).
Fill each field with REAL content quoted from /input/ — never write the words
"list" or "string". If a field is absent in the document write "не указано".

The FINAL REPORT is the ONLY source serialized into the structured output.
Fields come in TWO shapes — use the right one:
  • NAME fields (supreme/collegial/sole bodies): a short canonical ORGAN NAME only.
    Do NOT paste a clause number or a whole sentence from the charter — normalize to
    the name (e.g. "Наблюдательный совет", "Генеральный директор – Председатель Правления").
  • CLAUSE fields (transactions, restrictions, protocol): the FULL verbatim clause text,
    prefixed with the MOST SPECIFIC sub-clause number (e.g. 12.1.4(3)), never the parent.

  supreme_governing_body — Высший орган управления   [NAME]
        (one line, name only; default: Общее собрание участников, always present)
  collegial_governing_bodies — Коллегиальные органы управления   [NAME]
        (one name per body: Совет директоров, Наблюдательный совет, Правление, …)
  sole_executive_bodies — Единоличные органы управления   [NAME]
        (one name per SOLE body: Генеральный директор, Директор, Управляющий, … —
         do NOT include Правление/Дирекция, those are collegial)
  major_transaction_clauses — Пункты о крупных сделках   [CLAUSE]
        (one item per clause: "<clause_number>. <full clause text>";
         collect from the competence of EVERY organ that has it)
  related_party_transaction_clauses — Пункты о сделках с заинтересованностью   [CLAUSE]
        (one item per clause: "<clause_number>. <full clause text>";
         collect from the competence of EVERY organ that has it)
  general_meeting_minutes_protocol — Протокол общего собрания   [CLAUSE]
        (one line: clause number + method of certification)
  sole_executive_body_restrictions — Уставные ограничения единоличного ИО   [CLAUSE]
        (one item per restriction, the most specific sub-clause + its text)
"""


_FIELD_SUBAGENTS: list[dict] = [
    {
        "name": "extract-supreme-body",
        "description": "Extract supreme governing body (высший орган управления).",
        "field": "supreme_governing_body", "style": "name",
        "topic": "Высший орган управления (Общее собрание участников)",
    },
    {
        "name": "extract-collegial-bodies",
        "description": "Extract collegial governing bodies (Совет директоров, Наблюдательный совет, Правление).",
        "field": "collegial_governing_bodies", "style": "name",
        "topic": "Коллегиальные органы управления",
    },
    {
        "name": "extract-sole-executive",
        "description": "Extract sole executive bodies (Генеральный директор, Директор, Управляющий).",
        "field": "sole_executive_bodies", "style": "name",
        "topic": "Только ЕДИНОЛИЧНЫЕ исполнительные органы (Генеральный директор, Директор, Управляющий). НЕ Правление",
    },
    {
        "name": "extract-major-transactions",
        "description": "Extract clauses about major transactions (крупные сделки).",
        "field": "major_transaction_clauses",
        "topic": "Крупные сделки, % от активов, пороги одобрения — собери из компетенции ВСЕХ органов (и Общего собрания, и Наблюдательного совета / Совета директоров)",
    },
    {
        "name": "extract-related-party-transactions",
        "description": "Extract clauses about related-party transactions (сделки с заинтересованностью).",
        "field": "related_party_transaction_clauses",
        "topic": "Сделки с заинтересованностью — собери из компетенции ВСЕХ органов (и Общего собрания, и Наблюдательного совета / Совета директоров)",
    },
    {
        "name": "extract-meeting-protocol",
        "description": "Extract general-meeting minutes certification (протокол ОСУ).",
        "field": "general_meeting_minutes_protocol",
        "topic": "Протокол общего собрания, способ удостоверения",
    },
    {
        "name": "extract-executive-restrictions",
        "description": "Extract charter restrictions on the sole executive body.",
        "field": "sole_executive_body_restrictions",
        "topic": "Уставные ограничения единоличного исполнительного органа",
    },
]


def _build_subagent_prompt(field: str, topic: str, style: str = "clause") -> str:
    common = (
        f"You extract the OUTPUT SCHEMA field `{field}`.\n"
        f"Topic: {topic}.\n"
        "Work only from /input/ — never copy article numbers or names from examples.\n"
        "Workflow:\n"
        "  1. read_pdf('/input/<file>') if a /scratch/<file>.md does not yet exist.\n"
        "  2. list_sections('/scratch/<file>.md') to see the outline, then "
        "search_bm25 / search_vector for the topic keywords to locate the relevant section(s).\n"
        "  3. read_section('/scratch/<file>.md', '<key>') to read each relevant section IN FULL "
        "(e.g. the competence of EVERY organ that can hold this topic), so you don't miss anything "
        "that snippets cut off.\n"
        "  4. (Optional) search_examples(task_description=..., step_hint=<topic>) for FORMAT only.\n"
    )
    if style == "name":
        return common + (
            "OUTPUT: return the canonical ORGAN NAME(S) only — short, normalized. "
            "Do NOT prefix a clause/article number and do NOT paste a whole sentence from the "
            "charter (the article's opening sentence is a definition, not the answer). "
            "Extract just the name, e.g. \"Наблюдательный совет\".\n"
            f"Return: the name(s) for `{field}`, one per line."
        )
    return common + (
        "Rules: cite the MOST SPECIFIC sub-clause number (e.g. 12.1.4(3)), not the parent; "
        "return the FULL verbatim clause text, not a summary.\n"
        f"Return: every relevant item for `{field}`, each as \"<clause_number>. <verbatim text>\", "
        "with clause numbers that actually appear in the input doc."
    )


def _build_field_subagents() -> list[dict]:
    out = []
    for cfg in _FIELD_SUBAGENTS:
        out.append({
            "name": cfg["name"],
            "description": cfg["description"],
            "system_prompt": _build_subagent_prompt(
                cfg["field"], cfg["topic"], cfg.get("style", "clause")
            ),
            "tools": list(_EXTRACTION_TOOLS),
            "skills": ["/skills/"],
        })
    return out


def _load_text(source) -> str:
    """Resolve `source` to text: if it's an existing file path, read it; else use as-is."""
    if source is None:
        return ""
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
        return Path(source).read_text(encoding="utf-8")
    return str(source)


def _build_prompt(business_rules: str, startup_context: str = "") -> str:
    subagent_list = "\n".join(
        f"  - `{cfg['name']}` → field `{cfg['field']}` ({cfg['topic']})"
        for cfg in _FIELD_SUBAGENTS
    )
    preloaded = ""
    if startup_context.strip():
        preloaded = (
            "=== PRELOADED INSTRUCTIONS (already loaded for you — do NOT read these "
            "files again, it wastes a turn) ===\n" + startup_context.strip() + "\n\n"
        )
    return (
        "You are an orchestrator for bank process automation. "
        "You analyze LLC charters and extract structured legal data.\n\n"

        "=== BUSINESS RULES ===\n"
        + business_rules + "\n\n"

        + _OUTPUT_SCHEMA + "\n"

        + preloaded

        + "VIRTUAL FILESYSTEM:\n"
        "- /input/<file>     : input documents for this run (read-only by convention).\n"
        "- /scratch/<file>   : ephemeral scratch space (per-thread).\n"
        "- /memories/<file>  : long-term notes that persist across runs.\n"
        "- /skills/<name>/SKILL.md : on-demand playbooks (load when description matches).\n"
        "- /instructions/process.md and /instructions/tool_tips.md : workflow rules for ANY charter (read-only).\n\n"

        "PDF TOOLS (do NOT re-read the PDF after calling these):\n"
        "- read_pdf('/input/file.pdf') saves markdown to /scratch/file.md.\n"
        "- extract_tables('/input/file.pdf') saves CSVs under /scratch/.\n"
        "- For text search use built-in `grep`, or `search_bm25`/`search_vector` on /scratch/*.md.\n"
        "SECTION TOOLS (read structure, then whole sections):\n"
        "- list_sections('/scratch/file.md') → the outline (Статья/clause keys + titles).\n"
        "- read_section('/scratch/file.md', '12') → all of Статья 12 incl. 12.1, 12.1.4 …; "
        "pass a sub-key ('12.1.4') for just that sub-clause. Use this to read a relevant "
        "section IN FULL instead of relying on truncated snippets.\n\n"

        "STEP 1 - STARTUP: the process & tool-tips are already in PRELOADED INSTRUCTIONS above "
        "(do NOT read those files). If useful: ls /skills/ + read a matching SKILL.md, ls /memories/, "
        "search_examples(task_description=<task>).\n\n"

        "STEP 2 - PLAN: write_todos with one todo per OUTPUT SCHEMA field.\n\n"

        "STEP 3 - DELEGATE: for each todo, call task() with the matching named subagent and verify its result. "
        "Available extraction subagents:\n"
        f"{subagent_list}\n"
        "Verify: result must contain clause IDs that appear in /input/ AND substantive quoted text. "
        "Reject (≤2 retries) if it reuses example article numbers not present in the doc, then mark the todo done.\n\n"

        "STEP 4 - AGGREGATE: emit the FINAL REPORT (Markdown headers = OUTPUT SCHEMA field names). "
        "This report is the ONLY text serialized into the structured output. Respect each field's "
        "shape: NAME fields = a short organ name only; CLAUSE fields = the full verbatim clause "
        "text under its header.\n\n"
        "STEP 5 - SAVE: write_file('/memories/run.json', <lessons learned summary>).\n\n"

        "CRITICAL RULES:\n"
        "- The final report MUST cover all 7 OUTPUT SCHEMA fields, each in its [NAME]/[CLAUSE] shape.\n"
        "- NAME fields carry just the organ name (no clause number, no full sentence); sole_executive_bodies "
        "excludes Правление/Дирекция (collegial).\n"
        "- Every clause ID and legal text MUST come from /input/ — examples are for FORMAT only.\n"
        "- For крупные сделки and сделки с заинтересованностью, the result MUST cover the competence "
        "of EVERY organ (Общее собрание AND Наблюдательный совет / Совет директоров). Tell the "
        "subagent to list_sections + read_section each organ's article IN FULL, and reject a result "
        "that only covers one organ.\n"
        "- Each CLAUSE item must cite the MOST SPECIFIC sub-clause (e.g. 12.1.4(3)), never a whole parent clause.\n"
        "- Keep subagent instructions ≤ 2000 chars. Pass only essential context.\n"
        f"- Budget: max {MAX_SUBAGENTS} subagents, max {MAX_TOKENS} tokens per run.\n"
    )


# Content-block types a text-only model can't ingest (PDF/image/audio uploads).
_BINARY_BLOCK_TYPES = {"file", "image", "audio", "image_url", "input_file"}

_MINIMAX_OPEN_RE = re.compile(r"<minimax:tool_call>", re.IGNORECASE)
_MINIMAX_CLOSE_RE = re.compile(r"</minimax:tool_call>", re.IGNORECASE)
_MINIMAX_INVOKE_OPEN_RE = re.compile(r'<invoke\s+name="([^"]+)"\s*>', re.IGNORECASE)
_MINIMAX_INVOKE_CLOSE_RE = re.compile(r"</invoke>", re.IGNORECASE)
_MINIMAX_PARAM_OPEN_RE = re.compile(r'<parameter\s+name="([^"]+)"\s*>', re.IGNORECASE)
_MINIMAX_PARAM_CLOSE_RE = re.compile(r"</parameter>", re.IGNORECASE)


class _FixedToolIdModel(ChatOpenAI):
    """Workarounds for Minimax/OpenRouter:
    1. Fills in empty tool_call IDs.
    2. Converts <minimax:tool_call> XML emitted in message content into real
       OpenAI-format tool_calls so the agent loop can dispatch them. The parser
       tolerates truncated/unclosed tags so a response cut by max_tokens still
       yields a (possibly partial) tool call instead of a dead agent loop.
    Sanitizes before sending and after receiving; retries once on empty-id errors.
    """

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
    def _parse_minimax_xml(cls, msg: AIMessage) -> None:
        """Extract <minimax:tool_call> XML from message content into tool_calls.

        Tolerates unclosed </parameter>, </invoke>, </minimax:tool_call> so that
        responses cut off by max_tokens still produce a tool call.
        """
        content = msg.content if isinstance(msg.content, str) else ""
        if "<minimax:tool_call>" not in content.lower():
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

                args: dict = {}
                params = list(_MINIMAX_PARAM_OPEN_RE.finditer(invoke_body))
                for j, pm in enumerate(params):
                    pname = pm.group(1)
                    val_start = pm.end()
                    next_p_start = params[j + 1].start() if j + 1 < len(params) else len(invoke_body)
                    raw, _ = cls._slice_until(
                        invoke_body[val_start:], _MINIMAX_PARAM_CLOSE_RE,
                        next_p_start - val_start,
                    )
                    raw = raw.strip()
                    try:
                        args[pname] = json.loads(raw)
                    except (ValueError, TypeError):
                        args[pname] = raw

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
            msg.content = "".join(chunks).strip()

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
        self._strip_binary_blocks(messages)
        self._sanitize(messages)
        try:
            result = super().generate(messages, stop=stop, callbacks=callbacks, **kwargs)
        except ValueError as e:
            if "tool call id" not in str(e).lower():
                raise
            self._sanitize(messages)
            result = super().generate(messages, stop=stop, callbacks=callbacks, **kwargs)
        for gen_list in result.generations:
            for gen in gen_list:
                if isinstance(gen.message, AIMessage):
                    self._parse_minimax_xml(gen.message)
                    self._patch_ai(gen.message)
        return result


def _build_model() -> ChatOpenAI:
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    extra_body: dict | None = None
    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "openrouter":
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY")
        # OpenRouter load-balances a model across providers. Some (e.g. Novita,
        # AtlasCloud) don't support `tools`, so MiniMax tool calls come back as
        # unparsed XML in message content and the agent loop stalls. Pin routing
        # to tool-capable providers so we always get OpenAI-format tool_calls.
        provs = [p.strip() for p in os.getenv(
            "OPENROUTER_PROVIDERS", "minimax,together,parasail").split(",") if p.strip()]
        if provs:
            extra_body = {"provider": {
                "only": provs,
                "allow_fallbacks": True,                          # on a provider 429, try the next
                "sort": os.getenv("OPENROUTER_SORT", "throughput"),  # prefer least-congested provider
            }}
    else:  # custom — corporate hosted OpenAI-compatible endpoint
        base_url = os.getenv("CUSTOM_LLM_BASE_URL")
        api_key = os.getenv("CUSTOM_LLM_API_KEY")
        if not base_url or "<" in base_url:
            raise RuntimeError(
                "LLM_PROVIDER=custom requires CUSTOM_LLM_BASE_URL to point at the "
                "corporate gateway (…/v1). Refusing to start: with an empty value the "
                "client would silently fall back to https://api.openai.com and send "
                "the corporate key there."
            )
    kwargs: dict = dict(
        model=os.getenv("MODEL_NAME", "google/gemma-4"),
        openai_api_base=base_url,
        openai_api_key=api_key,
        max_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192")),
        # No client retries — we pace requests proactively instead (below).
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "0")),
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    # Proactive rate limiting: a token-bucket throttle paces every model call so
    # we stay UNDER the provider's limit instead of firing, hitting 429, and
    # backing off. Shared across the orchestrator + all subagents/threads in the
    # process. Set LLM_REQUESTS_PER_SECOND=0 to disable.
    rps = float(os.getenv("LLM_REQUESTS_PER_SECOND", "0.5"))
    if rps > 0:
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=rps,
            check_every_n_seconds=float(os.getenv("LLM_RATE_CHECK_SECONDS", "0.1")),
            max_bucket_size=int(os.getenv("LLM_MAX_BURST", "1")),
        )
    return _FixedToolIdModel(**kwargs)


_CHECKPOINT_CONNECTIONS: dict[str, sqlite3.Connection] = {}


def _checkpointer_for(agent_root: Path | None) -> SqliteSaver:
    """Persistent checkpointer so HITL `Command(resume=...)` can re-enter across docker runs."""
    if agent_root is not None:
        db_path = agent_root / "checkpoint.db"
    else:
        db_path = Path(WORK_ROOT) / "checkpoint.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(db_path)
    conn = _CHECKPOINT_CONNECTIONS.get(key)
    if conn is None:
        conn = sqlite3.connect(key, check_same_thread=False)
        _CHECKPOINT_CONNECTIONS[key] = conn
    return SqliteSaver(conn)


def build_agent(
    *,
    instructions_root: Path,
    skills_root: Path,
    business_rules: str,
    agent_root: Path | None = None,
    response_format: type | None = None,
):
    """Construct a fresh agent.

    Filesystem topology:
      /input/, /scratch/  → FilesystemBackend (real disk under WORK_ROOT, no shell)
      /instructions/      → FilesystemBackend (per-agent on-disk, read-only)
      /skills/            → FilesystemBackend (per-agent on-disk, read-only)
      /memories/          → StoreBackend (cross-thread persistent memory)
      anything else       → StateBackend (thread-scoped ephemeral)
    """
    model = _build_model()
    skills_root.mkdir(parents=True, exist_ok=True)

    # Inline the small, every-run instruction files into the system prompt instead
    # of making the agent fetch them with rate-limited tool calls each run. They
    # are static, so this is a free prebuild that also caches well as a prompt prefix.
    startup_parts: list[str] = []
    for fname in ("process.md", "tool_tips.md"):
        fpath = instructions_root / fname
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8").strip()
            if text:
                startup_parts.append(f"--- /instructions/{fname} ---\n{text}")
    startup_context = "\n\n".join(startup_parts)

    tools_path = agent_root / "custom_tools.py" if agent_root else None
    if tools_path and tools_path.exists():
        mod = _load_module(tools_path, "custom_tools")
        orchestrator_tools = [v for v in vars(mod).values()
                              if callable(v) and hasattr(v, "name") and hasattr(v, "invoke")]
    else:
        orchestrator_tools = list(_ORCHESTRATOR_TOOLS)

    # Plain FilesystemBackend, NOT LocalShellBackend: file operations only. No
    # backend in this composite implements SandboxBackendProtocol, so deepagents'
    # `execute` tool can never run shell commands — the model gets no shell.
    local_fs = FilesystemBackend(root_dir=WORK_ROOT, virtual_mode=True)
    instructions_fs = FilesystemBackend(root_dir=str(instructions_root), virtual_mode=True)
    skills_fs = FilesystemBackend(root_dir=str(skills_root), virtual_mode=True)

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/input/":        local_fs,
            "/scratch/":      local_fs,
            "/instructions/": instructions_fs,
            "/skills/":       skills_fs,
            "/memories/":     StoreBackend(),
        },
    )

    permissions = [
        FilesystemPermission(operations=["write"], paths=["/instructions/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/input/**"], mode="deny"),
    ]

    interrupt_on = None
    if HITL_ENABLED:
        interrupt_on = {
            "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
            "edit_file":  {"allowed_decisions": ["approve", "edit", "reject"]},
        }

    agent = create_deep_agent(
        model=model,
        tools=orchestrator_tools,
        system_prompt=_build_prompt(business_rules, startup_context=startup_context),
        backend=backend,
        store=InMemoryStore(),
        checkpointer=_checkpointer_for(agent_root),
        subagents=_build_field_subagents(),
        permissions=permissions,
        skills=["/skills/"],
        response_format=response_format,
        interrupt_on=interrupt_on,
    )
    return agent, model


def _workspace_report_lines(work_root: Path) -> list[str]:
    lines: list[str] = []
    inp = work_root / "input"
    scr = work_root / "scratch"
    if inp.exists():
        files = [f for f in inp.iterdir() if f.is_file() and not f.name.startswith(".")]
        lines.append(f"- **Input files:** {len(files)}")
        for f in sorted(files, key=lambda p: p.name)[:12]:
            lines.append(f"  - `{f.name}` ({f.stat().st_size:,} bytes)")
        if len(files) > 12:
            lines.append(f"  - … and {len(files) - 12} more")
    else:
        lines.append("- **Input files:** (none)")
    if scr.exists():
        files = [f for f in scr.iterdir() if f.is_file() and not f.name.startswith(".")]
        total = sum(f.stat().st_size for f in files)
        lines.append(f"- **Scratch files:** {len(files)} (~{total:,} bytes total)")
        for f in sorted(files, key=lambda p: p.name)[:10]:
            lines.append(f"  - `{f.name}` ({f.stat().st_size:,} bytes)")
        if len(files) > 10:
            lines.append(f"  - … and {len(files) - 10} more")
    else:
        lines.append("- **Scratch files:** (none)")
    return lines


def _structured_to_dict(structured) -> dict:
    if structured is None:
        return {}
    if hasattr(structured, "model_dump"):
        return structured.model_dump()
    if isinstance(structured, dict):
        return structured
    return {}


def _interrupt_payload(result: dict) -> list[dict] | None:
    """Extract a JSON-serializable list of pending interrupts, or None."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    payload = []
    for itr in interrupts:
        value = getattr(itr, "value", itr)
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        try:
            json.dumps(value, ensure_ascii=False)
        except TypeError:
            value = str(value)
        payload.append({"value": value})
    return payload


def _build_result_markdown(
    *,
    agent_name: str,
    timestamp: str,
    input_display: str,
    thread_hint: str,
    workspace_lines: list[str],
    structured: dict,
    final_msg: str,
) -> str:
    parts = [
        "# Charter analysis\n\n",
        "## Work report\n\n",
        "| Field | Value |\n| --- | --- |\n",
        f"| Agent | `{agent_name}` |\n",
        f"| Finished | `{timestamp}` |\n",
        f"| Input | `{input_display}` |\n",
        f"| Thread | `{thread_hint}` |\n\n",
        "### Workspace\n\n",
        "\n".join(workspace_lines) + "\n\n",
        "## Structured output (JSON)\n\n```json\n",
        json.dumps(structured, ensure_ascii=False, indent=2),
        "\n```\n\n## Agent report\n\n",
        final_msg,
    ]
    if not final_msg.endswith("\n"):
        parts.append("\n")
    return "".join(parts)


@dataclass
class RunResult:
    output: str
    output_dir: Path
    timestamp: str
    structured: dict | None = None
    interrupt: list[dict] | None = None
    thread_id: str | None = None


@dataclass
class TestResult:
    sample_name: str
    output: str
    score: ScoringResult


@dataclass
class Agent:
    name: str
    root: Path

    @property
    def instructions_dir(self) -> Path:
        return self.root / "instructions"

    @property
    def memories_dir(self) -> Path:
        return self.root / "memories"

    @property
    def business_rules_path(self) -> Path:
        return self.root / "agent_init" / "buisness_rules.md"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    def _custom_schema(self):
        schema_path = self.root / "custom_schema.py"
        if schema_path.exists():
            mod = _load_module(schema_path, "custom_schema")
            for v in vars(mod).values():
                if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel:
                    return v
        return CharterStructuredOutput

    def _build(self, response_format: type | None = None):
        return build_agent(
            instructions_root=self.instructions_dir,
            skills_root=self.skills_dir,
            business_rules=self.business_rules_path.read_text(encoding="utf-8"),
            agent_root=self.root,
            response_format=response_format,
        )

    def _persist_run(
        self,
        *,
        result: dict,
        timestamp: str,
        thread_id: str,
        file_name: str,
    ) -> RunResult:
        structured_dict = _structured_to_dict(result.get("structured_response"))
        messages = result.get("messages") or []
        final_msg = messages[-1].content if messages else ""
        ws_lines = _workspace_report_lines(Path(WORK_ROOT))
        md = _build_result_markdown(
            agent_name=self.name,
            timestamp=timestamp,
            input_display=f"/input/{file_name}",
            thread_hint=thread_id,
            workspace_lines=ws_lines,
            structured=structured_dict,
            final_msg=str(final_msg),
        )

        out_dir = OUTPUT_ROOT / self.name / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.md").write_text(md, encoding="utf-8")
        (out_dir / "structured.json").write_text(
            json.dumps(structured_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        interrupt_payload = _interrupt_payload(result)
        if interrupt_payload is not None:
            (out_dir / "interrupt.json").write_text(
                json.dumps({"thread_id": thread_id, "interrupts": interrupt_payload},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n[HITL] Interrupt detected. Resume with: --thread-id {thread_id}")

        shutil.copytree(WORK_ROOT, out_dir / "work", dirs_exist_ok=True)
        print(f"\nResult (Markdown + JSON) and work saved to: {out_dir}")
        return RunResult(
            output=md, output_dir=out_dir, timestamp=timestamp,
            structured=structured_dict, interrupt=interrupt_payload, thread_id=thread_id,
        )

    def _finalize_structured(self, model, result: dict) -> dict:
        """Serialize the agent's FINAL REPORT into the output schema with one
        structured call. The agent loop runs without a response_format escape
        hatch (so a weak model can't 'finish' empty on turn 1).

        We feed ONLY the agent's own report — not raw /scratch dumps. The report
        already holds the structured findings its subagents produced (per the
        STEP 4 AGGREGATE prompt, with the full verbatim clause text). Re-extracting
        from scratch instead used to (a) truncate the relevant text at 60k chars
        and (b) mix in stale markdown from other runs, silently dropping fields
        (e.g. the meeting-protocol clause). The final report is small, this-run
        only, and lossless to serialize. Fall back to all assistant turns if the
        last message is unexpectedly thin."""
        schema = self._custom_schema()

        def _text(m) -> str:
            c = getattr(m, "content", "")
            if isinstance(c, list):
                c = " ".join(
                    str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in c
                )
            return str(c).strip()

        messages = result.get("messages") or []
        context = _text(messages[-1]) if messages else ""
        if len(context) < 200:  # report missing/too thin → use every assistant turn
            context = "\n\n".join(
                _text(m) for m in messages if isinstance(m, AIMessage) and _text(m)
            )
        context = context.strip()[:60000]
        if not context:
            print("[finalize] agent produced no analysis to serialize — empty result.")
            return {}
        try:
            structured = model.with_structured_output(schema).invoke([
                {"role": "system", "content": (
                    "You extract structured legal data from an LLC charter analysis. "
                    "Using ONLY the analysis below, fill every schema field. Quote exact "
                    "clause numbers/wording where present. Leave a field empty only if "
                    "the analysis truly does not cover it."
                )},
                {"role": "user", "content": context},
            ])
            return _structured_to_dict(structured)
        except Exception as e:
            print(f"[finalize] structured extraction failed: {e}")
            return {}

    def _run_deterministic(self) -> dict:
        """Code-driven, per-field extraction. Does not rely on the model to
        orchestrate/delegate, so results are stable even on weak tool-calling
        models. Returns a dict shaped like an agent result."""
        from extraction import markdown_for_inputs, extract_charter, render_markdown_report
        model = _build_model()
        md_texts = markdown_for_inputs(Path(WORK_ROOT) / "input")
        if not md_texts:
            return {"structured_response": {},
                    "messages": [AIMessage(content="No input markdown could be produced.")]}
        structured = extract_charter(model, md_texts)
        report = render_markdown_report(structured)
        return {"structured_response": structured, "messages": [AIMessage(content=report)]}

    def run(self, sample, thread_id: str | None = None) -> RunResult:
        src = Path(sample)
        prepare_run_workspace(src)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        thread_id = thread_id or f"{self.name}-run-{timestamp}"
        file_name = src.name if src.is_file() else "<all files>"
        mode = os.getenv("EXTRACTION_MODE", "agent").lower()

        print(f"[{self.name}] [{timestamp}] Running on: {sample} (mode={mode})")
        print(f"Budget: {MAX_SUBAGENTS} subagents, {MAX_TOKENS} tokens; HITL={HITL_ENABLED}")
        print("-" * 60)

        if mode == "agent":
            config = get_run_config(
                {"configurable": {"thread_id": thread_id}},
                run_id=str(uuid.uuid4()),
            )
            # PREBUILD: convert the PDF to markdown up front (deterministic, no LLM)
            # and hand the agent the section outline + the ready text path, so it
            # doesn't burn rate-limited turns on read_pdf / list_sections.
            preload_note = ""
            if src.is_file() and src.suffix.lower() == ".pdf":
                try:
                    read_pdf.func(f"/input/{file_name}")  # → /scratch/<stem>.md (+ persistent cache)
                    stem = Path(file_name).stem
                    outline = list_sections.func(f"/scratch/{stem}.md")
                    preload_note = (
                        f"\n\nThe document text is ALREADY extracted to /scratch/{stem}.md "
                        f"(do NOT call read_pdf). Use read_section on it for full sections. "
                        f"Section outline:\n{outline}"
                    )
                except Exception as e:
                    print(f"[prebuild] pre-staging skipped: {e}")
            user_message = (
                f"Analyze the document at: /input/{file_name}\n"
                "Follow the PRELOADED INSTRUCTIONS in your system prompt and any matching /skills/*/SKILL.md. "
                "Delegate each OUTPUT SCHEMA field to the matching named subagent via task(). "
                "Verify every subagent result before proceeding."
                + preload_note
            )
            # Build WITHOUT response_format. Passing the schema makes it a callable
            # "finish" tool available from turn 1; weak models (deepseek-v4-flash,
            # minimax-m3) call it immediately with empty fields and stop — 0 reads,
            # 0 subagents (confirmed via LangSmith trace). With no escape hatch the
            # orchestrator must use real tools to make progress.
            agent, model = self._build(response_format=None)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=config,
            )
            # Serialize the agent's gathered findings into the schema in one
            # structured-output call, AFTER it has actually read + delegated.
            result["structured_response"] = self._finalize_structured(model, result)
        else:  # deterministic (default) — stable extraction, no model orchestration
            result = self._run_deterministic()

        final_msg = result["messages"][-1].content if result.get("messages") else "No output"

        print("\n" + "=" * 60)
        print("FINAL OUTPUT:")
        print("=" * 60)
        print(final_msg)

        return self._persist_run(
            result=result, timestamp=timestamp, thread_id=thread_id, file_name=file_name,
        )

    def resume(self, thread_id: str, decisions: list[dict]) -> RunResult:
        """Resume an interrupted HITL run with a list of decisions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = get_run_config(
            {"configurable": {"thread_id": thread_id}},
            run_id=str(uuid.uuid4()),
        )
        agent, _model = self._build(response_format=self._custom_schema())

        print(f"[{self.name}] [{timestamp}] Resuming thread: {thread_id}")
        print("-" * 60)

        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
        )
        final_msg = result["messages"][-1].content if result.get("messages") else "No output"
        print("\n" + "=" * 60)
        print("FINAL OUTPUT (resume):")
        print("=" * 60)
        print(final_msg)

        return self._persist_run(
            result=result, timestamp=timestamp, thread_id=thread_id, file_name="<resume>",
        )

    def train(self, samples: list) -> list:
        from training import load_sample, train_on_sample, format_training_report
        agent, model = self._build(response_format=self._custom_schema())
        results = []
        for s in samples:
            sample_obj = load_sample(s)
            print(f"\n[{self.name}] Training on sample: {sample_obj.name}")
            print(f"Input files: {len(sample_obj.input_files)}")
            print("-" * 60)
            results.append(train_on_sample(
                agent,
                model,
                sample_obj,
                instructions_dir=self.instructions_dir,
                skills_dir=self.skills_dir,
            ))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_md = format_training_report(results, timestamp=timestamp)
        out_dir = OUTPUT_ROOT / self.name / "training" / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "training_report.md"
        report_path.write_text(report_md, encoding="utf-8")
        print(f"\nTraining report saved to: {report_path}")
        return results

    def test(self, samples: list) -> list:
        from training import load_sample, run_agent_on_input, score_output
        agent, model = self._build(response_format=self._custom_schema())
        results = []
        for s in samples:
            sample_obj = load_sample(s)
            print(f"\n[{self.name}] Testing on sample: {sample_obj.name}")
            print("-" * 60)
            os.environ["ACTIVE_SAMPLE"] = sample_obj.name
            try:
                output, _ = run_agent_on_input(agent, sample_obj)
            finally:
                os.environ.pop("ACTIVE_SAMPLE", None)
            score = score_output(model, output, sample_obj.expected_output)
            print(f"  Score: {score.overall:.2f}")
            results.append(TestResult(sample_name=sample_obj.name, output=output, score=score))
        return results


def create_agent(name: str, business_rules,
                 process=None, tool_tips=None,
                 overwrite: bool = False) -> Agent:
    """Create a new named agent on disk under AGENTS_ROOT/<name>/.

    business_rules / process / tool_tips: each may be a file path or raw string content.
    """
    root = AGENTS_ROOT / name
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Agent already exists: {root}. Pass overwrite=True to recreate.")
        shutil.rmtree(root)

    (root / "instructions").mkdir(parents=True)
    (root / "skills").mkdir(parents=True)
    (root / "agent_init").mkdir(parents=True)

    rules_text = _load_text(business_rules) if business_rules is not None else \
                 DEFAULT_BUSINESS_RULES.read_text(encoding="utf-8")
    (root / "agent_init" / "buisness_rules.md").write_text(rules_text, encoding="utf-8")
    (root / "instructions" / "process.md").write_text(_load_text(process), encoding="utf-8")
    (root / "instructions" / "tool_tips.md").write_text(_load_text(tool_tips), encoding="utf-8")

    print(f"Created agent '{name}' at {root}")
    return Agent(name=name, root=root)


def load_agent(name: str) -> Agent:
    root = AGENTS_ROOT / name
    if not root.exists():
        raise FileNotFoundError(f"Agent not found: {root}")
    return Agent(name=name, root=root)


def cmd_create(args):
    create_agent(
        name=args.name,
        business_rules=args.business_rules,
        process=args.process,
        tool_tips=args.tool_tips,
        overwrite=args.overwrite,
    )


def cmd_run(args):
    load_agent(args.name).run(args.input)


def cmd_resume(args):
    decisions_raw = args.decisions
    if decisions_raw and Path(decisions_raw).is_file():
        decisions = json.loads(Path(decisions_raw).read_text(encoding="utf-8"))
    else:
        decisions = json.loads(decisions_raw)
    if isinstance(decisions, dict):
        decisions = [decisions]
    load_agent(args.name).resume(args.thread_id, decisions)


def cmd_train(args):
    from training import format_training_report
    results = load_agent(args.name).train(args.samples)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(format_training_report(results, timestamp=timestamp))


def cmd_test(args):
    results = load_agent(args.name).test(args.samples)
    print("\n" + "=" * 60)
    print("TEST REPORT:")
    print("=" * 60)
    for r in results:
        print(f"[{r.sample_name}] score={r.score.overall:.2f}  {r.score.reasoning}")
        for f in r.score.per_field:
            icon = {"correct": "OK ", "not_significant_error": "EXT", "significant_error": "ERR"}.get(f.status, "?  ")
            print(f"  {icon} {f.field}: {f.reasoning[:120]}")


def main():
    parser = argparse.ArgumentParser(description="Bank Process Deep Agent")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("create", help="Create a new named agent")
    p.add_argument("--name", required=True)
    p.add_argument("--business-rules", default=None, help="Path or inline text")
    p.add_argument("--process", default=None)
    p.add_argument("--tool-tips", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("run", help="Run agent on a single input file")
    p.add_argument("--name", required=True)
    p.add_argument("--input", required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="Resume a paused HITL run")
    p.add_argument("--name", required=True)
    p.add_argument("--thread-id", required=True)
    p.add_argument("--decisions", required=True,
                   help="JSON dict/list of decisions, or path to a JSON file")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("train", help="Train agent on one or more samples")
    p.add_argument("--name", required=True)
    p.add_argument("--samples", required=True, nargs="+")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("test", help="Score agent on samples (no state changes)")
    p.add_argument("--name", required=True)
    p.add_argument("--samples", required=True, nargs="+")
    p.set_defaults(func=cmd_test)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    try:
        args.func(args)
    finally:
        flush_tracing()  # ensure queued traces upload before the process exits


if __name__ == "__main__":
    main()
