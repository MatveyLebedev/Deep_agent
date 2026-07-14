"""LLM seam for agentic search.

raglib drives agentic search through a LangChain chat model **directly**: any
object exposing ``.invoke(messages) -> AIMessage`` works —
``langchain_gigachat.GigaChat`` in the contour, ``langchain_openai.ChatOpenAI``,
or the custom subclass your stack already uses. Build the model and pass it
into ``RagIndex.agentic_search(llm=...)``; there is no adapter to construct.
LangChain chat models accept OpenAI-style ``{"role", "content"}`` dicts, which
is exactly what raglib sends.

``MockLLM`` is a scripted stand-in with the same ``.invoke`` contract for
offline tests.
"""
from __future__ import annotations

from typing import Any, Callable, List, Protocol, Union, runtime_checkable

Message = dict  # OpenAI-style {"role": "...", "content": "..."}; accepted by .invoke


@runtime_checkable
class ChatModel(Protocol):
    """Structural type of a LangChain chat model, as far as raglib uses it."""

    def invoke(self, messages: List[Message]) -> Any: ...


def content_to_text(content: Any) -> str:
    """Flatten a LangChain message ``.content`` to plain text.

    Most models return a string; some return a list of content blocks
    (``[{"type": "text", "text": ...}, ...]``) — join their text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some models return content blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


class _AIMessage:
    """Minimal AIMessage stand-in: just the ``.content`` attribute raglib reads."""

    def __init__(self, content: Any):
        self.content = content


class MockLLM:
    """Scripted LangChain-style chat model for tests. Each response is a string
    or a callable (messages -> string), consumed in order; ``.invoke`` returns
    an AIMessage-like object whose ``.content`` is that text (or, when
    ``as_blocks=True``, a one-element content-block list, to exercise the
    flattening path). Records every call in ``.calls``."""

    def __init__(self, responses: List[Union[str, Callable[[List[Message]], str]]],
                 *, as_blocks: bool = False, model_name: str = "mock-llm"):
        self._responses = list(responses)
        self._as_blocks = as_blocks
        self.model_name = model_name
        self.calls: List[List[Message]] = []

    def invoke(self, messages: List[Message]) -> _AIMessage:
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("MockLLM: no scripted responses left")
        resp = self._responses.pop(0)
        text = resp(messages) if callable(resp) else resp
        content = [{"type": "text", "text": text}] if self._as_blocks else text
        return _AIMessage(content)
