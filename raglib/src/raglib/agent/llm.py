"""LLM seam for agentic search: a one-method protocol, no langchain in core.

``OpenAICompatChatLLM`` talks to any OpenAI-compatible /chat/completions
gateway (the corporate one in the target contour). ``MockLLM`` replays a
scripted sequence of responses for offline tests.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Union, runtime_checkable

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

Message = dict  # {"role": "...", "content": "..."}


@runtime_checkable
class ChatLLM(Protocol):
    def complete(self, messages: List[Message]) -> str:
        """One chat completion; returns the assistant message text."""
        ...


class OpenAICompatChatLLM:
    def __init__(self, *, base_url: str, api_key: str, model: str,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 timeout: float = 120.0, extra_headers: dict | None = None):
        if requests is None:
            raise RuntimeError(
                "OpenAICompatChatLLM requires 'requests': pip install raglib[llm]")
        if not base_url:
            raise ValueError("base_url is required (e.g. https://gateway.corp/v1)")
        self._base_url = base_url.rstrip("/")
        self.model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }
        self._session = requests.Session()

    def complete(self, messages: List[Message]) -> str:
        resp = self._session.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json={"model": self.model_name, "messages": messages,
                  "temperature": self._temperature, "max_tokens": self._max_tokens},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        return message.get("content") or ""


class LangChainChatLLM:
    """Adapt any LangChain chat model to raglib's ChatLLM protocol.

    Wraps a model exposing ``.invoke(messages) -> message`` (e.g.
    ``langchain_openai.ChatOpenAI``, or the custom ChatOpenAI subclass the
    Deep agent uses) so the SAME model that drives your LangChain / deepagents
    stack also drives agentic search. Duck-typed — importing raglib does not
    pull in langchain; you pass an already-built model in.

    Embeddings need no adapter: raglib's embeddings protocol
    (embed_documents / embed_query) is already satisfied by any LangChain
    embeddings object, so pass e.g. GigaChatEmbeddings / OpenAIEmbeddings
    straight into RagIndex.build(embeddings=...).
    """

    def __init__(self, model):
        if not hasattr(model, "invoke"):
            raise TypeError(
                "LangChainChatLLM expects a LangChain chat model with .invoke() "
                "(e.g. langchain_openai.ChatOpenAI(...)); got "
                f"{type(model).__name__}")
        self._model = model
        self.model_name = (getattr(model, "model_name", None)
                           or getattr(model, "model", None)
                           or type(model).__name__)

    @classmethod
    def from_openai(cls, *, model: str, base_url: Optional[str] = None,
                    api_key: Optional[str] = None, temperature: float = 0.0,
                    max_tokens: int = 2048, timeout: float = 180.0,
                    **kwargs) -> "LangChainChatLLM":
        """Build from the LangChain OpenAI adapter (``langchain_openai.ChatOpenAI``)
        in one call. Works against any OpenAI-compatible gateway via base_url
        (corporate endpoint, OpenRouter, vLLM). Extra kwargs pass through to
        ChatOpenAI. Requires ``langchain-openai`` (extra ``[langchain]``)."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "LangChainChatLLM.from_openai needs langchain-openai: "
                "pip install raglib[langchain]") from e
        return cls(ChatOpenAI(model=model, base_url=base_url, api_key=api_key,
                              temperature=temperature, max_tokens=max_tokens,
                              timeout=timeout, **kwargs))

    @staticmethod
    def _content_to_text(content) -> str:
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

    def complete(self, messages: List[Message]) -> str:
        # LangChain chat models accept OpenAI-style {"role","content"} dicts
        result = self._model.invoke(messages)
        return self._content_to_text(getattr(result, "content", result))


class MockLLM:
    """Scripted LLM for tests. Each response is a string or a callable
    (messages -> string), consumed in order. Records every call in .calls."""

    def __init__(self, responses: List[Union[str, Callable[[List[Message]], str]]]):
        self._responses = list(responses)
        self.calls: List[List[Message]] = []

    def complete(self, messages: List[Message]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("MockLLM: no scripted responses left")
        resp = self._responses.pop(0)
        return resp(messages) if callable(resp) else resp
