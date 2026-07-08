"""Embeddings via any OpenAI-compatible /embeddings endpoint (corporate
gateway, vLLM, OpenRouter, ...). Requires the [llm] extra."""
from __future__ import annotations

from typing import List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


class OpenAICompatEmbeddings:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int = 64,
        timeout: float = 60.0,
        extra_headers: Optional[dict] = None,
    ) -> None:
        if requests is None:
            raise RuntimeError(
                "OpenAICompatEmbeddings requires 'requests': pip install raglib[llm]"
            )
        if not base_url:
            raise ValueError("base_url is required (e.g. https://gateway.corp/v1)")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model_name = model
        self._batch_size = max(1, int(batch_size))
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }
        self._session = requests.Session()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._session.post(
            f"{self._base_url}/embeddings",
            headers=self._headers,
            json={"model": self.model_name, "input": texts},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        data.sort(key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(self._embed_batch(texts[i:i + self._batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]
