"""GigaChat (Sber) embeddings adapter.

GigaChat is NOT OpenAI-compatible. This adapter:
  * performs the GigaChat OAuth2 flow (Authorization key -> short-lived Bearer
    token) and caches/refreshes the token automatically;
  * calls POST {base}/embeddings with GigaChat's request/response schema;
  * batches inputs and retries on token expiry (401);
  * handles the Russian "Минцифры" root CA (custom CA bundle or, as a last
    resort, disabled verification).

Works for the public cloud and on-prem deployments (override URLs, or supply
a static GIGACHAT_ACCESS_TOKEN to skip OAuth). Requires the [llm] extra
(`pip install raglib[llm]`).
"""
from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from typing import List, Optional

try:
    import requests
except ImportError:  # pragma: no cover - exercised only without the extra
    requests = None  # type: ignore[assignment]

DEFAULT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_MODEL = "Embeddings"           # also available: "EmbeddingsGigaR"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"    # or GIGACHAT_API_B2B / GIGACHAT_API_CORP


def _truthy(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class GigaChatEmbeddings:
    """Embeddings backed by the GigaChat /embeddings endpoint."""

    def __init__(
        self,
        *,
        auth_key: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        scope: str = DEFAULT_SCOPE,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        oauth_url: str = DEFAULT_OAUTH_URL,
        ca_bundle: Optional[str] = None,
        verify_ssl: bool = True,
        batch_size: int = 50,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if requests is None:
            raise RuntimeError(
                "GigaChatEmbeddings requires the 'requests' package: "
                "pip install raglib[llm]"
            )
        if not auth_key and client_id and client_secret:
            auth_key = base64.b64encode(
                f"{client_id}:{client_secret}".encode("utf-8")
            ).decode("ascii")
        self._auth_key = auth_key
        self._static_token = access_token  # bypasses OAuth when provided
        if not self._auth_key and not self._static_token:
            raise ValueError(
                "GigaChatEmbeddings needs either a static access_token, or an "
                "auth_key (base64 of 'client_id:client_secret'), or both "
                "client_id and client_secret."
            )

        self._scope = scope
        self.model_name = model
        self._base_url = base_url.rstrip("/")
        self._oauth_url = oauth_url
        self._batch_size = max(1, int(batch_size))
        self._timeout = timeout
        self._max_retries = max(1, int(max_retries))

        if ca_bundle:
            self._verify = ca_bundle
        elif not verify_ssl:
            self._verify = False
            try:  # silence the noisy per-request warning
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass
        else:
            self._verify = True

        self._session = requests.Session()
        self._token: Optional[str] = self._static_token
        self._token_exp_ms: float = float("inf") if self._static_token else 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "GigaChatEmbeddings":
        """Build from GIGACHAT_* environment variables (see the Deep agent
        deployment docs): CREDENTIALS/AUTH_KEY or CLIENT_ID+CLIENT_SECRET or
        ACCESS_TOKEN; SCOPE, EMBED_MODEL, BASE_URL, OAUTH_URL, CA_BUNDLE,
        VERIFY_SSL, EMBED_BATCH."""
        return cls(
            auth_key=os.environ.get("GIGACHAT_CREDENTIALS")
            or os.environ.get("GIGACHAT_AUTH_KEY"),
            client_id=os.environ.get("GIGACHAT_CLIENT_ID"),
            client_secret=os.environ.get("GIGACHAT_CLIENT_SECRET"),
            access_token=os.environ.get("GIGACHAT_ACCESS_TOKEN"),
            scope=os.environ.get("GIGACHAT_SCOPE", DEFAULT_SCOPE),
            model=os.environ.get("GIGACHAT_EMBED_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("GIGACHAT_BASE_URL", DEFAULT_BASE_URL),
            oauth_url=os.environ.get("GIGACHAT_OAUTH_URL", DEFAULT_OAUTH_URL),
            ca_bundle=os.environ.get("GIGACHAT_CA_BUNDLE"),
            verify_ssl=_truthy(os.environ.get("GIGACHAT_VERIFY_SSL"), True),
            batch_size=int(os.environ.get("GIGACHAT_EMBED_BATCH", "50")),
        )

    def _get_token(self, force: bool = False) -> str:
        if self._static_token:
            return self._static_token
        now_ms = time.time() * 1000.0
        if not force and self._token and now_ms < self._token_exp_ms - 60_000:
            return self._token
        with self._lock:
            now_ms = time.time() * 1000.0
            if not force and self._token and now_ms < self._token_exp_ms - 60_000:
                return self._token
            resp = self._session.post(
                self._oauth_url,
                headers={
                    "Authorization": f"Basic {self._auth_key}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": self._scope},
                verify=self._verify,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            self._token_exp_ms = float(
                payload.get("expires_at", (time.time() + 1500) * 1000.0)
            )
            return self._token

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            token = self._get_token(force=attempt > 0)
            resp = self._session.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"model": self.model_name, "input": texts},
                verify=self._verify,
                timeout=self._timeout,
            )
            if resp.status_code == 401 and not self._static_token:
                last_exc = requests.HTTPError("401 from GigaChat embeddings")
                continue
            resp.raise_for_status()
            data = resp.json().get("data", [])
            data.sort(key=lambda d: d.get("index", 0))  # preserve input order
            return [d["embedding"] for d in data]
        raise RuntimeError(
            f"GigaChat embeddings failed after {self._max_retries} attempts"
        ) from last_exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(self._embed_batch(texts[i:i + self._batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]
