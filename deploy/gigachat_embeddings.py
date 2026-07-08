"""GigaChat embeddings adapter for LangChain.

GigaChat (Sber) is NOT OpenAI-compatible, so `OpenAIEmbeddings` cannot talk to
it. This module implements a drop-in `langchain_core.embeddings.Embeddings`
subclass that:

  * performs the GigaChat OAuth2 flow (Authorization key -> short-lived
    Bearer access token) and caches/refreshes the token automatically;
  * calls POST {base}/embeddings with GigaChat's request/response schema;
  * batches inputs and retries once on token expiry (401);
  * handles the Russian "Минцифры" root CA (custom CA bundle or, as a last
    resort, disabled verification).

It works for both the public cloud endpoints and an on-prem GigaChat
deployment (override the URLs, or supply a static GIGACHAT_ACCESS_TOKEN to
skip OAuth entirely).

Configuration is via environment variables (see `from_env`) or constructor
arguments.
"""
from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

DEFAULT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_MODEL = "Embeddings"           # also available: "EmbeddingsGigaR"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"    # or GIGACHAT_API_B2B / GIGACHAT_API_CORP


def _truthy(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class GigaChatEmbeddings(Embeddings):
    """LangChain Embeddings backed by the GigaChat /embeddings endpoint."""

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
        # --- credentials ---------------------------------------------------
        # auth_key is the base64(client_id:client_secret) string shown in the
        # GigaChat cabinet. If only client_id/secret are given, build it.
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
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._oauth_url = oauth_url
        self._batch_size = max(1, int(batch_size))
        self._timeout = timeout
        self._max_retries = max(1, int(max_retries))

        # --- TLS -----------------------------------------------------------
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

        # --- token cache ---------------------------------------------------
        self._token: Optional[str] = self._static_token
        self._token_exp_ms: float = float("inf") if self._static_token else 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ env
    @classmethod
    def from_env(cls) -> "GigaChatEmbeddings":
        """Build from environment variables.

        GIGACHAT_CREDENTIALS / GIGACHAT_AUTH_KEY  base64(client_id:client_secret)
        GIGACHAT_CLIENT_ID, GIGACHAT_CLIENT_SECRET  (alternative to the above)
        GIGACHAT_ACCESS_TOKEN     static Bearer token (skips OAuth; e.g. on-prem)
        GIGACHAT_SCOPE            default GIGACHAT_API_PERS
        GIGACHAT_EMBED_MODEL      default "Embeddings"
        GIGACHAT_BASE_URL         default cloud /api/v1
        GIGACHAT_OAUTH_URL        default cloud oauth endpoint
        GIGACHAT_CA_BUNDLE        path to the Минцифры root CA (.pem)
        GIGACHAT_VERIFY_SSL       "false" to disable verification (insecure)
        GIGACHAT_EMBED_BATCH      inputs per request (default 50)
        """
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

    # ---------------------------------------------------------------- token
    def _get_token(self, force: bool = False) -> str:
        if self._static_token:
            return self._static_token
        now_ms = time.time() * 1000.0
        # refresh ~60s before expiry
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
            # expires_at is epoch milliseconds; default to +25 min if missing
            self._token_exp_ms = float(
                payload.get("expires_at", (time.time() + 1500) * 1000.0)
            )
            return self._token

    # ----------------------------------------------------------- embeddings
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
                json={"model": self._model, "input": texts},
                verify=self._verify,
                timeout=self._timeout,
            )
            if resp.status_code == 401 and not self._static_token:
                # token likely expired/revoked — force-refresh and retry
                last_exc = requests.HTTPError("401 from GigaChat embeddings")
                continue
            resp.raise_for_status()
            data = resp.json().get("data", [])
            # preserve input order via the returned index
            data.sort(key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]
        raise RuntimeError(
            f"GigaChat embeddings failed after {self._max_retries} attempts"
        ) from last_exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(self._embed_batch(texts[i : i + self._batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]
