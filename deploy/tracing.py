"""Tracing abstraction: LangSmith, LangFuse, or none.

Controlled by TRACING_PROVIDER env (langsmith | langfuse | none).
"""
import os
from typing import Any

_CALLBACK_HANDLER: Any = None


def tracing_provider() -> str:
    return os.getenv("TRACING_PROVIDER", "langsmith").lower()


def _verify_langsmith() -> None:
    """Best-effort startup check so a bad key/endpoint/flag surfaces immediately
    instead of silently dropping every trace (LangSmith's background thread eats
    403s). Never raises; only prints a clear, actionable diagnostic."""
    flag = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").lower()
    if flag not in ("1", "true", "yes", "on"):
        print("[tracing] LangSmith selected but LANGCHAIN_TRACING_V2 is not 'true' "
              "— set LANGCHAIN_TRACING_V2=true or traces will be dropped.")
        return

    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not key:
        print("[tracing] LangSmith: no LANGCHAIN_API_KEY/LANGSMITH_API_KEY set "
              "— traces will be dropped.")
        return

    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "default"
    try:
        from langsmith import Client
        client = Client()
        # /info is public and does NOT validate the key; list_projects hits an
        # authed endpoint (GET /sessions) and is what actually fails on a bad key.
        next(iter(client.list_projects(limit=1)), None)
        print(f"[tracing] LangSmith active → project '{project}' @ {client.api_url}")
    except Exception as e:
        reason = (str(e).splitlines() or [type(e).__name__])[0]
        print(
            "[tracing] WARNING: LangSmith rejected the credentials — traces will NOT appear.\n"
            f"          reason: {reason}\n"
            "          fixes:\n"
            "            • generate a fresh key at https://smith.langchain.com "
            "(Settings → API Keys) and set LANGCHAIN_API_KEY\n"
            "            • EU workspace? also set "
            "LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com\n"
            "            • offline/closed network? set TRACING_PROVIDER=none"
        )


def setup_tracing() -> None:
    """Initialize tracing. LangSmith uses LANGCHAIN_* env vars; we verify them at
    startup so a bad key fails loudly. LangFuse registers a CallbackHandler."""
    global _CALLBACK_HANDLER
    provider = tracing_provider()
    if provider == "langfuse":
        from langfuse.langchain import CallbackHandler
        _CALLBACK_HANDLER = CallbackHandler()
    elif provider == "langsmith":
        _CALLBACK_HANDLER = None
        _verify_langsmith()
    else:
        _CALLBACK_HANDLER = None


def get_run_config(base_config: dict | None = None, run_id: str | None = None) -> dict:
    """Merge tracing callbacks and LangFuse trace_id metadata into a LangGraph config."""
    config: dict = dict(base_config or {})
    configurable = dict(config.get("configurable") or {})
    config["configurable"] = configurable

    if run_id:
        config["run_id"] = run_id
        if tracing_provider() == "langfuse":
            metadata = dict(config.get("metadata") or {})
            metadata["langfuse_trace_id"] = run_id
            config["metadata"] = metadata

    if _CALLBACK_HANDLER is not None:
        callbacks = list(config.get("callbacks") or [])
        if _CALLBACK_HANDLER not in callbacks:
            callbacks.append(_CALLBACK_HANDLER)
        config["callbacks"] = callbacks

    return config


def flush_tracing() -> None:
    """Block until queued traces are uploaded. LangSmith uploads on a background
    thread; a short-lived process (e.g. `docker compose run --rm`) can exit before
    that thread flushes, silently dropping traces. Call this before the process
    exits so every run shows up."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass
    if _CALLBACK_HANDLER is not None:  # LangFuse: flush its client buffer too
        try:
            _CALLBACK_HANDLER.flush()
        except Exception:
            pass
