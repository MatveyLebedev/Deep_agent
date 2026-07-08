"""Opt-in strict egress guard for closed-network deployments.

NETWORK_GUARD=strict makes this process refuse ANY outbound connection except
to the endpoints explicitly configured via environment variables (the corporate
LLM gateway and GigaChat), plus loopback. Everything else — library telemetry,
accidental defaults to public clouds (api.openai.com, openrouter.ai,
huggingface.co, LangSmith/LangFuse) — fails with BlockedEgressError BEFORE any
byte (headers, keys, document text) leaves the process.

Implementation: wraps socket.getaddrinfo, which every hostname-based connection
in the HTTP stacks used here (requests, httpx, urllib) passes through. The
check runs before DNS resolution, so a blocked host is never even looked up.
Direct-to-IP connections are checked against the same allowlist (the IP must
then be what an allowed URL is configured with).

This is not a substitute for the network perimeter — it is a second,
in-process line of defense that also protects internet-connected build/test
hosts, where a misconfigured variable would otherwise leak keys or document
text to a public endpoint.
"""
import os
import socket
from urllib.parse import urlparse

# Mirrors the defaults in gigachat_embeddings.py (plain strings here so the
# guard has zero imports beyond the stdlib and installs before anything else).
_GIGACHAT_DEFAULT_OAUTH = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
_GIGACHAT_DEFAULT_BASE = "https://gigachat.devices.sberbank.ru/api/v1"

# "" covers getaddrinfo(None, port) used for local binds.
_LOOPBACK = {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}

_installed = False


class BlockedEgressError(OSError):
    """A connection was attempted to a host outside the configured allowlist."""


def _host_of(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlparse(url.strip()).hostname
    except ValueError:
        return None
    return host.lower() if host else None


def allowed_hosts() -> set[str]:
    """Hosts derived from the ACTIVE provider configuration, plus explicit extras.

    Only endpoints the operator deliberately configured contribute — e.g. with
    EMBED_PROVIDER=gigachat the OpenAI-compatible embeddings default is not in
    the set, so a stray fallback to openrouter.ai can never connect. Tracing
    hosts are never derived: in strict mode tracing must stay off (or its host
    be added to NETWORK_EXTRA_ALLOWED_HOSTS on purpose).
    """
    hosts: set[str] = set()

    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    if provider == "custom":
        hosts.add(_host_of(os.getenv("CUSTOM_LLM_BASE_URL")))
    elif provider == "openai":
        hosts.add(_host_of(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    else:  # openrouter
        hosts.add(_host_of(os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")))

    if os.getenv("EMBED_PROVIDER", "openai").lower() == "gigachat":
        hosts.add(_host_of(os.getenv("GIGACHAT_BASE_URL", _GIGACHAT_DEFAULT_BASE)))
        hosts.add(_host_of(os.getenv("GIGACHAT_OAUTH_URL", _GIGACHAT_DEFAULT_OAUTH)))
    else:
        # Deliberately no default: an unset EMBED_API_BASE must not open
        # openrouter.ai in strict mode.
        hosts.add(_host_of(os.getenv("EMBED_API_BASE")))

    for extra in os.getenv("NETWORK_EXTRA_ALLOWED_HOSTS", "").split(","):
        extra = extra.strip().lower()
        if extra:
            hosts.add(extra)

    hosts.discard(None)
    return hosts


def install_guard() -> None:
    """Activate the guard when NETWORK_GUARD=strict (no-op otherwise)."""
    global _installed
    mode = os.getenv("NETWORK_GUARD", "off").strip().lower()
    if mode not in ("strict", "1", "true", "yes", "on") or _installed:
        return

    allowed = allowed_hosts()
    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        h = host.decode() if isinstance(host, (bytes, bytearray)) else (host or "")
        h = h.strip("[]").lower()
        if h not in _LOOPBACK and h not in allowed:
            raise BlockedEgressError(
                f"[netguard] blocked connection to '{h}:{port}' — NETWORK_GUARD=strict "
                f"allows only the configured endpoints {sorted(allowed)} and loopback. "
                "If this destination is intended, add it to NETWORK_EXTRA_ALLOWED_HOSTS."
            )
        return real_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = guarded_getaddrinfo
    _installed = True
    print(f"[netguard] strict egress guard active — allowed hosts: "
          f"{sorted(allowed) or '(none)'} + loopback")
