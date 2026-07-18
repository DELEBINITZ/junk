"""Single shared Langfuse client — initialized once, reused everywhere.

One place owns the Langfuse client so the whole app (tracing callback, PROMPT MANAGEMENT
via client.get_prompt, scoring, datasets) talks to the SAME configured instance instead
of each caller constructing its own.

Contract for callers:
  - `get_langfuse_client()` returns the client, or **None** when Langfuse is not installed
    or not configured. Callers MUST handle None (no-op / local fallback) — never assume it.
  - `get_prompt(name, ...)` is a thin, None-safe wrapper over client.get_prompt so prompt
    lookups degrade to a local fallback when Langfuse is unavailable.

langfuse 4.x already keeps a client registry keyed by public_key (langfuse.get_client);
this module is the app-level front door to it: it pins construction to our Settings and
guarantees the tracing handler + prompt fetches resolve the identical client.
"""

from __future__ import annotations

import threading
from typing import Any

from security_intel.config import Settings
from security_intel.observability.logging import get_logger

logger = get_logger("langfuse")

# Import defensively — an unavailable/incompatible langfuse must never crash app boot.
try:
    from langfuse import Langfuse
except Exception:  # noqa: BLE001
    Langfuse = None  # type: ignore[assignment,misc]

_client: Any = None
_init_done = False
_lock = threading.Lock()


def init_langfuse_client(settings: Settings) -> Any:
    """Construct the shared client once from Settings. Idempotent (safe to call repeatedly).

    Call this at app startup with the real Settings so environment/release are pinned.
    Returns the client, or None when langfuse is missing or unconfigured.
    """
    global _client, _init_done
    if _init_done:
        return _client
    with _lock:
        if _init_done:
            return _client
        _init_done = True

        if Langfuse is None:
            logger.info("langfuse not installed — shared client disabled")
            _client = None
            return None
        if not settings.langfuse_host or not settings.langfuse_public_key:
            logger.info("Langfuse not configured (host/public key) — shared client disabled")
            _client = None
            return None

        try:
            _client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                environment=getattr(settings, "environment", "") or None,
                release=getattr(settings, "app_version", "") or None,
            )
            logger.info(f"Langfuse client initialized: {settings.langfuse_host}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Langfuse client init failed: {e}")
            _client = None
    return _client


def get_langfuse_client() -> Any:
    """Return the shared client (or None). Lazily inits from env-backed Settings on first
    use so scripts/workers that never call init_langfuse_client still get one client."""
    if not _init_done:
        init_langfuse_client(Settings())
    return _client


def get_prompt(
    name: str,
    *,
    label: str = "production",
    version: int | None = None,
    type: str = "text",
    fallback: Any = None,
    cache_ttl_seconds: int | None = None,
) -> Any:
    """None-safe prompt fetch for Langfuse PROMPT MANAGEMENT.

    Returns a Langfuse prompt object (``.compile(**vars)``, ``.prompt``,
    ``.get_langchain_prompt()``) or ``fallback`` when Langfuse is unavailable / the fetch
    fails — so a Langfuse outage degrades to the local prompt instead of breaking the app.
    Pass ``fallback`` (the hardcoded prompt text) so the caller always has something usable.
    """
    client = get_langfuse_client()
    if client is None:
        return fallback
    try:
        return client.get_prompt(
            name,
            label=label,
            version=version,
            type=type,
            fallback=fallback,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Langfuse get_prompt('{name}') failed, using fallback: {e}")
        return fallback


def flush() -> None:
    """Flush pending traces/observations (call on shutdown or after short-lived scripts)."""
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Langfuse flush failed: {e}")
