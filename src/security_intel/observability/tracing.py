"""Langfuse tracing for the LangGraph pipeline — optional observability.

Enable by setting LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY. If langfuse
is missing or unconfigured, everything here degrades to a no-op and the app runs unchanged.

Targets langfuse >= 3, where the CallbackHandler carries NO credentials and reads trace
attributes (user / session / name / tags) from per-run METADATA keys. Credentials and the
single shared client live in langfuse_client.py; this module only builds the callback
handler and attaches it (plus trace metadata) to a RunnableConfig.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from security_intel.config import Settings
from security_intel.observability.langfuse_client import init_langfuse_client
from security_intel.observability.logging import get_logger

logger = get_logger("tracing")

try:
    from langfuse.langchain import CallbackHandler
except Exception:  # noqa: BLE001 — langfuse optional / not installed
    CallbackHandler = None
    logger.warning("langfuse.langchain unavailable — tracing disabled; app boots without it.")


def get_langfuse_handler(settings: Settings):
    """Return one reusable Langfuse callback handler, or None if unavailable/unconfigured."""
    if CallbackHandler is None or not settings.langfuse_host or not settings.langfuse_public_key:
        return None
    # The handler resolves its client via get_client(); construct the shared client first
    # or it silently "skips tracing — no client initialized".
    if init_langfuse_client(settings) is None:
        return None
    logger.info(f"Langfuse tracing enabled: {settings.langfuse_host}")
    return CallbackHandler(public_key=settings.langfuse_public_key)


def traced_config(
    base_config: RunnableConfig,
    langfuse_handler,
    trace_name: str = "",
    user_id: str = "",
    session_id: str = "",
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> RunnableConfig:
    """Attach the Langfuse handler + trace attributes to a RunnableConfig.

    langfuse reads trace grouping (user / session / name / tags) off these metadata keys.
    Returns ``base_config`` unchanged when tracing is disabled.
    """
    if not langfuse_handler:
        return base_config

    config = dict(base_config)  # preserve configurable / recursion_limit / etc.
    config["callbacks"] = [*(base_config.get("callbacks") or []), langfuse_handler]

    meta = dict(base_config.get("metadata") or {})
    if trace_name:
        meta["langfuse_trace_name"] = trace_name
    if user_id:
        meta["langfuse_user_id"] = user_id
    if session_id:
        meta["langfuse_session_id"] = session_id
    if tags:
        meta["langfuse_tags"] = list(tags)
    if metadata:
        meta.update(metadata)
    config["metadata"] = meta
    return config
