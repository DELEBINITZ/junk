"""Langfuse integration for LLM observability and agent tracing.

Traces every LLM call, tool invocation, and agent decision through Langfuse.
Enable by setting LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY in env.
"""

from __future__ import annotations

import os
import inspect

# CallbackHandler moved across langfuse majors: langfuse 2.x exposes it at
# `langfuse.callback`, langfuse >=3 at `langfuse.langchain`. Import defensively and
# NEVER let an unavailable/incompatible langfuse (or a missing langchain integration)
# crash app boot — tracing is optional observability, so degrade to "disabled".
try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler  # langfuse 2.x
except Exception:  # noqa: BLE001
    try:
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # langfuse >=3
    except Exception:  # noqa: BLE001
        LangfuseCallbackHandler = None

from langchain_core.runnables import RunnableConfig

from security_intel.config import Settings
from security_intel.observability.logging import get_logger

logger = get_logger("tracing")

if LangfuseCallbackHandler is None:
    logger.warning(
        "Langfuse CallbackHandler unavailable (langfuse not installed or version "
        "mismatch) — tracing disabled; the app boots without it."
    )

_HANDLER_PARAMS = (
    set(inspect.signature(LangfuseCallbackHandler.__init__).parameters.keys())
    if LangfuseCallbackHandler is not None else set()
)

# langfuse 2.x takes credentials + trace attributes (user_id/session_id/trace_name) in
# the CONSTRUCTOR. langfuse 3.x/4.x take only public_key and read those attributes from
# per-run METADATA keys (langfuse_user_id / langfuse_session_id / langfuse_trace_name).
# Detect which by signature so we adapt without pinning a version.
_CTOR_TAKES_CREDS = "secret_key" in _HANDLER_PARAMS


def _make_handler(**kwargs) -> LangfuseCallbackHandler:
    """Build handler with only params the installed version accepts."""
    filtered = {k: v for k, v in kwargs.items() if k in _HANDLER_PARAMS and v is not None}
    return LangfuseCallbackHandler(**filtered)


def get_langfuse_handler(settings: Settings) -> LangfuseCallbackHandler | None:
    """Create Langfuse callback handler if configured and available."""
    if LangfuseCallbackHandler is None or not settings.langfuse_host or not settings.langfuse_public_key:
        return None

    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)

    # langfuse 3.x/4.x: the CallbackHandler holds no credentials — it resolves its client
    # via get_client(public_key=…). Unless a Langfuse client for that key was constructed
    # (which registers it), the handler silently "skips tracing (no client initialized)".
    # So construct the client here. langfuse 2.x needs none (creds live on the handler).
    if not _CTOR_TAKES_CREDS:
        try:
            from langfuse import Langfuse

            Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Langfuse client init failed: {e}")
            return None

    try:
        handler = _make_handler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            url=settings.langfuse_host,
        )
        logger.info(f"Langfuse tracing enabled: {settings.langfuse_host}")
        return handler
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")
        return None


def traced_config(
    base_config: RunnableConfig,
    langfuse_handler: LangfuseCallbackHandler | None,
    trace_name: str = "",
    user_id: str = "",
    session_id: str = "",
    metadata: dict | None = None,
) -> RunnableConfig:
    """Enrich a RunnableConfig with Langfuse tracing callbacks.

    Creates a fresh handler per trace so each request gets its own trace in Langfuse.
    """
    if not langfuse_handler:
        return base_config

    # Preserve every existing key (configurable, recursion_limit, tags, …); only augment
    # callbacks + metadata. Rebuilding from scratch previously dropped fields.
    new_config = dict(base_config)
    callbacks = list(base_config.get("callbacks", []) or [])
    meta = dict(base_config.get("metadata", {}) or {})

    if _CTOR_TAKES_CREDS:
        # langfuse 2.x — a fresh per-trace handler carries the attributes in its constructor.
        try:
            trace_handler = _make_handler(
                public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
                host=os.environ.get("LANGFUSE_HOST", ""),
                url=os.environ.get("LANGFUSE_HOST", ""),
                trace_name=trace_name or "orchestrator",
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
            callbacks.append(trace_handler)
        except Exception as e:
            logger.warning(f"Failed to create trace handler: {e}")
            callbacks.append(langfuse_handler)
    else:
        # langfuse 3.x/4.x — reuse the shared handler; trace attributes travel as the
        # metadata keys the handler reads off each run.
        callbacks.append(langfuse_handler)
        if trace_name:
            meta["langfuse_trace_name"] = trace_name
        if user_id:
            meta["langfuse_user_id"] = user_id
        if session_id:
            meta["langfuse_session_id"] = session_id
        if metadata:
            meta.update(metadata)

    new_config["callbacks"] = callbacks
    new_config["metadata"] = meta
    return new_config
