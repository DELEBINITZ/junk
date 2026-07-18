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

    callbacks = list(base_config.get("callbacks", []) or [])

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

    return RunnableConfig(
        configurable=base_config.get("configurable", {}),
        callbacks=callbacks,
        recursion_limit=base_config.get("recursion_limit"),
    )
