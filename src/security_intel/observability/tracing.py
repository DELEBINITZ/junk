"""Langfuse integration for LLM observability and agent tracing.

Traces every LLM call, tool invocation, and agent decision through Langfuse.
Enable by setting LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY in env.
"""

from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from langchain_core.runnables import RunnableConfig

from security_intel.config import Settings
from security_intel.observability.logging import get_logger

logger = get_logger("tracing")


def get_langfuse_handler(settings: Settings) -> LangfuseCallbackHandler | None:
    """Create Langfuse callback handler if configured."""
    if not settings.langfuse_host or not settings.langfuse_public_key:
        return None

    try:
        handler = LangfuseCallbackHandler(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
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

    This makes every LLM call and tool use within the graph automatically traced.
    """
    if not langfuse_handler:
        return base_config

    callbacks = list(base_config.get("callbacks", []) or [])

    # Create a trace-scoped handler
    trace_handler = langfuse_handler.get_trace_handler(
        name=trace_name or "orchestrator",
        user_id=user_id,
        session_id=session_id,
        metadata=metadata or {},
    )
    callbacks.append(trace_handler)

    return RunnableConfig(
        configurable=base_config.get("configurable", {}),
        callbacks=callbacks,
        recursion_limit=base_config.get("recursion_limit"),
    )
