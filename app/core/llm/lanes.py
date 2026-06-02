"""Lane router: maps fast/standard/deep to LLM clients (plan §14).

The planner emits a `lane`; this returns the client for it. With the default
deterministic provider all lanes share one client (local dev). With
LLM_PROVIDER=sglang each lane maps to its model (Qwen 7B / Qwen 72B /
DeepSeek-V3.1). The deep lane is opt-in and should be quota-capped in admission
control (plan §13.3).
"""

from __future__ import annotations

from app.config import settings
from app.llm.client import LLMClient, get_llm_client


class Lane:
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


def _build_clients() -> dict[str, LLMClient]:
    provider = settings.llm_provider.lower()
    if provider == "sglang":
        from app.core.llm.sglang_client import SGLangClient

        return {
            Lane.FAST: SGLangClient(settings.sglang_base_url, settings.sglang_model_fast),
            Lane.STANDARD: SGLangClient(settings.sglang_base_url, settings.sglang_model_standard),
            Lane.DEEP: SGLangClient(settings.sglang_base_url, settings.sglang_model_deep),
        }
    # deterministic / ollama / vllm: one client serves every lane
    client = get_llm_client()
    return {Lane.FAST: client, Lane.STANDARD: client, Lane.DEEP: client}


class LaneRouter:
    def __init__(self, clients: dict[str, LLMClient] | None = None):
        self._clients = clients or _build_clients()

    def client_for(self, lane: str) -> LLMClient:
        return self._clients.get(lane, self._clients[Lane.STANDARD])
