"""OpenAI-compatible HTTP client — backs both SGLang (self-hosted, prod) and the
optional OpenAI dev provider. Implements non-streaming, streaming, and
function-calling against ``POST {base_url}/chat/completions``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

from app.core.errors import UpstreamError
from app.core.llm.base import (
    ChatMessage,
    Lane,
    LLMResponse,
    LLMToolCall,
    LLMToolSpec,
    LLMUsage,
)


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        lane_models: dict[Lane, str],
        provider: str = "sglang",
        default_temperature: float = 0.1,
        default_max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or "EMPTY"
        self._lane_models = lane_models
        self._temperature = default_temperature
        self._max_tokens = default_max_tokens
        self._timeout = timeout
        self._client = None  # lazy httpx.AsyncClient

    def _model(self, lane: Lane) -> str:
        return self._lane_models.get(lane) or self._lane_models.get(Lane.STANDARD) or "default"

    def _http(self):
        if self._client is None:
            import httpx  # lazy

            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    def _payload(self, messages, lane, temperature, max_tokens, **extra) -> dict:
        return {
            "model": self._model(lane),
            "messages": [m.to_openai() for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            **extra,
        }

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        lane: Lane = Lane.STANDARD,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self._payload(messages, lane, temperature, max_tokens, stream=False)
        try:
            r = await self._http().post(f"{self._base_url}/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"{self.provider} completion failed: {exc}") from exc
        return self._parse(data, lane)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        lane: Lane = Lane.STANDARD,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, lane, temperature, max_tokens, stream=True)
        try:
            async with self._http().stream(
                "POST", f"{self._base_url}/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                        delta = obj["choices"][0].get("delta", {}).get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"{self.provider} stream failed: {exc}") from exc

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[LLMToolSpec],
        *,
        lane: Lane = Lane.STANDARD,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self._payload(
            messages, lane, temperature, max_tokens, stream=False,
            tools=[t.to_openai() for t in tools], tool_choice=tool_choice,
        )
        try:
            r = await self._http().post(f"{self._base_url}/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"{self.provider} tool-call failed: {exc}") from exc
        return self._parse(data, lane)

    def _parse(self, data: dict, lane: Lane) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        tool_calls: list[LLMToolCall] = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(LLMToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args or {}))
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=msg.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop") or "stop",
            model=data.get("model", self._model(lane)),
            lane=lane,
            usage=LLMUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["OpenAICompatClient"]
