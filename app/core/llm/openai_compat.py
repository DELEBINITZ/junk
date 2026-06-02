"""OpenAI-compatible HTTP client — the REAL LLM transport.

"OpenAI-compatible" is the key idea: OpenAI's ``/chat/completions`` request and
response shape has become a de-facto industry standard, and many servers speak
it — including SGLang, our self-hosted production serving stack. So ONE client
class, pointed at different URLs, backs both prod (SGLang) and an optional
OpenAI dev provider. The agent code above never knows the difference.

This file implements the three generation modes from the LLMClient contract by
POSTing to ``{base_url}/chat/completions``:
  * complete            — ``stream:false`` -> parse one JSON response.
  * stream              — ``stream:true``  -> read server-sent-events (SSE): the
                          server pushes ``data: {...}`` lines, each carrying a
                          small "delta" of the answer, which we yield as tokens.
  * complete_with_tools — adds ``tools`` + ``tool_choice`` so the model can reply
                          with function calls instead of prose.

It also normalizes every reply into our ``LLMResponse`` (``_parse``) and turns
any transport/HTTP failure into an ``UpstreamError`` — so a flaky model server
surfaces as a clean, typed error rather than a raw httpx exception.
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
    """An ``LLMClient`` that talks HTTP to any OpenAI-compatible chat endpoint.
    Configured once (URL, key, and the lane->model map) and reused for the life
    of the process via a single pooled HTTP connection."""

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
        self._base_url = base_url.rstrip("/")          # normalize so f"{base}/path" never doubles "/"
        # Some self-hosted servers ignore auth but still require the header to be
        # present; "EMPTY" is the conventional placeholder when no key is set.
        self._api_key = api_key or "EMPTY"
        self._lane_models = lane_models                # which served model each lane maps to
        self._temperature = default_temperature
        self._max_tokens = default_max_tokens
        self._timeout = timeout
        self._client = None  # lazy httpx.AsyncClient — created on first use, see _http()

    def _model(self, lane: Lane) -> str:
        """Resolve a lane to its concrete model name, degrading gracefully: the
        lane's own model, else the STANDARD model, else the literal "default".
        This keeps a request working even if a lane was left unmapped in config."""
        return self._lane_models.get(lane) or self._lane_models.get(Lane.STANDARD) or "default"

    def _http(self):
        """Lazily build and cache the shared httpx AsyncClient.

        Lazy for two reasons: (1) httpx is only imported if a real provider is
        actually used (the deterministic default never pays for it); (2) one
        long-lived client reuses TCP/TLS connections across calls instead of
        reconnecting every request. The bearer token is set once as a default
        header here, so every request is authenticated automatically."""
        if self._client is None:
            import httpx  # lazy

            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    def _payload(self, messages, lane, temperature, max_tokens, **extra) -> dict:
        """Assemble the JSON body for /chat/completions. Resolves the lane to a
        model, converts each ChatMessage to its OpenAI dict, and applies the
        per-call temperature/max_tokens (falling back to the configured
        defaults when the caller passes None). ``**extra`` carries mode-specific
        keys like ``stream``, ``tools``, ``tool_choice``."""
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
        """One-shot completion: POST with ``stream:false`` and parse the single
        JSON reply. ``raise_for_status`` turns any non-2xx into an exception,
        which we re-raise as a typed UpstreamError so callers see one error
        shape regardless of whether the URL, network, or server was at fault."""
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
        """Streaming completion over server-sent events (SSE).

        With ``stream:true`` the server replies as a live text stream of lines,
        each shaped ``data: {json chunk}``, plus a final ``data: [DONE]`` sentinel.
        Each chunk carries a "delta" — the next slice of generated content. We
        parse line by line and yield only the content deltas, which the caller
        forwards to the UI as live tokens. The parsing rules below are pure SSE
        hygiene:
          * ignore blank lines and any line that is not a ``data:`` event;
          * stop on the ``[DONE]`` sentinel;
          * skip malformed/keepalive chunks rather than crashing the stream;
          * a delta can be empty (e.g. the opening role-only chunk) — skip those.
        """
        payload = self._payload(messages, lane, temperature, max_tokens, stream=True)
        try:
            async with self._http().stream(
                "POST", f"{self._base_url}/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()       # drop the "data:" prefix
                    if chunk == "[DONE]":          # end-of-stream sentinel
                        break
                    try:
                        obj = json.loads(chunk)
                        delta = obj["choices"][0].get("delta", {}).get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue                   # tolerate keepalives / partial frames
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
        """Function-calling completion: same POST as ``complete`` but with the
        advertised ``tools`` and a ``tool_choice`` policy ("auto" = let the model
        decide). The reply may contain ``tool_calls`` instead of text; ``_parse``
        extracts them. The specialist's LLM planner uses this on the FAST lane."""
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
        """Translate the raw provider JSON into our normalized LLMResponse.

        Defensive throughout (``or {}`` / ``or []``) because not every
        OpenAI-compatible server fills every field. The fiddly bit is tool-call
        arguments: the spec returns them as a JSON *string*, so we json-decode it
        back into a dict (falling back to ``{}`` if the model emitted malformed
        JSON) — giving the agent ready-to-use arguments."""
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        tool_calls: list[LLMToolCall] = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments")
            if isinstance(args, str):              # arguments arrive as a JSON string
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}                      # malformed -> empty args, never crash
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
        """Close the pooled HTTP client and drop it, releasing its connections.
        Guarded so it is safe to call even if no request was ever made (the lazy
        client was never created). Reset to None so a later call could re-open."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["OpenAICompatClient"]
