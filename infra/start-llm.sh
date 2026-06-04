#!/usr/bin/env bash
# =============================================================================
#  start-llm.sh — boots the main answer LLM: Qwen2.5-72B-Instruct on vLLM
#  (OpenAI-compatible, :8000). This is the latency-critical service — EVERY agent
#  step (route, plan, tool-call, critic, answer, summarize) hits it, because vLLM
#  serves ONE model and all lanes collapse onto it.
#
#  Runs on its own GPU box (72B needs ~2x80GB fp16, or 4x48GB, or quantized).
#  Run:
#     export HF_TOKEN=hf_xxxxx        # Qwen is ungated, but keep it set for cache auth
#     ./start-llm.sh                  # up (default)
#     ./start-llm.sh down             # stop + remove
# =============================================================================
set -euo pipefail

HF_TOKEN="${HF_TOKEN:-}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
LLM_SERVED="${LLM_SERVED:-Qwen/Qwen2.5-72B-Instruct}"   # MUST equal app VLLM_MODEL
LLM_PORT="${LLM_PORT:-8000}"                            # MUST match VLLM_BASE_URL port
TP_SIZE="${TP_SIZE:-2}"                                 # tensor-parallel = #GPUs (2x80GB fp16)
MAX_LEN="${MAX_LEN:-32768}"
MEM_UTIL="${MEM_UTIL:-0.92}"
# Quantization (optional, to fit smaller VRAM): set LLM_QUANT=fp8, or awq + an -AWQ model.
LLM_QUANT="${LLM_QUANT:-none}"

if [[ "${1:-up}" == "down" ]]; then
  docker rm -f vllm-llm 2>/dev/null || true
  echo "vllm-llm stopped."
  exit 0
fi

command -v docker >/dev/null || { echo "FATAL: docker not found"; exit 1; }
mkdir -p "$HF_CACHE"

ARGS=(--model "$LLM_MODEL" --served-model-name "$LLM_SERVED"
      --tensor-parallel-size "$TP_SIZE" --max-model-len "$MAX_LEN"
      --gpu-memory-utilization "$MEM_UTIL"
      # PROMPT CACHING: vLLM auto-caches the KV of shared PREFIXES across requests.
      # The agent keeps the persona/rules prefix byte-stable + first, so every turn in
      # a session (and each replan round) reuses that cached prefix -> big TTFT/throughput
      # win. No cache_control needed (that's an Anthropic concept; vLLM is automatic).
      --enable-prefix-caching)
[[ "$LLM_QUANT" != "none" ]] && ARGS+=(--quantization "$LLM_QUANT")

echo "Starting vllm-llm: $LLM_MODEL  TP=$TP_SIZE  :$LLM_PORT  (quant=$LLM_QUANT)"
docker rm -f vllm-llm >/dev/null 2>&1 || true
docker run -d --name vllm-llm --restart unless-stopped \
  --gpus all --shm-size 32g -p "${LLM_PORT}:8000" \
  -v "$HF_CACHE":/root/.cache/huggingface -e HF_TOKEN="$HF_TOKEN" \
  "$VLLM_IMAGE" "${ARGS[@]}"

printf "  waiting for vllm-llm (72B load is slow — minutes)"
for _ in $(seq 1 120); do
  if curl -fsS "http://localhost:${LLM_PORT}/health" >/dev/null 2>&1; then echo " UP"; break; fi
  printf "."; sleep 5
done

cat <<EOF

=== smoke test ===
  curl -s localhost:${LLM_PORT}/v1/chat/completions -H 'content-type: application/json' \\
    -d '{"model":"${LLM_SERVED}","messages":[{"role":"user","content":"say hi"}],"max_tokens":16}'

Point the app at this host:
  LLM_PROVIDER=vllm   VLLM_BASE_URL=http://<this-host>:${LLM_PORT}/v1   VLLM_MODEL=${LLM_SERVED}
Stop:  ./start-llm.sh down
EOF
