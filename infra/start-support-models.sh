#!/usr/bin/env bash
# =============================================================================
#  start-support-models.sh  —  boots the 2 SUPPORT models for the platform:
#     1. Embedder     Qwen3-Embedding-4B   (TEI,  :8080, dim 2560)
#     2. Reranker     bge-reranker-v2-m3   (TEI,  :8081)
#
#  NO guard-model servers: injection + content safety run on the MAIN deployed
#  LLM (LLMJudgeGuard — 72B in prod, 32B in staging). The answer-LLM runs on a
#  SEPARATE box and is NOT started here.
#
#  Drop this file on the support-model server and run it:
#     export HF_TOKEN=hf_xxxxx            # OPTIONAL — both default models are ungated
#     export TEI_IMAGE=...                # optional override (default cuda-1.9 universal)
#     ./start-support-models.sh           # up   (default)
#     ./start-support-models.sh down      # stop + remove both
#
#  Re-running is safe: each container is removed and recreated.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config — every value is overridable via environment.
# ---------------------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:-}"
GPU_DEVICE="${GPU_DEVICE:-0}"                       # physical GPU index the GPU models share
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"    # model cache (persists downloads)

# TEI image: universal CUDA tag by default (works across recent arches; matches
# staging). Older per-arch tags still work via TEI_IMAGE override:
#   H100 hopper-1.6 | A100 1.6 | A10/A6000/3090 86-1.6 | L4/L40S/4090 89-1.6 | T4 turing-1.6
TEI_IMAGE="${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference:cuda-1.9}"

EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"   # dim 2560 — MUST match app EMBEDDING_DIM
EMBED_PORT="${EMBED_PORT:-8080}"
RERANK_MODEL="${RERANK_MODEL:-BAAI/bge-reranker-v2-m3}"
RERANK_PORT="${RERANK_PORT:-8081}"

GPUS_FLAG="\"device=${GPU_DEVICE}\""                   # docker --gpus single-device form

# ---------------------------------------------------------------------------
# down: stop + remove everything, then exit. (Also removes the guard containers
# from older versions of this stack, if any are still around.)
# ---------------------------------------------------------------------------
if [[ "${1:-up}" == "down" ]]; then
  echo "Stopping support models..."
  docker rm -f tei-embed tei-rerank prompt-guard llama-guard 2>/dev/null || true
  echo "Done."
  exit 0
fi

# ---------------------------------------------------------------------------
# Preflight.
# ---------------------------------------------------------------------------
command -v docker >/dev/null || { echo "FATAL: docker not found"; exit 1; }
# Both default models are UNGATED — HF_TOKEN only needed if you override with gated repos.
[[ -n "$HF_TOKEN" ]] || echo "NOTE: HF_TOKEN not set (fine for the ungated defaults)"
mkdir -p "$HF_CACHE"
recreate() { docker rm -f "$1" >/dev/null 2>&1 || true; }

echo "=== Support models -> GPU device $GPU_DEVICE, HF cache $HF_CACHE ==="

# ---------------------------------------------------------------------------
# 1. Embedder — TEI, Qwen3-Embedding-4B (dim 2560, last-token pooling).
# ---------------------------------------------------------------------------
echo "[1/2] tei-embed   ($EMBED_MODEL) :$EMBED_PORT"
recreate tei-embed
docker run -d --name tei-embed --restart unless-stopped \
  --gpus "$GPUS_FLAG" -p "${EMBED_PORT}:80" \
  -v "$HF_CACHE":/data -e HF_TOKEN="$HF_TOKEN" \
  "$TEI_IMAGE" --model-id "$EMBED_MODEL" --pooling last-token --max-client-batch-size 64

# ---------------------------------------------------------------------------
# 2. Reranker — TEI cross-encoder (/rerank).
# ---------------------------------------------------------------------------
echo "[2/2] tei-rerank  ($RERANK_MODEL) :$RERANK_PORT"
recreate tei-rerank
docker run -d --name tei-rerank --restart unless-stopped \
  --gpus "$GPUS_FLAG" -p "${RERANK_PORT}:80" \
  -v "$HF_CACHE":/data -e HF_TOKEN="$HF_TOKEN" \
  "$TEI_IMAGE" --model-id "$RERANK_MODEL"

# ---------------------------------------------------------------------------
# Wait for health (models download + load on first run — can take minutes).
# ---------------------------------------------------------------------------
wait_http() {  # name url
  printf "  waiting for %-13s" "$1"
  for _ in $(seq 1 90); do
    if curl -fsS "$2" >/dev/null 2>&1; then echo " UP"; return 0; fi
    sleep 5
  done
  echo " NOT healthy after 7.5m — check: docker logs $1"
  return 1
}
echo "=== health (first run pulls + loads models; be patient) ==="
wait_http tei-embed    "http://localhost:${EMBED_PORT}/health"   || true
wait_http tei-rerank   "http://localhost:${RERANK_PORT}/health"  || true

cat <<EOF

=== up. smoke-test (shapes the app actually uses) ===
  curl -s localhost:${EMBED_PORT}/embed -H 'content-type: application/json' \\
    -d '{"inputs":["hello"]}' | python3 -c "import sys,json;print('embed dim',len(json.load(sys.stdin)[0]))"   # expect 2560
  curl -s localhost:${RERANK_PORT}/rerank -H 'content-type: application/json' \\
    -d '{"query":"open ports","texts":["exposed port 22","a cake recipe"]}'

Point the app at this host:
  TEI_EMBED_URL=http://<this-host>:${EMBED_PORT}   EMBEDDING_DIM=2560
  TEI_RERANK_URL=http://<this-host>:${RERANK_PORT}
Stop all:  ./start-support-models.sh down
EOF
