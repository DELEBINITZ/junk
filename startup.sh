#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-contract-intelligence-poc}"
export COMPOSE_PROJECT_NAME
CLEAN_START=false
WITH_OLLAMA=false
HOST_OLLAMA=false
SKIP_OLLAMA_PULL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN_START=true
      ;;
    --with-ollama)
      WITH_OLLAMA=true
      ;;
    --host-ollama|--use-host-ollama)
      HOST_OLLAMA=true
      ;;
    --skip-ollama-pull)
      SKIP_OLLAMA_PULL=true
      ;;
    --ollama-model)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--ollama-model requires a model name, for example qwen2.5:7b-instruct"
        exit 1
      fi
      export OLLAMA_MODEL="$1"
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./startup.sh [--clean] [--with-ollama] [--host-ollama] [--skip-ollama-pull] [--ollama-model MODEL]"
      exit 1
      ;;
  esac
  shift
done

if [[ "$WITH_OLLAMA" == "true" && "$HOST_OLLAMA" == "true" ]]; then
  echo "Choose either --with-ollama for Dockerized Ollama or --host-ollama for your local Ollama server, not both."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not on PATH."
  echo "Install Docker Desktop, then rerun ./startup.sh."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed, but the Docker daemon is not running."
  echo "Start Docker Desktop, wait until it is ready, then rerun ./startup.sh."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required."
  echo "Install/update Docker Desktop, then rerun ./startup.sh."
  exit 1
fi

if [[ "$WITH_OLLAMA" == "true" ]]; then
  export COMPOSE_PROFILES="${COMPOSE_PROFILES:-llm}"
  export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
fi

if [[ "$HOST_OLLAMA" == "true" ]]; then
  export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
fi

if [[ "$CLEAN_START" == "true" ]]; then
  echo "Stopping existing containers and removing volumes..."
  docker compose down -v
fi

echo "Building and starting Enterprise Contract Intelligence PoC..."
docker compose up -d --build

if [[ "$WITH_OLLAMA" == "true" && "$SKIP_OLLAMA_PULL" == "false" ]]; then
  MODEL_TO_PULL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
  echo "Waiting for Dockerized Ollama service..."
  for attempt in $(seq 1 60); do
    if docker compose exec -T ollama ollama list >/tmp/contract-intelligence-ollama-list.txt 2>/dev/null; then
      break
    fi

    if [[ "$attempt" == "60" ]]; then
      echo "Ollama did not become ready in time."
      echo "Recent Ollama logs:"
      docker compose logs --tail=80 ollama
      exit 1
    fi

    sleep 2
  done

  if awk 'NR > 1 {print $1}' /tmp/contract-intelligence-ollama-list.txt | grep -Fxq "$MODEL_TO_PULL"; then
    echo "Ollama model already available in Docker volume: $MODEL_TO_PULL"
  else
    echo "Pulling Ollama model into Docker volume: $MODEL_TO_PULL"
    echo "This can take several minutes the first time."
    if ! docker compose exec -T ollama ollama pull "$MODEL_TO_PULL"; then
      echo "Model pull failed. You can retry later with:"
      echo "  docker compose exec ollama ollama pull $MODEL_TO_PULL"
      echo "Or use an existing host model with:"
      echo "  ./startup.sh --host-ollama --ollama-model $MODEL_TO_PULL"
      exit 1
    fi
  fi
  rm -f /tmp/contract-intelligence-ollama-list.txt
fi

echo "Waiting for API health check on http://127.0.0.1:${PORT}/health ..."
for attempt in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/contract-intelligence-health.json 2>/dev/null; then
    echo "API is healthy:"
    cat /tmp/contract-intelligence-health.json
    echo
    rm -f /tmp/contract-intelligence-health.json
    break
  fi

  if [[ "$attempt" == "60" ]]; then
    echo "API did not become healthy in time."
    echo "Recent API logs:"
    docker compose logs --tail=80 api
    exit 1
  fi

  sleep 2
done

echo "Waiting for enterprise frontend health check on http://127.0.0.1:${FRONTEND_PORT}/health ..."
for attempt in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${FRONTEND_PORT}/health" >/tmp/contract-intelligence-frontend-health.json 2>/dev/null; then
    echo "Enterprise frontend is healthy:"
    cat /tmp/contract-intelligence-frontend-health.json
    echo
    rm -f /tmp/contract-intelligence-frontend-health.json
    break
  fi

  if [[ "$attempt" == "60" ]]; then
    echo "Frontend did not become healthy in time."
    echo "Recent frontend logs:"
    docker compose logs --tail=80 frontend
    exit 1
  fi

  sleep 2
done

echo
echo "Application started successfully."
echo "Enterprise UI: http://127.0.0.1:${FRONTEND_PORT}/"
echo "Swagger UI:    http://127.0.0.1:${PORT}/docs"
echo "API Health:    http://127.0.0.1:${PORT}/health"
echo "LLM Provider: ${LLM_PROVIDER:-deterministic}"
if [[ "${LLM_PROVIDER:-deterministic}" == "ollama" ]]; then
  echo "Ollama URL:   ${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
  echo "Ollama model: ${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
  if [[ "$WITH_OLLAMA" == "true" ]]; then
    echo "Ollama mode:  Dockerized Ollama"
  elif [[ "$HOST_OLLAMA" == "true" ]]; then
    echo "Ollama mode:  Host Ollama using your local model cache"
  fi
fi
echo
echo "Demo credentials:"
echo "  alice@techcorp.com / password123"
echo "  bob@techcorp.com / password123"
echo "  charlie@techcorp.com / password123"
echo "  diana@medicareplus.com / password123"
echo "  eve@medicareplus.com / password123"
echo
echo "Useful commands:"
echo "  docker compose logs -f api"
echo "  docker compose logs -f frontend"
echo "  ./startup.sh --with-ollama"
echo "  ./startup.sh --host-ollama --ollama-model ${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
echo "  docker compose ps"
echo "  docker compose down"
