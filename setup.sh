#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${SETUP_MODE:-docker}"
PORT="${PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
CHECK_ONLY=false
DETACH=false
RUN_TESTS=false
HOST_OLLAMA=false
WITH_OLLAMA=false
STARTUP_ARGS=()

usage() {
  cat <<'EOF'
Enterprise Contract Intelligence setup helper

Usage:
  ./setup.sh [--docker|--local] [options]
  ./set-up.sh [--docker|--local] [options]

Recommended assignment review path:
  ./setup.sh --docker

Local developer path:
  ./setup.sh --local

Options:
  --docker              Validate Docker prerequisites and run Docker Compose.
  --local               Create/use .venv, install Python deps, run API + enterprise UI locally.
  --check               Only validate prerequisites; do not install/start.
  --detach              Local mode only: start local API/UI in background and exit.
  --run-tests           Local mode only: run pytest after dependency install.
  --clean               Docker mode: remove Docker volumes before starting.
  --with-ollama         Docker mode: start Dockerized Ollama and pull the configured model.
  --host-ollama         Use an already-running host Ollama server.
  --skip-ollama-pull    Docker mode: do not pull the Ollama model.
  --ollama-model MODEL  Ollama model name, for example qwen2.5:7b-instruct.
  -h, --help            Show this help.

URLs after startup:
  Enterprise UI: http://127.0.0.1:3000/
  API docs:      http://127.0.0.1:8000/docs
  API health:    http://127.0.0.1:8000/health
EOF
}

log() {
  printf '\n==> %s\n' "$1"
}

info() {
  printf '    %s\n' "$1"
}

fail() {
  printf '\nERROR: %s\n' "$1" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

install_hint() {
  local tool="$1"
  case "$tool" in
    docker)
      cat <<'EOF'
Please install Docker Desktop and start it:
  macOS/Windows: https://www.docker.com/products/docker-desktop/
  Linux: install Docker Engine + Docker Compose v2 for your distribution.
EOF
      ;;
    python3)
      cat <<'EOF'
Please install Python 3.11 or newer:
  macOS: brew install python
  Ubuntu/Debian: sudo apt-get install python3 python3-venv python3-pip
EOF
      ;;
    node)
      cat <<'EOF'
Please install Node.js 18 or newer:
  macOS: brew install node
  Ubuntu/Debian: install Node.js 20 LTS from NodeSource or your package manager.
EOF
      ;;
    curl)
      cat <<'EOF'
Please install curl:
  macOS: curl is normally preinstalled.
  Ubuntu/Debian: sudo apt-get install curl
EOF
      ;;
    ollama)
      cat <<'EOF'
Please install and start Ollama if you want host-ollama mode:
  https://ollama.com/download
Then run:
  ollama pull qwen2.5:7b-instruct
EOF
      ;;
  esac
}

require_command() {
  local tool="$1"
  if ! command_exists "$tool"; then
    install_hint "$tool"
    fail "$tool is required for this setup mode."
  fi
}

check_python_version() {
  require_command python3
  python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required. Current: " + sys.version.split()[0])
PY
}

check_node_version() {
  require_command node
  node -e "const major = Number(process.versions.node.split('.')[0]); if (major < 18) { console.error('Node.js 18+ is required. Current: ' + process.version); process.exit(1); }"
}

check_docker() {
  require_command docker
  if ! docker info >/dev/null 2>&1; then
    install_hint docker
    fail "Docker is installed, but the Docker daemon is not running. Start Docker Desktop and rerun setup."
  fi
  if ! docker compose version >/dev/null 2>&1; then
    fail "Docker Compose v2 is required. Update Docker Desktop or install the Docker Compose plugin."
  fi
}

check_corpus() {
  if [[ ! -d "Assignment_org" ]]; then
    fail "Missing Assignment_org contract corpus directory."
  fi
  if ! find Assignment_org -maxdepth 1 -name '*.txt' | grep -q .; then
    fail "Assignment_org exists, but no contract .txt files were found."
  fi
}

ensure_env_file() {
  if [[ ! -f ".env" && -f ".env.example" ]]; then
    cp .env.example .env
    info "Created .env from .env.example"
  fi
}

load_env_file() {
  if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi

  export APP_NAME="${APP_NAME:-Enterprise Contract Intelligence PoC}"
  export JWT_SECRET="${JWT_SECRET:-demo-local-secret-change-before-prod}"
  export JWT_ISSUER="${JWT_ISSUER:-contract-intelligence-poc}"
  export ACCESS_TOKEN_MINUTES="${ACCESS_TOKEN_MINUTES:-120}"
  export CONTRACT_CORPUS_DIR="${CONTRACT_CORPUS_DIR:-Assignment_org}"
  export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-hash}"
  export EMBEDDING_DIMENSIONS="${EMBEDDING_DIMENSIONS:-384}"
  export LLM_PROVIDER="${LLM_PROVIDER:-deterministic}"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
  export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
  export LOG_LEVEL="${LOG_LEVEL:-INFO}"
  export LOG_FORMAT="${LOG_FORMAT:-json}"
}

port_available() {
  local port="$1"
  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(1)
PY
}

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      info "Stopping previous local process $pid"
      kill "$pid" >/dev/null 2>&1 || true
      sleep 1
    fi
    rm -f "$pid_file"
  fi
}

wait_for_health() {
  local name="$1"
  local url="$2"
  local log_file="$3"

  for attempt in $(seq 1 60); do
    if curl -fsS "$url" >/tmp/contract-intelligence-setup-health.json 2>/dev/null; then
      info "$name is healthy: $(cat /tmp/contract-intelligence-setup-health.json)"
      rm -f /tmp/contract-intelligence-setup-health.json
      return 0
    fi

    if [[ "$attempt" == "60" ]]; then
      printf '\n%s did not become healthy in time.\n' "$name" >&2
      printf 'Recent log output from %s:\n' "$log_file" >&2
      tail -80 "$log_file" >&2 || true
      exit 1
    fi

    sleep 2
  done
}

check_host_ollama() {
  require_command curl
  local base_url="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
  if ! curl -fsS "${base_url}/api/tags" >/dev/null 2>&1; then
    install_hint ollama
    fail "Host Ollama is not reachable at ${base_url}. Start Ollama or use Docker mode with --with-ollama."
  fi
}

run_docker_setup() {
  log "Checking Docker prerequisites"
  check_docker
  check_corpus

  if [[ "$CHECK_ONLY" == "true" ]]; then
    info "Docker prerequisites look good."
    return 0
  fi

  log "Starting Docker deployment"
  exec ./startup.sh "${STARTUP_ARGS[@]}"
}

install_local_dependencies() {
  if [[ ! -d ".venv" ]]; then
    log "Creating local Python virtual environment"
    python3 -m venv .venv
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate

  log "Installing local Python dependencies"
  python -m pip install --upgrade pip
  python -m pip install -e ".[test]"
}

run_local_setup() {
  log "Checking local prerequisites"
  check_python_version
  check_node_version
  require_command curl
  check_corpus
  if [[ "$CHECK_ONLY" != "true" ]]; then
    ensure_env_file
  fi
  load_env_file

  if [[ "$HOST_OLLAMA" == "true" ]]; then
    export LLM_PROVIDER="ollama"
    check_host_ollama
  fi

  if [[ "$WITH_OLLAMA" == "true" ]]; then
    fail "--with-ollama is Docker-only. For local mode, start Ollama on your machine and use --host-ollama."
  fi

  if [[ "$CHECK_ONLY" == "true" ]]; then
    info "Local prerequisites look good."
    return 0
  fi

  install_local_dependencies

  if [[ "$RUN_TESTS" == "true" ]]; then
    log "Running pytest"
    pytest
  fi

  mkdir -p .run
  stop_pid_file .run/local-api.pid
  stop_pid_file .run/local-frontend.pid

  if ! port_available "$PORT"; then
    fail "Port ${PORT} is already in use. Stop the existing API or run: docker compose down"
  fi
  if ! port_available "$FRONTEND_PORT"; then
    fail "Port ${FRONTEND_PORT} is already in use. Stop the existing frontend or run: docker compose down"
  fi

  log "Starting local API and enterprise UI"
  python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" >.run/api.log 2>&1 &
  API_PID="$!"
  echo "$API_PID" >.run/local-api.pid

  (
    cd frontend
    API_BASE_URL="http://127.0.0.1:${PORT}" FRONTEND_PORT="$FRONTEND_PORT" node server.js
  ) >.run/frontend.log 2>&1 &
  FRONTEND_PID="$!"
  echo "$FRONTEND_PID" >.run/local-frontend.pid

  wait_for_health "API" "http://127.0.0.1:${PORT}/health" ".run/api.log"
  wait_for_health "Enterprise frontend" "http://127.0.0.1:${FRONTEND_PORT}/health" ".run/frontend.log"

  cat <<EOF

Application started successfully.
Enterprise UI: http://127.0.0.1:${FRONTEND_PORT}/
Swagger UI:    http://127.0.0.1:${PORT}/docs
API Health:    http://127.0.0.1:${PORT}/health

Demo credentials:
  bob@techcorp.com / password123
  alice@techcorp.com / password123
  charlie@techcorp.com / password123
  diana@medicareplus.com / password123
  eve@medicareplus.com / password123

Logs:
  .run/api.log
  .run/frontend.log
EOF

  if [[ "$DETACH" == "true" ]]; then
    info "Local services are running in the background. Stop them with: kill \$(cat .run/local-api.pid .run/local-frontend.pid)"
    return 0
  fi

  cleanup() {
    printf '\nStopping local services...\n'
    kill "$API_PID" "$FRONTEND_PID" >/dev/null 2>&1 || true
    rm -f .run/local-api.pid .run/local-frontend.pid
  }
  trap cleanup INT TERM EXIT

  log "Local services are running. Press Ctrl-C to stop."
  wait "$API_PID" "$FRONTEND_PID"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker)
      MODE="docker"
      ;;
    --local)
      MODE="local"
      ;;
    --check|--no-start)
      CHECK_ONLY=true
      ;;
    --detach)
      DETACH=true
      ;;
    --run-tests)
      RUN_TESTS=true
      ;;
    --clean|--skip-ollama-pull)
      STARTUP_ARGS+=("$1")
      ;;
    --with-ollama)
      WITH_OLLAMA=true
      STARTUP_ARGS+=("$1")
      ;;
    --host-ollama|--use-host-ollama)
      HOST_OLLAMA=true
      STARTUP_ARGS+=("$1")
      ;;
    --ollama-model)
      shift
      if [[ $# -eq 0 ]]; then
        fail "--ollama-model requires a model name, for example qwen2.5:7b-instruct"
      fi
      export OLLAMA_MODEL="$1"
      STARTUP_ARGS+=("--ollama-model" "$1")
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      fail "Unknown option: $1"
      ;;
  esac
  shift
done

case "$MODE" in
  docker)
    run_docker_setup
    ;;
  local)
    run_local_setup
    ;;
  *)
    usage
    fail "Unknown setup mode: $MODE"
    ;;
esac
