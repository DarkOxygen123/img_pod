#!/usr/bin/env bash
set -euo pipefail

APP_DIR=${APP_DIR:-/workspace/img_pod}
SERVICE=${1:-}
PORT=${2:-}

if [[ -z "$SERVICE" || -z "$PORT" ]]; then
  echo "Usage: run_service.sh <interface|llm|profile|text2img> <port>" >&2
  exit 1
fi

cd "$APP_DIR"
source .venv/bin/activate

case "$SERVICE" in
  interface) TARGET="services.interface.main:app" ;;
  llm) TARGET="services.llm_service.main:app" ;;
  profile) TARGET="services.profile_worker.main:app" ;;
  text2img) TARGET="services.text2img_worker.main:app" ;;
  *) echo "Unknown service: $SERVICE" >&2; exit 1 ;;
 esac

# If venv doesn't exist, create it.
if [[ ! -d "$APP_DIR/.venv" ]]; then
  python3 -m venv "$APP_DIR/.venv"
fi

source .venv/bin/activate

# Ensure deps for this service are installed (very lightweight check).
STAMP="$APP_DIR/.venv/.installed_${SERVICE}"
# Optional hard cache clear and reinstall on demand
if [[ "${CLEAR_CACHE:-}" == "1" ]]; then
  echo "CLEAR_CACHE=1: Purging caches and forcing reinstall..."
  rm -rf "$HOME/.cache/huggingface" "$HOME/.cache/torch" "$HOME/.cache/pip" || true
  rm -rf "/root/.cache/huggingface" "/root/.cache/torch" "/root/.cache/pip" || true
  rm -f "$STAMP" || true
fi
if [[ ! -f "$STAMP" ]]; then
  REQ="requirements.interface.txt"
  case "$SERVICE" in
    interface) REQ="requirements.interface.txt" ;;
    llm) REQ="requirements.llm.txt" ;;
    profile) REQ="requirements.profile.txt" ;;
    text2img) REQ="requirements.gpu.txt" ;;
  esac
  pip install --upgrade pip
  pip install -r "$REQ"
  date > "$STAMP"
fi

export APP_CONFIG_PATH=${APP_CONFIG_PATH:-"$APP_DIR/config/config.yaml"}

pkill -f "uvicorn ${TARGET}" || true
nohup uvicorn "$TARGET" --host 0.0.0.0 --port "$PORT" > "/tmp/${SERVICE}-${PORT}.log" 2>&1 &

echo "Started $SERVICE on port $PORT (logs: /tmp/${SERVICE}-${PORT}.log)"
