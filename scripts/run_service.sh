#!/usr/bin/env bash
set -euo pipefail

APP_DIR=${APP_DIR:-/app/Sukofinal}
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

export APP_CONFIG_PATH=${APP_CONFIG_PATH:-"$APP_DIR/config/config.yaml"}

pkill -f "uvicorn ${TARGET}" || true
nohup uvicorn "$TARGET" --host 0.0.0.0 --port "$PORT" > "/tmp/${SERVICE}-${PORT}.log" 2>&1 &

echo "Started $SERVICE on port $PORT (logs: /tmp/${SERVICE}-${PORT}.log)"
