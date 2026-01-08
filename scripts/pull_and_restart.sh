#!/usr/bin/env bash
set -euo pipefail

SERVICE_PATH=${1:-services/interface}
PORT=${2:-8000}

cd /app

echo "Pulling latest code..."
git pull --rebase

echo "Restarting service at ${SERVICE_PATH} on port ${PORT}..."
# Example using uvicorn; replace with your supervisor if needed
pkill -f "uvicorn ${SERVICE_PATH}.main:app" || true
nohup uvicorn ${SERVICE_PATH}.main:app --host 0.0.0.0 --port ${PORT} > /tmp/${PORT}.log 2>&1 &

echo "Started pid $(pgrep -f "uvicorn ${SERVICE_PATH}.main:app")"
