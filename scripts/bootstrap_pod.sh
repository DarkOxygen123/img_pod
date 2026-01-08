#!/usr/bin/env bash
set -euo pipefail

REPO_URL=${1:-}
APP_DIR=${2:-/workspace/img_pod}
SERVICE=${3:-}

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: bootstrap_pod.sh <git_repo_url> [app_dir] [service: interface|llm|profile|text2img]" >&2
  exit 1
fi

mkdir -p "$(dirname "$APP_DIR")"

if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "Cloning repo into $APP_DIR..."
  git clone "$REPO_URL" "$APP_DIR"
else
  echo "Repo already exists at $APP_DIR"
fi

cd "$APP_DIR"

echo "Setting up venv..."
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

REQ="requirements.interface.txt"
if [[ -n "$SERVICE" ]]; then
  case "$SERVICE" in
    interface) REQ="requirements.interface.txt" ;;
    llm) REQ="requirements.llm.txt" ;;
    profile) REQ="requirements.profile.txt" ;;
    text2img) REQ="requirements.gpu.txt" ;;
    *) echo "Unknown service '$SERVICE'" >&2; exit 1 ;;
  esac
fi

pip install -r "$REQ"

echo "Done. Next: configure config/config.yaml (or env vars) and run uvicorn."
