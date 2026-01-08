#!/usr/bin/env bash
set -euo pipefail

REPO_URL=${1:-}
APP_DIR=${2:-/app/Sukofinal}

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: bootstrap_pod.sh <git_repo_url> [app_dir]" >&2
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
pip install -r requirements.txt

echo "Done. Next: configure config/config.yaml (or env vars) and run uvicorn."
