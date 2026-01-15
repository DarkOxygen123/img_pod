#!/bin/bash
# Deployment script for Scenes Worker (Z-Image Turbo with NSFW moderation)
# Requires RTX 4090 (24GB VRAM) or similar

set -e

echo "========================================="
echo "Deploying Scenes Worker"
echo "========================================="

# Navigate to workspace
cd /workspace/img_pod

echo "[1/6] Clearing old logs and caches..."
rm -f /tmp/scenes-*.log 2>/dev/null || true
rm -rf /root/.cache/pip /root/.cache/torch 2>/dev/null || true
find /workspace -type f -name '*.pyc' -delete 2>/dev/null || true
find /workspace -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Pull latest code
echo "[2/6] Pulling latest code..."
git pull

# Install/update Python dependencies
echo "[3/6] Installing dependencies..."
pip install --upgrade pip
pip install -r services/scenes_worker/requirements.txt

# Kill existing service
echo "[4/6] Stopping existing service..."
ps aux | grep "scenes_worker.main:app" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || echo "No existing service found"
sleep 2

# Start service
echo "[5/6] Starting scenes_worker on port 8007..."
nohup python -m uvicorn services.scenes_worker.main:app \
    --host 0.0.0.0 \
    --port 8007 \
    --workers 1 \
    > /tmp/scenes-8007.log 2>&1 &

sleep 5

# Check logs
echo "[6/6] Service logs:"
tail -n 40 /tmp/scenes-8007.log

echo ""
echo "========================================="
echo "Deployment complete!"
echo "Health check: curl http://localhost:8007/healthz"
echo "Logs: tail -f /tmp/scenes-8007.log"
echo "========================================="
