#!/bin/bash
# Deployment script for 1:1 Chat Worker (Z-Image Turbo for contextual chat images)
# Requires RTX 4090 (24GB VRAM) or similar

set -e

echo "========================================="
echo "Deploying 1:1 Chat Worker"
echo "========================================="

# Navigate to workspace
cd /workspace/img_pod

# Pull latest code
echo "[1/5] Pulling latest code..."
git pull

# Install/update Python dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r services/chat1to1_worker/requirements.txt

# Kill existing service
echo "[3/5] Stopping existing service..."
ps aux | grep "chat1to1_worker.main:app" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || echo "No existing service found"
sleep 2

# Start service
echo "[4/5] Starting chat1to1_worker on port 8005..."
nohup python -m uvicorn services.chat1to1_worker.main:app \
    --host 0.0.0.0 \
    --port 8005 \
    --workers 1 \
    > /tmp/chat1to1-8005.log 2>&1 &

sleep 5

# Check logs
echo "[5/5] Service logs:"
tail -n 40 /tmp/chat1to1-8005.log

echo ""
echo "========================================="
echo "Deployment complete!"
echo "Health check: curl http://localhost:8005/healthz"
echo "Logs: tail -f /tmp/chat1to1-8005.log"
echo "========================================="
