#!/bin/bash
# Deployment script for Profile Worker (Image Generation)
# Pod: ika5ni96asnpts (RTX 4090 24GB)

set -e

echo "========================================="
echo "Deploying Profile Worker"
echo "========================================="

cd /workspace/img_pod

echo "[1/5] Pulling latest code..."
git pull

echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r shared/requirements.txt

echo "[3/5] Stopping existing service..."
ps aux | grep "profile_worker.main:app" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || echo "No existing service found"
sleep 2

echo "[4/5] Starting profile_worker on port 8003..."
nohup python -m uvicorn services.profile_worker.main:app \
    --host 0.0.0.0 \
    --port 8003 \
    --workers 1 \
    > /tmp/profile-8003.log 2>&1 &

sleep 5

echo "[5/5] Service logs:"
tail -n 40 /tmp/profile-8003.log

echo ""
echo "========================================="
echo "Deployment complete!"
echo "Health check: curl http://localhost:8003/healthz"
echo "Logs: tail -f /tmp/profile-8003.log"
echo "========================================="
