#!/bin/bash
# Deployment script for Interface (CPU Router)
# Pod: i8yfozejwi4lyk (CPU)

set -e

echo "========================================="
echo "Deploying Interface Service"
echo "========================================="

cd /workspace/img_pod

echo "[1/5] Pulling latest code..."
git pull

echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r shared/requirements.txt

echo "[3/5] Stopping existing service..."
ps aux | grep "interface.main:app" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || echo "No existing service found"
sleep 2

echo "[4/5] Starting interface on port 8000..."
nohup python -m uvicorn services.interface.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    > /tmp/interface-8000.log 2>&1 &

sleep 3

echo "[5/5] Service logs:"
tail -n 40 /tmp/interface-8000.log

echo ""
echo "========================================="
echo "Deployment complete!"
echo "Health check: curl http://localhost:8000/healthz"
echo "Logs: tail -f /tmp/interface-8000.log"
echo "========================================="
