#!/bin/bash
# Deployment script for LLM Service (Text Generation)
# Pod: cf0pc97a1e9lki (RTX 4090 24GB)

set -e

echo "========================================="
echo "Deploying LLM Service"
echo "========================================="

cd /workspace/img_pod

echo "[1/5] Pulling latest code..."
git pull

echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.llm.txt

echo "[3/5] Stopping existing service..."
ps aux | grep "llm_service.main:app" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || echo "No existing service found"
sleep 2

echo "[4/5] Starting llm_service on port 8002..."
nohup python -m uvicorn services.llm_service.main:app \
    --host 0.0.0.0 \
    --port 8002 \
    --workers 1 \
    > /tmp/llm-8002.log 2>&1 &

sleep 5

echo "[5/5] Service logs:"
tail -n 40 /tmp/llm-8002.log

echo ""
echo "========================================="
echo "Deployment complete!"
echo "Health check: curl http://localhost:8002/healthz"
echo "Logs: tail -f /tmp/llm-8002.log"
echo "========================================="
