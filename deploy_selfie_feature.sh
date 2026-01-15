#!/bin/bash
# Deployment script for Selfie Feature Worker (Qwen2-VL-7B + YOLOv8-Face)
# Pod: wp6is8takig1z0 (RTX 6000 Ada 48GB)

set -e

echo "========================================="
echo "Deploying Selfie Feature Worker"
echo "========================================="

# Navigate to workspace
cd /workspace/img_pod

echo "[1/7] Clearing old logs and caches..."
rm -f /tmp/selfie-*.log 2>/dev/null || true
rm -rf /root/.cache/pip /root/.cache/torch 2>/dev/null || true
find /workspace -type f -name '*.pyc' -delete 2>/dev/null || true
find /workspace -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Pull latest code
echo "[2/7] Pulling latest code..."
git pull

# Install/update Python dependencies
echo "[3/7] Installing dependencies..."
pip install --upgrade pip
pip install -r services/selfie_feature_worker/requirements.txt

# Download YOLOv8 model (if not exists)
echo "[4/7] Downloading YOLOv8 model..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLOv8 download initiated"

# Kill existing service
echo "[5/7] Stopping existing service..."
ps aux | grep "selfie_feature_worker.main:app" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || echo "No existing service found"
sleep 2

# Start service
echo "[5/6] Starting selfie_feature_worker on port 8003..."
nohup python -m uvicorn services.selfie_feature_worker.main:app \
    --host 0.0.0.0 \
    --port 8003 \
    --workers 1 \
    > /tmp/selfie-feature-8003.log 2>&1 &

sleep 5

# Check logs
echo "[6/6] Service logs:"
tail -n 40 /tmp/selfie-feature-8003.log

echo ""
echo "========================================="
echo "Deployment complete!"
echo "Health check: curl http://localhost:8003/healthz"
echo "Logs: tail -f /tmp/selfie-feature-8003.log"
echo "========================================="
