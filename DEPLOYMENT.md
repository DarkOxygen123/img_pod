# Deployment Guide - New GPU Infrastructure

## Pod Configuration

### 1. Interface (CPU Router)
- **Pod ID**: i8yfozejwi4lyk
- **SSH**: `ssh root@213.173.105.83 -p 58672 -i ~/.ssh/id_ed25519`
- **URL**: https://i8yfozejwi4lyk-8000.proxy.runpod.net
- **Type**: CPU only

### 2. Selfie Feature Worker (VQA + Face Detection)
- **Pod ID**: wp6is8takig1z0
- **SSH**: `ssh root@195.26.233.53 -p 23674 -i ~/.ssh/id_ed25519`
- **URL**: https://wp6is8takig1z0-8003.proxy.runpod.net
- **GPU**: RTX 6000 Ada (48GB VRAM)
- **Models**: Qwen2-VL-7B (~16GB), YOLOv8-Face (~500MB)

### 3. Profile Worker (Image Generation)
- **Pod ID**: ika5ni96asnpts
- **SSH**: `ssh root@103.196.86.90 -p 15610 -i ~/.ssh/id_ed25519`
- **URL**: https://ika5ni96asnpts-8003.proxy.runpod.net
- **GPU**: RTX 4090 (24GB VRAM)
- **Models**: Z-Image Turbo (~3GB)

### 4. LLM Worker
- **Pod ID**: cf0pc97a1e9lki
- **SSH**: `ssh root@82.221.170.242 -p 45164 -i ~/.ssh/id_ed25519`
- **URL**: https://cf0pc97a1e9lki-8002.proxy.runpod.net
- **GPU**: RTX 4090 (24GB VRAM)

---

## Deployment Instructions

### Initial Setup (First Time Only)

On **each pod**, run:
```bash
# Clone repository
cd /workspace
git clone https://github.com/DarkOxygen123/img_pod.git
cd img_pod
```

### Deploy Services

#### 1. Deploy Interface
```bash
# SSH to interface pod
ssh root@213.173.105.83 -p 58672 -i ~/.ssh/id_ed25519

# Run deployment
cd /workspace/img_pod
bash deploy_interface.sh
```

#### 2. Deploy Selfie Feature Worker
```bash
# SSH to selfie feature pod
ssh root@195.26.233.53 -p 23674 -i ~/.ssh/id_ed25519

# Run deployment (will download Qwen2-VL-7B ~16GB, takes 5-10 min first time)
cd /workspace/img_pod
bash deploy_selfie_feature.sh
```

#### 3. Deploy Profile Worker
```bash
# SSH to profile worker pod
ssh root@103.196.86.90 -p 15610 -i ~/.ssh/id_ed25519

# Run deployment
cd /workspace/img_pod
bash deploy_profile.sh
```

---

## Scaling Instructions

### Add More Selfie Feature Workers
1. Create new pod with 48GB VRAM
2. Deploy using `deploy_selfie_feature.sh`
3. Update `shared/settings.py`:
   ```python
   selfie_feature_worker_urls: List[HttpUrl] = [
       "https://wp6is8takig1z0-8003.proxy.runpod.net",
       "https://NEW_POD_ID-8003.proxy.runpod.net"  # Add here
   ]
   ```
4. Redeploy interface: `bash deploy_interface.sh`

### Add More Profile Workers
1. Create new pod with 24GB VRAM
2. Deploy using `deploy_profile.sh`
3. Update `shared/settings.py`:
   ```python
   profile_worker_urls: List[HttpUrl] = [
       "https://ika5ni96asnpts-8003.proxy.runpod.net",
       "https://NEW_POD_ID-8003.proxy.runpod.net"  # Add here
   ]
   ```
4. Redeploy interface: `bash deploy_interface.sh`

---

## New Features

### 1. **Qwen2-VL-7B Model** (upgraded from 2B)
   - 3.5x larger model for better accuracy
   - Better understanding of hair textures, accessories, facial features
   - Minimal speed impact (~12s vs ~10s on large images)

### 2. **YOLOv8-Face Detection**
   - Robust face detection (handles angles, occlusions)
   - Fallback to OpenCV if YOLOv8 fails
   - Detects multiple faces and picks largest

### 3. **Graceful Error Handling**
   - **No face detected**: Uses full image with quality penalty
   - **Multiple faces**: Picks largest face with quality penalty
   - **Poor quality**: Only rejects if quality < 0.2 (was 0.35)
   - No more hard rejections for edge cases

### 4. **Load Balancing**
   - Round-robin distribution across workers
   - Easily add more workers by updating settings
   - Automatic failover (handled by interface retry logic)

---

## Testing

After deployment, test the pipeline:
```bash
# From local machine
cd /Users/uthamkumar/Uthamkumar/SUKO/Sukofinal
.venv/bin/python testing/profile_create_and_save.py
```

Expected improvements:
- ✅ selfie_3 (no face) should now work with full image
- ✅ selfie_4, selfie_6 (multiple faces) should pick largest face
- ✅ Better feature extraction with 7B model
- ⚠️ selfie_2 timeout may still occur if image is huge (90s limit)

---

## Monitoring

### Check Service Logs
```bash
# On each pod
tail -f /tmp/selfie-feature-8003.log  # Selfie feature
tail -f /tmp/profile-8003.log         # Profile worker
tail -f /tmp/interface-8000.log       # Interface
```

### Health Checks
```bash
curl https://wp6is8takig1z0-8003.proxy.runpod.net/healthz  # Selfie feature
curl https://ika5ni96asnpts-8003.proxy.runpod.net/healthz  # Profile worker
curl https://i8yfozejwi4lyk-8000.proxy.runpod.net/healthz  # Interface
```

---

## Troubleshooting

### Service won't start
```bash
# Check if port is in use
netstat -tlnp | grep :8003

# Kill existing process
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill -9

# Check logs for errors
tail -n 100 /tmp/selfie-feature-8003.log
```

### Out of memory
```bash
# Check GPU memory
nvidia-smi

# If Qwen2-VL-7B won't fit, downgrade to 2B in selfie_feature_worker/main.py:
# VQA_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
```

### YOLOv8-Face not loading
- Service will automatically fallback to OpenCV
- Check logs for "yolov8_face_load_failed_will_use_opencv_fallback"
- OpenCV still works, just less robust for difficult angles
