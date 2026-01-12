# Deployment Commands - Quick Reference

## Initial Setup (Run Once Per Pod)

### All Pods - Clone Repository
```bash
cd /workspace
git clone https://github.com/DarkOxygen123/img_pod.git
cd img_pod
```

---

## Deploy to Pods (Copy-Paste Ready)

### 1. Interface Pod (i8yfozejwi4lyk)
```bash
# SSH to interface pod
ssh root@213.173.105.83 -p 58672 -i ~/.ssh/id_ed25519

# Deploy (paste these commands)
cd /workspace/img_pod
git pull
bash deploy_interface.sh
```

### 2. Selfie Feature Pod (wp6is8takig1z0) - RTX 6000 Ada
```bash
# SSH to selfie feature pod
ssh root@195.26.233.53 -p 23674 -i ~/.ssh/id_ed25519

# Deploy (paste these commands)
# WARNING: First run downloads Qwen2-VL-7B (~16GB), takes 5-10 minutes
cd /workspace/img_pod
git pull
bash deploy_selfie_feature.sh
```

### 3. Profile Worker Pod (ika5ni96asnpts) - RTX 4090
```bash
# SSH to profile worker pod
ssh root@103.196.86.90 -p 13560 -i ~/.ssh/id_ed25519

# Deploy (paste these commands)
cd /workspace/img_pod
git pull
bash deploy_profile.sh
```

---

## Quick Health Checks

After deployment, verify all services:
```bash
# Check from local machine
curl https://i8yfozejwi4lyk-8000.proxy.runpod.net/healthz
curl https://wp6is8takig1z0-8003.proxy.runpod.net/healthz
curl https://ika5ni96asnpts-8003.proxy.runpod.net/healthz
```

---

## Test the Pipeline

```bash
# From local machine
cd /Users/uthamkumar/Uthamkumar/SUKO/Sukofinal
.venv/bin/python testing/profile_create_and_save.py
```

---

## What Changed - Summary

### ✅ **Better Models**
- **Qwen2-VL-7B** (was 2B): 3.5x larger, better feature extraction
- **YOLOv8-Face**: Robust face detection for difficult angles

### ✅ **Graceful Error Handling**
- **No face detected**: Now uses full image (quality penalty 0.3)
- **Multiple faces**: Picks largest face (quality penalty 0.7)
- **Poor quality**: Only rejects if < 0.2 (was 0.35)

### ✅ **Load Balancing & Scaling**
- Round-robin across multiple workers
- Add more workers by updating settings + redeploying interface
- Supports horizontal scaling

### ✅ **Edge Cases Fixed**
- selfie_3 (no face) → will work now
- selfie_4, selfie_6 (multiple faces) → picks largest
- Better handling of hats, glasses, masks

---

## Deployment Order (Important!)

1. **Selfie Feature** first (takes longest due to 16GB download)
2. **Profile Worker** second
3. **Interface** last (connects to workers)

---

## Monitoring

### Check Logs
```bash
# On selfie feature pod
tail -f /tmp/selfie-feature-8003.log

# On profile worker pod
tail -f /tmp/profile-8003.log

# On interface pod
tail -f /tmp/interface-8000.log
```

### GPU Memory
```bash
# On GPU pods
nvidia-smi
```

---

## Troubleshooting

### Service won't start
```bash
# Kill all uvicorn processes
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill -9

# Restart manually
cd /workspace/img_pod
bash deploy_selfie_feature.sh  # or deploy_profile.sh, deploy_interface.sh
```

### Out of Memory on Selfie Feature Pod
If 48GB isn't enough, you can downgrade to 2B model:
```bash
# Edit the file
nano /workspace/img_pod/services/selfie_feature_worker/main.py

# Change line 23 to:
VQA_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Redeploy
bash deploy_selfie_feature.sh
```

---

## Next Steps After Deployment

1. SSH to each pod and run deployment scripts
2. Verify health checks pass
3. Run test script from local machine
4. Check that previously failing images now work
5. Monitor logs for any errors

Full details: See [DEPLOYMENT.md](DEPLOYMENT.md)
