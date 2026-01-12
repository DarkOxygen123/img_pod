# 🚀 SUKO Deployment Checklist - Ready to Deploy

**Status:** ✅ All fixes applied, ready for deployment  
**Date:** January 12, 2026

---

## ✅ Pre-Deployment Verification

### Files Fixed & Created
- [x] `config/config.yaml` - Production config created with correct pod URLs
- [x] `requirements.selfie_feature.txt` - Added sentencepiece, ultralytics
- [x] `requirements.profile.txt` - Fixed dependencies 
- [x] `requirements.llm.txt` - Added sentencepiece, updated versions
- [x] `deploy_profile.sh` - Fixed to use correct requirements
- [x] `deploy_llm.sh` - Created new deployment script
- [x] `FIXES_SUMMARY.md` - Complete documentation of all fixes
- [x] `DEPLOY_QUICK_START.md` - Updated with LLM service deployment
- [x] All deployment scripts made executable

### Issues Resolved
- [x] Missing `config.yaml` 
- [x] Missing `sentencepiece` dependency (critical for transformers)
- [x] Missing `ultralytics` (YOLOv8) dependency
- [x] Broken `requirements.profile.txt` referencing wrong files
- [x] Profile deployment script using wrong requirements
- [x] NumPy version conflicts
- [x] No LLM deployment script

---

## 📋 Deployment Order (IMPORTANT)

Deploy in this **exact order** to minimize downtime:

### Step 1: Push to GitHub
```bash
cd /Users/uthamkumar/Uthamkumar/SUKO/Sukofinal
git status
git add -A
git commit -m "Fix core dependencies and module errors across all pods"
git push
```

### Step 2: Deploy Selfie Feature Worker (FIRST - longest download)
```bash
ssh root@195.26.233.53 -p 23674 -i ~/.ssh/id_ed25519
```
Then on the pod:
```bash
cd /workspace/img_pod
git pull
bash deploy_selfie_feature.sh
# Wait for Qwen2-VL-7B download (~16GB) + model load
# Should see: "loaded_vqa_model" in logs
```
**Expected time:** 5-10 minutes first time, 30-60s subsequent

### Step 3: Deploy Profile Worker
```bash
ssh root@103.196.86.90 -p 13560 -i ~/.ssh/id_ed25519
```
Then on the pod:
```bash
cd /workspace/img_pod
git pull
bash deploy_profile.sh
# Wait for Z-Image model load
```
**Expected time:** 15-30 seconds

### Step 4: Deploy LLM Service
```bash
ssh root@82.221.170.242 -p 45164 -i ~/.ssh/id_ed25519
```
Then on the pod:
```bash
cd /workspace/img_pod
git pull
bash deploy_llm.sh
# Wait for Qwen2.5-7B model load
```
**Expected time:** 20-40 seconds

### Step 5: Deploy Interface (LAST - connects to all workers)
```bash
ssh root@213.173.105.83 -p 58672 -i ~/.ssh/id_ed25519
```
Then on the pod:
```bash
cd /workspace/img_pod
git pull
bash deploy_interface.sh
# Should start immediately (no model loading)
```
**Expected time:** 3-5 seconds

---

## ✅ Post-Deployment Health Checks

### 1. Check All Services Respond
```bash
# From local machine
curl https://i8yfozejwi4lyk-8000.proxy.runpod.net/healthz
curl https://wp6is8takig1z0-8003.proxy.runpod.net/healthz
curl https://ika5ni96asnpts-8003.proxy.runpod.net/healthz
curl https://cf0pc97a1e9lki-8002.proxy.runpod.net/healthz
```
**Expected:** All return `{"status":"ok","service":"..."}`

### 2. Check Logs for Errors
On each pod, verify no import errors:
```bash
# Selfie Feature
tail -n 100 /tmp/selfie-feature-8003.log | grep -i error

# Profile Worker
tail -n 100 /tmp/profile-8003.log | grep -i error

# LLM Service
tail -n 100 /tmp/llm-8002.log | grep -i error

# Interface
tail -n 100 /tmp/interface-8000.log | grep -i error
```
**Expected:** No "ModuleNotFoundError" or "ImportError"

### 3. Verify Model Loading
Check logs for successful model loads:

**Selfie Feature:**
```bash
tail -n 200 /tmp/selfie-feature-8003.log | grep loaded
```
Should see:
- `loaded_yolov8`
- `loaded_vqa_model`

**Profile Worker:**
```bash
tail -n 100 /tmp/profile-8003.log | grep loaded
```
Should see:
- `loaded_profile_model`

**LLM Service:**
```bash
tail -n 100 /tmp/llm-8002.log | grep loaded
```
Should see:
- `LLM loaded` or similar

### 4. Run End-to-End Test
```bash
cd /Users/uthamkumar/Uthamkumar/SUKO/Sukofinal
.venv/bin/python testing/profile_create_and_save.py
```
**Expected:** 
- Features extracted successfully
- Images generated in `testing/output/`
- No module errors

---

## 🔍 Verification Commands

### Check Service Processes
On each pod:
```bash
ps aux | grep uvicorn
```
Should see Python processes on correct ports (8000, 8002, 8003)

### Check GPU Usage
On GPU pods:
```bash
nvidia-smi
```
Should see:
- Selfie Feature: ~18-20GB VRAM used
- Profile Worker: ~4-6GB VRAM used
- LLM Service: ~14-16GB VRAM used

### Check Disk Space
On all pods (models are large):
```bash
df -h /workspace
```
Ensure sufficient space for models

---

## ⚠️ Common Issues & Solutions

### "No module named 'sentencepiece'"
**Cause:** Old venv or cache  
**Fix:** 
```bash
pip install --upgrade --force-reinstall sentencepiece
```

### "No module named 'ultralytics'"
**Cause:** Selfie worker didn't install correctly  
**Fix:**
```bash
pip install ultralytics>=8.1.0
```

### Service Won't Start
**Cause:** Port already in use  
**Fix:**
```bash
ps aux | grep uvicorn | grep 8003 | awk '{print $2}' | xargs kill -9
# Then redeploy
```

### CUDA Out of Memory
**Cause:** Wrong GPU assigned  
**Fix:** Ensure pod has correct VRAM:
- Selfie Feature needs **48GB** (RTX 6000 Ada)
- Others need **24GB** (RTX 4090)

---

## 📊 Expected Resource Usage

### After Successful Deployment

| Pod | CPU | VRAM | Disk | Status |
|-----|-----|------|------|--------|
| Interface | 5-10% | 0 | ~2GB | ✅ Running |
| Selfie Feature | 10-20% | 18-20GB | ~20GB | ✅ Running |
| Profile Worker | 5-15% | 4-6GB | ~5GB | ✅ Running |
| LLM Service | 10-20% | 14-16GB | ~15GB | ✅ Running |

---

## 🎯 Success Criteria

Before considering deployment complete:

- [ ] All 4 health checks pass
- [ ] No module import errors in any logs
- [ ] All models loaded successfully
- [ ] End-to-end test generates valid images
- [ ] GPU memory usage within expected ranges
- [ ] Services respond to requests < 90s

---

## 📝 Rollback Plan

If deployment fails:

### Option 1: Redeploy from scratch
```bash
cd /workspace/img_pod
git checkout HEAD~1  # Go back one commit
bash deploy_<service>.sh
```

### Option 2: Manual fix
```bash
# Install missing dependency
pip install <package>

# Restart service
ps aux | grep uvicorn | grep <port> | awk '{print $2}' | xargs kill -9
bash deploy_<service>.sh
```

---

## ✅ Final Checklist

Before marking deployment as complete:

- [ ] Git changes pushed to GitHub
- [ ] All 4 pods deployed successfully
- [ ] Health checks pass for all services
- [ ] Logs show no import/module errors  
- [ ] Models loaded correctly
- [ ] End-to-end test passes
- [ ] GPU memory usage normal
- [ ] Response times acceptable

---

## 📞 Next Actions

1. **Deploy Now:** Follow steps above in order
2. **Monitor for 1 hour:** Watch logs for any errors
3. **Run load tests:** Ensure stability under load
4. **Document any new issues:** Update this checklist

---

**Ready to deploy! All core infrastructure and module issues resolved.** ✅
