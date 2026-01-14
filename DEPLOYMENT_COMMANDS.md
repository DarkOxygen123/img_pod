# Deployment Commands for All Pods

## Pod Connection & Deployment

### 1. Interface Pod (CPU - Port 8000)
```bash
ssh root@154.54.102.17 -p 18245 -i ~/.ssh/suko_runpod

# Inside pod
cd /workspace
git clone <YOUR_REPO_URL> img_pod
cd img_pod
bash deploy_interface.sh

# Check logs
tail -f /tmp/interface-8000.log
```

**Pod ID**: bw77wupwq7k752  
**URL**: https://bw77wupwq7k752-8000.proxy.runpod.net

---

### 2. LLM Service Pod (A40 - Port 8002)
```bash
ssh root@213.173.102.200 -p 18662 -i ~/.ssh/suko_runpod

# Inside pod
cd /workspace
git clone <YOUR_REPO_URL> img_pod
cd img_pod
bash deploy_llm.sh

# Check logs
tail -f /tmp/llm-service-8002.log
```

**Pod ID**: abj0jt7cd4hgdy  
**URL**: https://abj0jt7cd4hgdy-8002.proxy.runpod.net

---

### 3. Selfie Feature Worker Pod (RTX 6000 Ada - Port 8003)
```bash
ssh root@216.81.245.79 -p 39920 -i ~/.ssh/suko_runpod

# Inside pod
cd /workspace
git clone <YOUR_REPO_URL> img_pod
cd img_pod
bash deploy_selfie_feature.sh

# Check logs (first run downloads ~16GB models, takes 5-10 mins)
tail -f /tmp/selfie-feature-8003.log
```

**Pod ID**: 97dsnjce4yxe96  
**URL**: https://97dsnjce4yxe96-8003.proxy.runpod.net

---

### 4. Profile Worker Pod (A40 - Port 8003)
```bash
ssh root@69.30.85.91 -p 22125 -i ~/.ssh/suko_runpod

# Inside pod
cd /workspace
git clone <YOUR_REPO_URL> img_pod
cd img_pod
bash deploy_profile.sh

# Check logs
tail -f /tmp/profile-worker-8003.log
```

**Pod ID**: x690sjq9dtevw4  
**URL**: https://x690sjq9dtevw4-8003.proxy.runpod.net

---

### 5. 1:1 Chat Worker Pod (A40 - Port 8005)
```bash
ssh root@194.68.245.48 -p 22019 -i ~/.ssh/suko_runpod

# Inside pod
cd /workspace
git clone <YOUR_REPO_URL> img_pod
cd img_pod
bash deploy_chat1to1.sh

# Check logs
tail -f /tmp/chat1to1-worker-8005.log
```

**Pod ID**: 6eubyihk4kt8l0  
**URL**: https://6eubyihk4kt8l0-8005.proxy.runpod.net

---

### 6. Shorts Worker Pod (A40 - Port 8006)
```bash
ssh root@194.68.245.18 -p 22111 -i ~/.ssh/suko_runpod

# Inside pod
cd /workspace
git clone <YOUR_REPO_URL> img_pod
cd img_pod
bash deploy_shorts.sh

# Check logs
tail -f /tmp/shorts-worker-8006.log
```

**Pod ID**: klu0524bz1nx1i  
**URL**: https://klu0524bz1nx1i-8006.proxy.runpod.net

---

### 7. Scenes Worker Pod (A40 - Port 8007)
```bash
ssh root@103.196.86.192 -p 13042 -i ~/.ssh/suko_runpod

# Inside pod
cd /workspace
git clone <YOUR_REPO_URL> img_pod
cd img_pod
bash deploy_scenes.sh

# Check logs
tail -f /tmp/scenes-worker-8007.log
```

**Pod ID**: 5r1nfrz20lc715  
**URL**: https://5r1nfrz20lc715-8007.proxy.runpod.net

---

## Deployment Order

**IMPORTANT**: Deploy in this order to avoid connection errors:

1. **LLM Service Pod** (8002) - Interface depends on this
2. **Selfie Feature Worker Pod** (8003) - Takes longest to download models
3. **Profile Worker Pod** (8003)
4. **1:1 Chat Worker Pod** (8005)
5. **Shorts Worker Pod** (8006)
6. **Scenes Worker Pod** (8007)
7. **Interface Pod** (8000) - Deploy LAST after all workers are ready

## Health Check Commands

After deployment, verify each service:

```bash
# From your local machine
curl https://bw77wupwq7k752-8000.proxy.runpod.net/healthz  # Interface
curl https://abj0jt7cd4hgdy-8002.proxy.runpod.net/healthz  # LLM
curl https://97dsnjce4yxe96-8003.proxy.runpod.net/healthz  # Selfie Feature
curl https://x690sjq9dtevw4-8003.proxy.runpod.net/healthz  # Profile
curl https://6eubyihk4kt8l0-8005.proxy.runpod.net/healthz  # Chat1to1
curl https://klu0524bz1nx1i-8006.proxy.runpod.net/healthz  # Shorts
curl https://5r1nfrz20lc715-8007.proxy.runpod.net/healthz  # Scenes
```

All should return: `{"status":"healthy"}`

## Quick Redeploy (After Git Push)

```bash
# SSH into specific pod, then:
cd /workspace/img_pod
git pull
bash deploy_<service>.sh  # e.g., deploy_interface.sh
```

## Troubleshooting

**Pod won't start**: Check logs at `/tmp/<service>-<port>.log`  
**Port already in use**: Deployment script auto-kills old process  
**Model download stuck**: Check disk space with `df -h` (need 200GB+)  
**Import errors**: Verify requirements installed with `pip list | grep transformers`
