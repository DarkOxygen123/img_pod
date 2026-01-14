# AI Coding Agent Instructions - Image Generation Pipeline

## Architecture Overview

This is a **distributed microservices system** running on RunPod GPUs for AI-powered avatar generation. Seven independent FastAPI services communicate via HTTP:

- **Interface (CPU)**: Load balancer/router at port 8000, handles queue management and orchestration
- **Selfie Feature Worker (RTX 6000 Ada, 48GB)**: Port 8003, runs Qwen2-VL-7B (~16GB VRAM) + YOLOv8-Face for feature extraction
- **Profile Worker (RTX 4090, 24GB)**: Port 8003, runs Z-Image Turbo diffusion model for avatar generation
- **1:1 Chat Worker (RTX 4090, 24GB)**: Port 8005, runs Z-Image Turbo for contextual chat images (NO NSFW restrictions)
- **Shorts Worker (RTX 4090, 24GB)**: Port 8006, runs Z-Image Turbo for shorts generation (NSFW moderated)
- **Scenes Worker (RTX 4090, 24GB)**: Port 8007, runs Z-Image Turbo for scenes generation (NSFW moderated)
- **LLM Service (RTX 4090, 24GB)**: Port 8002, runs Qwen2.5-7B for text generation and prompt expansion

### Critical Data Flow Pattern
**Profile Creation**:
```
POST /v1/profile/create (Interface)
  → Selfie Feature Worker: POST /v1/profile/analyze (extracts features from selfie)
  → Profile Worker: POST /v1/profile/generate (generates avatar image)
  → Returns: profile image + features JSON
```

**1:1 Chat Contextual Image** (NEW):
```
POST /v1/chat/1to1/imagegen (Interface)
  → LLM Service: POST /v1/chat/1to1/expand (analyzes chat context, expands prompt)
  → 1:1 Chat Worker: POST /v1/chat/1to1/generate (generates image from expanded prompt)
  → LLM Service: POST /v1/chat/1to1/caption (generates 10-15 word caption)
  → Returns: {image_base64, caption, prompt_used}
```

**Shorts Generation** (NEW):
```
POST /v1/chat/shorts/generate (Interface)
  → LLM Service: POST /v1/chat/shorts/expand (expands prompt with NSFW moderation)
  → Shorts Worker: POST /v1/chat/shorts/generate (generates image)
  → LLM Service: POST /v1/chat/1to1/caption (generates caption)
  → Returns: {image_base64, caption, prompt_used}
```

**Scenes Generation** (NEW):
```
POST /v1/chat/scenes/generate (Interface)
  → LLM Service: POST /v1/chat/scenes/expand (expands prompt with NSFW moderation)
  → Scenes Worker: POST /v1/chat/scenes/generate (generates image)
  → LLM Service: POST /v1/chat/1to1/caption (generates caption)
  → Returns: {image_base64, caption, prompt_used}
```

**Load balancing**: Interface uses round-robin across worker pools (`pick_selfie_feature_worker()`, `pick_profile_worker()`, `pick_chat1to1_worker()`, `pick_shorts_worker()`, `pick_scenes_worker()` in [services/interface/main.py](../services/interface/main.py)). Worker URLs configured in [shared/settings.py](../shared/settings.py).

## Configuration System

All services use **dual configuration**: environment variables override YAML config. Pattern in [shared/settings.py](../shared/settings.py):

```python
class InterfaceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INTERFACE_", extra="ignore")
    # Environment variables like INTERFACE_CHAT1TO1_WORKER_URLS override defaults
```

### Easy Pod Updates (When Pods Change)

**Problem**: RunPod pods can be recreated with new IDs/URLs, requiring credential updates.

**Solution**: Use environment variables for quick updates without code changes:

```bash
# On interface pod, update worker URLs via env vars

### Queue Management

**BoundedQueue

**Services & Ports**:
- Interface: port 8000 (`deploy_interface.sh`)
- Selfie Feature: port 8003 (`deploy_selfie_feature.sh`)
- Profile Worker: port 8003 (`deploy_profile.sh`)
- 1:1 Chat Worker: port 8005 (`deploy_chat1to1.sh`)
- LLM Service: port 8002 (`deploy_llm.sh`)s** with SLA-based rejection (see [shared/queue_manager.py](../shared/queue_manager.py)):

```python
profile_queue = BoundedQueue(capacity=interface_settings.profile_queue_max)
# When full, returns 429 with Retry-After header
```

- Queue fullness → HTTP 429 with `Retry-After` header (capped at `retry_after_cap_ms`)
- Each endpoint has independent queue: `profile_queue`, `text2img_queue`
- Single worker per queue processes items via `SingleWorker` class

## Deployment Workflow

**Critical**: Each pod deploys ONE service using dedicated deployment scripts:

```bash
# From pod SSH session
cdShorts Worker: port 8006 (`deploy_shorts.sh`)
- Scenes Worker: port 8007 (`deploy_scenes.sh`)
-  /workspace/img_pod
git pull
bash deploy_selfie_feature.sh  # or deploy_interface.sh, deploy_profile.sh, deploy_llm.sh
```

Deployment scripts ([deploy_*.sh](../deploy_selfie_feature.sh)):
1. Install service-specific requirements (`requirements.selfie_feature.txt`, etc.)
```python
profile_queue = BoundedQueue(capacity=interface_settings.profile_queue_max)
# When full, returns 429 with Retry-After header
```

- Queue fullness → HTTP 429 with `Retry-After` header (capped at `retry_after_cap_ms`)
- Each endpoint has independent queue: `profile_queue`, `text2img_queue`, `chat1to1_queue`, `shorts_queue`, `scenes --port <port>`
4. Logs to `/tmp/<service>-<port>.log`

**Do NOT use `scripts/run_service.sh`** for deployment - it's for local dev only.

## Requirements Structure

**Project-specific pattern**: Base requirements split by hardware capability:

- `requirements.interface.txt`: CPU-only dependencies (FastAPI, httpx, Pydantic)
- `requirements.gpu.txt`: GPU dependencies (torch, transformers, diffusers)
- `requirements.<service>.txt`: Service-specific (e.g., `ultralytics` for selfie_feature)

**Critical constraint**: All GPU services require `numpy>=1.26.0,<2.0.0` and `sentencepiece>=0.1.99` (tokenizer dependency for Qwen models).

## Testing Pattern

Tests in [testing/](../testing/) directory use **config.json** (not YAML):

```python
# testing/profile_create_and_save.py pattern
with open("testing/config.json") as f:
    config = json.load(f)
response = requests.post(f"{config['interface_base_url']}/v1/profile/create", ...)
```

Generate test selfies first: `python testing/generate_test_selfies.py`  
Run integration test: `python testing/profile_create_and_save.py`  
Outputs saved to `testing/selfies_output/`

## Policy & Sanitization

Content filtering in [shared/policy.py](../shared/policy.py) uses **two-tier blocking**:

1. **Hard block** (`HARD_BLOCK_KEYWORDS`): Returns immediate 422 error
2. **Soft block** (`EXPLICIT_KEYWORDS`): Appends safety constraints to prompt

Used in interface endpoints:
- `evaluate_private_chat()`: For `/v1/chat/private/imagegen`
- `sanitize_general()`: For `/v1/imagegen`

## Model Management

**Selfie Feature Worker**: Models auto-download on first run:
- Qwen2-VL-7B: `transformers.Qwen2VLForConditionalGeneration.from_pretrained()`
- YOLOv8-Face: `ultralytics.YOLO('yolov8n-face.pt')`

Models cache to `/root/.cache/huggingface/` and `/root/.ultralytics/` respectively.

**Critical**: First deployment downloads ~16GB, takes 5-10 minutes. Check logs at `/tmp/selfie-feature-8003.log`.

## Shared Module Pattern

Services import from `shared/` directory (not a package, just sys.path):

```python
from shared.logging_config import get_logger  # Structured logging
from shared.models import ProfileCreateResponse  # Pydantic models
from shared.http_client import post_json, post_multipart_file  # Async HTTP
```

## NSFW Moderation Pattern

**Context**: Different endpoints have different NSFW policies:
- **1:1 Chat** (`/v1/chat/1to1/imagegen`): NO restrictions - generates based on chat context
- **Shorts/Scenes** (`/v1/chat/shorts/generate`, `/v1/chat/scenes/generate`): NSFW moderated - avoids straight nudity

**Implementation**: LLM expansion endpoints ([services/llm_service/main.py](../services/llm_service/main.py)) handle moderation:

```python
# For shorts/scenes: POST /v1/chat/shorts/expand, POST /v1/chat/scenes/expand
system_prompt = """...(expand user message)...
CRITICAL NSFW MODERATION:
- If content is sexually suggestive/explicit, add clothing/occlusion instructions
- Examples: "draped in silk sheets", "wearing lace lingerie", "positioned behind translucent curtains"
- Never generate straight nudity - always include coverage elements
"""
```

**Key difference from 1:1 chat**:
- 1:1 chat expansion (`/v1/chat/1to1/expand`) has NO NSFW moderation instructions
- Shorts/scenes expansion adds clothing/occlusion constraints to avoid explicit nudity
- Both use same caption endpoint (`/v1/chat/1to1/caption`) - captions are always safe

## Scaling Workers

To add more workers (e.g., second selfie feature worker):

1. Create new RunPod pod with required GPU
2. Deploy service: `bash deploy_selfie_feature.sh`
3. Update [shared/settings.py](../shared/settings.py):
   ```python
   selfie_feature_worker_urls: List[HttpUrl] = [
       "https://wp6is8takig1z0-8003.proxy.runpod.net",
       "https://NEW_POD_ID-8003.proxy.runpod.net"  # Add here
   ]
   ```
4. Redeploy interface: `bash deploy_interface.sh`

Interface automatically round-robins across all URLs in list.

## Common Gotchas

1. **Don't confuse service ports**: Profile Worker and Selfie Feature Worker both use 8003 (different pods), 1:1 Chat uses 8005, Shorts uses 8006, Scenes uses 8007
2. **Queue items need futures**: See `QueueItem` class in [services/interface/main.py](../services/interface/main.py) - payload + asyncio.Future for response
3. **VQA prompts in worker**: Prompts hardcoded in [services/selfie_feature_worker/main.py](../services/selfie_feature_worker/main.py) `_vqa_describe_features()` - modify there, not in interface
4. **Health checks use `/healthz`**: Not `/health` (except text2img worker uses `/health`)
5. **Face detection failures**: Worker uses graceful degradation - no face detected uses full image with quality penalty (see `FaceMeta.quality_score` in [shared/models.py](../shared/models.py))
6. **1:1 Chat has NO NSFW restrictions**: Unlike shorts/scenes, `/v1/chat/1to1/imagegen` generates any content based on chat context
7. **Shorts vs Scenes are identical code**: Separate workers for independent GPU tracking, both have NSFW moderation
8. **LLM timeout handling**: Chat expansion can take 10-15s, caption generation ~5s - adjust SLAs accordingly
9. **Message shortening**: LLM expansion shortens messages >25 words while preserving intent

## Key Files Reference

- Architecture docs: [DEPLOYMENT.md](../DEPLOYMENT.md), [DEPLOY_QUICK_START.md](../DEPLOY_QUICK_START.md)
- Pod SSH details: [DEPLOYMENT.md](../DEPLOYMENT.md) lines 4-27
- Bug fixes history: [FIXES_SUMMARY.md](../FIXES_SUMMARY.md)
- Settings schema: [shared/settings.py](../shared/settings.py)
- Data models: [shared/models.py](../shared/models.py)
