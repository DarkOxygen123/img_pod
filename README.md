# Image Generation Pipeline (RunPod)

This repo contains four FastAPI services:

- Interface/Load Balancer (public, port 8000)
- LLM Service (gpu_llm, port 8002)
- Profile Worker (gpu_profile_1, port 8003)
- Text2Img Worker (gpu_worker_1, port 8004)

## Quick Start

1) Create a virtualenv and install dependencies (see `requirements.txt`).
2) Copy `config/config.example.yaml` to `config/config.yaml` and adjust URLs/limits.
3) Run the desired service, e.g.:
   - `uvicorn services.interface.main:app --host 0.0.0.0 --port 8000`
   - `uvicorn services.llm_service.main:app --host 0.0.0.0 --port 8002`
   - `uvicorn services.profile_worker.main:app --host 0.0.0.0 --port 8003`
   - `uvicorn services.text2img_worker.main:app --host 0.0.0.0 --port 8004`

## Pod Deployment (Git Pull)

On each pod, keep a checkout at `/app/Sukofinal` and run one service:

- CPU pod (public): `./scripts/run_service.sh interface 8000`
- LLM pod: `./scripts/run_service.sh llm 8002`
- Profile GPU pod: `./scripts/run_service.sh profile 8003`
- Text2Img GPU pod: `./scripts/run_service.sh text2img 8004`

Workers can still be hit directly via their RunPod proxy URLs for development testing, but only the interface URL should be used by your backend.

## Config

All critical values are configurable via environment variables or `config/config.yaml`:
- Queue lengths per endpoint (default 10)
- Retry-after cap (default 120s)
- Regen attempts (default 2 total)
- SLA caps: profile 10s, text2img 8s
- Worker URLs (LLM, profile worker, text2img workers)

## Deployment

Each pod can be updated via `git pull` + service restart. A simple SSH/rsync workflow is assumed.
