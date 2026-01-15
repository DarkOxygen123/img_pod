import base64
import io
import os
import time

import torch
import diffusers
import transformers
from diffusers import DiffusionPipeline
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from shared.logging_config import get_logger
from shared.models import WorkerChat1to1Request
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.chat1to1_worker_settings()

MODEL_ID = os.getenv("CHAT1TO1_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEVICE = os.getenv("CHAT1TO1_DEVICE", "cuda")

_pipe = None


@app.on_event("startup")
async def startup() -> None:
    global _pipe
    t0 = time.time()
    
    logger.info("loading_chat1to1_model", extra={"extra_fields": {"model_id": MODEL_ID}})
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    try:
        _pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        _pipe.enable_model_cpu_offload()
    except Exception:
        _pipe = _pipe.to(DEVICE)
    _pipe.set_progress_bar_config(disable=True)
    logger.info("loaded_chat1to1_model", extra={"extra_fields": {"seconds": round(time.time() - t0, 2)}})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "chat1to1_image_generator"})


@app.get("/debug/versions")
async def debug_versions() -> JSONResponse:
    return JSONResponse(
        {
            "torch": getattr(torch, "__version__", None),
            "diffusers": getattr(diffusers, "__version__", None),
            "transformers": getattr(transformers, "__version__", None),
            "model_id": MODEL_ID,
            "device": DEVICE,
        }
    )


@app.post("/v1/chat/1to1/generate", response_class=JSONResponse)
async def generate_chat_image(request: WorkerChat1to1Request = Body(...)) -> JSONResponse:
    """
    Generate 1:1 chat contextual image from expanded prompt.
    
    No NSFW restrictions - generates any content based on expanded prompt.
    Uses Z-Image Turbo diffusion model.
    """
    t_start = time.time()
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(
        "chat1to1_generation_started",
        extra={
            "extra_fields": {
                "prompt_length": len(request.prompt),
                "height": request.height,
                "width": request.width,
            }
        },
    )
    
    # Generate image with provided prompt (no NSFW filtering)
    t_infer0 = time.time()
    try:
        try:
            out = _pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            )
        except TypeError:
            # Fallback if negative_prompt not supported
            out = _pipe(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            )
    except Exception as e:
        logger.exception("chat1to1_generation_failed")
        raise HTTPException(
            status_code=500,
            detail={"error": type(e).__name__, "message": str(e)[:500]},
        )
    t_infer_s = time.time() - t_infer0
    
    image = out.images[0]
    
    # Encode to bytes
    t_encode0 = time.time()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    t_encode_s = time.time() - t_encode0
    
    t_elapsed = time.time() - t_start
    logger.info(
        "chat1to1_generation_complete",
        extra={
            "extra_fields": {
                "seconds": round(t_elapsed, 2),
                "infer_seconds": round(t_infer_s, 3),
                "encode_seconds": round(t_encode_s, 3),
                "image_size": len(image_bytes),
            }
        },
    )
    
    return JSONResponse({"image_bytes_b64": base64.b64encode(image_bytes).decode()})
