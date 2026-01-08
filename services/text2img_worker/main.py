import io
import os
import time
import asyncio

import torch
import os
from diffusers import DiffusionPipeline
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import Response

from shared.logging_config import get_logger
from shared.models import WorkerText2ImgRequest
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.text2img_worker_settings()

MODEL_ID = os.getenv("TEXT2IMG_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DTYPE = os.getenv("TEXT2IMG_TORCH_DTYPE", "bfloat16")
DEVICE = os.getenv("TEXT2IMG_DEVICE", "cuda")

_pipe = None
_lock = asyncio.Lock()


def _torch_dtype():
    if DTYPE.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if DTYPE.lower() in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


@app.on_event("startup")
async def load_model() -> None:
    global _pipe
    t0 = time.time()
    torch_dtype = _torch_dtype()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logger.info(
        "loading_text2img_model",
        extra={"extra_fields": {"model_id": MODEL_ID, "dtype": str(torch_dtype), "device": DEVICE}},
    )
    _pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)
    # reduce VRAM pressure
    try:
        _pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        _pipe.enable_model_cpu_offload()
    except Exception:
        _pipe = _pipe.to(DEVICE)
    _pipe.set_progress_bar_config(disable=True)
    logger.info("loaded_text2img_model", extra={"extra_fields": {"seconds": round(time.time() - t0, 2)}})


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model_id": MODEL_ID}


@app.post("/", response_class=Response)
async def generate(request: WorkerText2ImgRequest = Body(...)) -> Response:
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    async with _lock:
        try:
            out = _pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            )
            image = out.images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(
        "text2img_generated",
        extra={"extra_fields": {"elapsed_ms": elapsed_ms, "prompt_len": len(request.prompt), "bytes": len(image_bytes)}},
    )
    return Response(content=image_bytes, media_type="image/png")
