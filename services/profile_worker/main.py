import base64
import os
import io
import time
from typing import Dict

import torch
from diffusers import DiffusionPipeline
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from shared.logging_config import get_logger
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.profile_worker_settings()

MODEL_ID = os.getenv("PROFILE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEVICE = os.getenv("PROFILE_DEVICE", "cuda")

_pipe = None


@app.on_event("startup")
async def startup() -> None:
    global _pipe
    t0 = time.time()
    
    # Load diffusion model only
    logger.info("loading_profile_text2img_model", extra={"extra_fields": {"model_id": MODEL_ID}})
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
    logger.info("loaded_profile_model", extra={"extra_fields": {"seconds": round(time.time() - t0, 2)}})


def _profile_prompt(features: dict) -> str:
    """Build preset prompt from features dict for waist-level profile photo."""
    observed = features.get("observed", {})
    
    # Build detailed feature description
    parts = []
    if observed.get("hair_color"):
        parts.append(f"{observed['hair_color']} hair")
    if observed.get("hair_type"):
        parts.append(f"{observed['hair_type']} hair texture")
    if observed.get("hair_length"):
        parts.append(f"{observed['hair_length']} hair length")
    if observed.get("eye_color"):
        parts.append(f"{observed['eye_color']} eyes")
    if observed.get("skin_tone"):
        parts.append(f"{observed['skin_tone']} skin tone")
    if observed.get("nose_bridge"):
        parts.append(f"{observed['nose_bridge']} nose bridge")
    if observed.get("lip_fullness"):
        parts.append(f"{observed['lip_fullness']} lips")
    if observed.get("face_shape"):
        parts.append(f"{observed['face_shape']} face shape")
    if observed.get("facial_hair_type") and observed.get("facial_hair_type") != "none":
        parts.append(f"{observed['facial_hair_type']}")
    if observed.get("age_appearance"):
        parts.append(f"{observed['age_appearance']} appearance")
    
    desc = ", ".join(parts) if parts else "pleasant face"
    
    # Waist-level professional profile photo
    return (f"Professional portrait photo from waist up: {desc}. "
            f"IMPORTANT: Must accurately depict ALL specified features including hair texture and color. "
            f"3D Disney Pixar animation style, front-facing centered composition, "
            f"direct eye contact with camera, upper body visible, "
            f"cinematic studio lighting, expressive facial features, "
            f"professional headshot quality, photorealistic rendering")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "profile_image_generator"})


@app.post("/v1/profile/generate", response_class=JSONResponse)
async def generate_from_features(avatar_features: dict = Body(...)) -> JSONResponse:
    """Generate profile image from features JSON. Returns JSON with base64 PNG."""
    t_start = time.time()
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = _profile_prompt(avatar_features)
    logger.info("profile_prompt_generated", extra={"extra_fields": {"prompt": prompt}})
    
    out = _pipe(prompt=prompt, height=512, width=512, num_inference_steps=4, guidance_scale=0.0)
    image = out.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    
    t_elapsed = time.time() - t_start
    logger.info("image_generation_complete", extra={"extra_fields": {"seconds": round(t_elapsed, 2), "image_size": len(image_bytes)}})
    return JSONResponse({"image_bytes_b64": base64.b64encode(image_bytes).decode()})
