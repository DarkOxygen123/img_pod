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
    # Log incoming features for debugging
    logger.info("prompt_features_received", extra={"extra_fields": {"features": features}})
    
    observed = features.get("observed", {})
    dress = features.get("dress", {})
    
    # Extract features with defaults
    gender = observed.get("gender") or "person"
    age = observed.get("age_appearance") or "adult"
    skin_tone = observed.get("skin_tone") or "natural"
    hair_color = observed.get("hair_color") or "dark"
    hair_type = observed.get("hair_type") or "natural"
    hair_length = observed.get("hair_length") or "medium"
    eye_color = observed.get("eye_color") or "brown"
    face_shape = observed.get("face_shape") or "oval"
    facial_hair = observed.get("facial_hair")
    dress_color = dress.get("dress_color") or "casual"
    dress_type = dress.get("dress_type") or "clothing"
    
    # Build template-based description
    description_parts = []
    
    # Core identity
    description_parts.append(f"A {age} {gender}")
    
    # Facial features
    description_parts.append(f"with {skin_tone} skin tone")
    description_parts.append(f"{hair_color} {hair_type} hair ({hair_length} length)")
    description_parts.append(f"{eye_color} eyes")
    description_parts.append(f"{face_shape} face shape")
    
    # Facial hair if present
    if facial_hair and facial_hair != "none":
        description_parts.append(f"{facial_hair} facial hair")
    
    # Clothing
    description_parts.append(f"wearing {dress_color} {dress_type}")
    
    full_description = ", ".join(description_parts)
    
    # Build final prompt with emphasis on accuracy
    return (
        f"Professional waist-up portrait: {full_description}. "
        f"CRITICAL REQUIREMENTS: "
        f"Hair MUST be {hair_type} texture with {hair_color} color and {hair_length} length. "
        f"Skin tone MUST be {skin_tone}. "
        f"Eyes MUST be {eye_color}. "
        f"Face shape MUST be {face_shape}. "
        f"Style: 3D Disney Pixar animation, photorealistic details, "
        f"front-facing centered composition, direct eye contact with camera, "
        f"upper body visible from waist up, cinematic studio lighting, "
        f"expressive facial features, professional quality rendering"
    )


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
