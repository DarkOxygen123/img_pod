import base64
import gc
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


def _profile_prompt(payload: dict) -> tuple[str, str]:
    """Build preset prompt from features dict for waist-level profile photo."""
    # Unwrap avatar_features if nested
    features = payload.get("avatar_features", payload)
    observed = features.get("observed", {})
    dress = features.get("dress", {})
    accessories = features.get("accessories", {})
    
    # Extract features with defaults
    gender_raw = (observed.get("gender") or "person").strip()
    # Normalize gender into prompt-friendly terms.
    if gender_raw == "male":
        gender = "man"
    elif gender_raw == "female":
        gender = "woman"
    else:
        gender = "person"
    age = observed.get("age_appearance") or "adult"
    skin_tone = observed.get("skin_tone") or "natural"
    skin_undertone = observed.get("skin_undertone")
    hair_color = observed.get("hair_color") or "dark"
    hair_type = observed.get("hair_type") or "natural"
    hair_style = observed.get("hair_style")
    hair_length = observed.get("hair_length") or "medium"
    hairline_type = observed.get("hairline_type")
    balding_pattern = observed.get("balding_pattern")
    eye_color = observed.get("eye_color") or "brown"
    eye_shape = observed.get("eye_shape")
    face_shape = observed.get("face_shape") or "oval"
    facial_hair = observed.get("facial_hair")
    facial_hair_density = observed.get("facial_hair_density")
    beard_style = observed.get("beard_style")
    mustache_style = observed.get("mustache_style")
    facial_marks = observed.get("facial_marks")
    facial_mark_position = observed.get("facial_mark_position")
    expression = observed.get("expression")
    dress_color = dress.get("dress_color") or "casual"
    dress_type = dress.get("dress_type") or "clothing"
    
    # Extract accessories
    hat_present = accessories.get("hat_present") == "yes"
    hat_style = accessories.get("hat_style")
    hat_color = accessories.get("hat_color")
    glasses_present = accessories.get("glasses_present") == "yes"
    glasses_type = accessories.get("glasses_type")
    glasses_color = accessories.get("glasses_color")
    mask_present = accessories.get("mask_present") == "yes"
    mask_type = accessories.get("mask_type")
    mask_color = accessories.get("mask_color")
    
    # Build template-based description
    description_parts = []
    
    # Core identity (simple buckets tend to be more reliable than ranges)
    description_parts.append(f"A {age} {gender}")
    
    # Facial features
    if skin_undertone and skin_undertone != "none":
        description_parts.append(f"with {skin_tone} skin tone ({skin_undertone} undertone)")
    else:
        description_parts.append(f"with {skin_tone} skin tone")

    description_parts.append(f"{hair_color} {hair_type} hair ({hair_length} length)")
    if hair_style and hair_style != "none":
        description_parts.append(f"hairstyle: {hair_style}")
    if hairline_type and hairline_type != "none":
        description_parts.append(f"hairline: {hairline_type}")
    if balding_pattern and balding_pattern not in ("none", "no"):
        description_parts.append(f"balding: {balding_pattern}")

    if eye_shape and eye_shape != "none":
        description_parts.append(f"{eye_color} eyes ({eye_shape} shape)")
    else:
        description_parts.append(f"{eye_color} eyes")

    description_parts.append(f"{face_shape} face shape")
    
    # Facial hair rules
    if gender == "woman":
        facial_hair = "none"
        facial_hair_density = "none"
        beard_style = "none"
        mustache_style = "none"

    # Prefer split fields if present so we can represent BOTH mustache and beard.
    beard = (beard_style or "").strip()
    mustache = (mustache_style or "").strip()
    if beard and beard != "none" and mustache and mustache != "none":
        if facial_hair_density and facial_hair_density != "none":
            description_parts.append(f"with {facial_hair_density} {mustache} mustache and {beard} beard")
        else:
            description_parts.append(f"with {mustache} mustache and {beard} beard")
    elif beard and beard != "none":
        if facial_hair_density and facial_hair_density != "none":
            description_parts.append(f"with {facial_hair_density} {beard} beard")
        else:
            description_parts.append(f"with {beard} beard")
    elif mustache and mustache != "none":
        if facial_hair_density and facial_hair_density != "none":
            description_parts.append(f"with {facial_hair_density} {mustache} mustache")
        else:
            description_parts.append(f"with {mustache} mustache")
    elif facial_hair and facial_hair != "none":
        # Fallback: legacy single-field facial hair.
        if facial_hair_density and facial_hair_density != "none":
            description_parts.append(f"with {facial_hair_density} {facial_hair} facial hair")
        else:
            description_parts.append(f"with {facial_hair} facial hair")
    elif gender == "man" and not mask_present and (
        (beard == "none" and mustache == "none") or (facial_hair == "none")
    ):
        # Only assert clean-shaven when explicitly detected none and lower face is visible.
        description_parts.append("clean-shaven")

    # Facial marks
    if facial_marks and facial_marks != "none":
        if facial_mark_position and facial_mark_position != "none":
            description_parts.append(f"with a visible {facial_marks} on the {facial_mark_position}")
        else:
            description_parts.append(f"with a visible {facial_marks} on the face")

    # Expression (helps profile consistency)
    if expression and expression != "none":
        description_parts.append(f"expression: {expression}")
    
    # Clothing
    description_parts.append(f"wearing {dress_color} {dress_type}")
    
    # Accessories
    if hat_present and hat_style and hat_style != "none":
        if hat_color and hat_color != "none":
            description_parts.append(f"wearing a {hat_color} {hat_style}")
        else:
            description_parts.append(f"wearing a {hat_style}")
    
    if glasses_present and glasses_type and glasses_type != "none":
        if glasses_color and glasses_color != "none":
            description_parts.append(f"wearing {glasses_color} {glasses_type}")
        else:
            description_parts.append(f"wearing {glasses_type}")
    
    if mask_present and mask_type and mask_type != "none":
        if mask_color and mask_color != "none":
            description_parts.append(f"wearing a {mask_color} {mask_type} mask")
        else:
            description_parts.append(f"wearing a {mask_type} mask")
    
    full_description = ", ".join([p for p in description_parts if p])

    # Add more explicit hair guidance.
    hair_desc = f"{hair_color} {hair_type} hair ({hair_length} length)"
    if hair_style and hair_style != "none":
        hair_desc = f"{hair_desc}, hairstyle: {hair_style}"
    if hairline_type and hairline_type != "none":
        hair_desc = f"{hair_desc}, hairline: {hairline_type}"
    if balding_pattern and balding_pattern not in ("none", "no"):
        hair_desc = f"{hair_desc}, balding: {balding_pattern}"

    negative_bits = [
        "low quality",
        "blurry",
        "bad anatomy",
        "deformed face",
        "extra faces",
        "multiple heads",
        "two people",
        "two persons",
        "multiple people",
        "group",
        "crowd",
        "extra person",
        "duplicate person",
        "photorealistic",
        "realistic photo",
        "camera",
        "dslr",
        "film grain",
    ]
    if gender == "woman":
        negative_bits += ["beard", "mustache", "facial hair"]
    negative_prompt = ", ".join(negative_bits)

    sunglasses_types = {"sunglasses", "aviators"}
    sunglasses_present = bool(glasses_present and glasses_type in sunglasses_types)

    # Occlusion-aware "must match" rules: avoid hallucinating hidden attributes.
    must_bits = []
    must_bits.append(f"Gender must read clearly as a {gender}.")
    must_bits.append(f"Skin tone MUST be {skin_tone}.")
    if not hat_present:
        must_bits.append(f"Hair MUST match exactly: {hair_desc}.")
    else:
        must_bits.append("Hat is present; hair may be partially hidden. Do not invent extra hair details.")
    if not sunglasses_present:
        must_bits.append(f"Eyes MUST be {eye_color}.")
    else:
        must_bits.append("Sunglasses are present; do not invent eye color.")
    if not mask_present:
        must_bits.append(f"Face shape MUST be {face_shape}.")
    else:
        must_bits.append("Mask is present; do not invent detailed jawline/face-shape features hidden by the mask.")
    
    # Build final prompt with emphasis on accuracy
    prompt = (
        f"Professional waist-up portrait: {full_description}. "
        f"Single subject only: EXACTLY ONE person in the image. "
        f"CRITICAL REQUIREMENTS: {' '.join(must_bits)} "
        f"Style: high-quality stylized 3D cartoon character render (NOT photorealistic), "
        f"toon shading, smooth clean materials, simplified geometry but high detail, "
        f"animated/avatar look, "
        f"front-facing centered composition, direct eye contact, waist-up framing, "
        f"soft studio lighting, sharp focus, clean plain background. "
    )
    return prompt, negative_prompt


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "profile_image_generator"})


@app.post("/v1/profile/generate", response_class=JSONResponse)
async def generate_from_features(avatar_features: dict = Body(...)) -> JSONResponse:
    """Generate profile image from features JSON. Returns JSON with base64 PNG."""
    t_start = time.time()
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Pass the full dict - prompt function will extract avatar_features
    t_prompt0 = time.time()
    prompt, negative_prompt = _profile_prompt(avatar_features)
    t_prompt_s = time.time() - t_prompt0
    
    # Pass negative_prompt when supported (helps prevent extra people/faces).
    t_infer0 = time.time()
    try:
        out = _pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
        )
    except TypeError:
        out = _pipe(prompt=prompt, height=512, width=512, num_inference_steps=4, guidance_scale=0.0)
    t_infer_s = time.time() - t_infer0
    image = out.images[0]
    t_encode0 = time.time()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    t_encode_s = time.time() - t_encode0
    
    t_elapsed = time.time() - t_start
    logger.info(
        "image_generation_complete",
        extra={
            "extra_fields": {
                "seconds": round(t_elapsed, 2),
                "prompt_seconds": round(t_prompt_s, 3),
                "infer_seconds": round(t_infer_s, 3),
                "encode_seconds": round(t_encode_s, 3),
                "image_size": len(image_bytes),
            }
        },
    )
    
    # Clear memory to prevent leaks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return JSONResponse({"image_bytes_b64": base64.b64encode(image_bytes).decode()})
