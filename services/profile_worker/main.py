import base64
import os
import io
import os
import time
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from diffusers import DiffusionPipeline
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis

from shared.logging_config import get_logger
from shared.models import FaceMeta, FaceObserved, FaceProfileFeaturesV1
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.profile_worker_settings()

MODEL_ID = os.getenv("PROFILE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEVICE = os.getenv("PROFILE_DEVICE", "cuda")

_pipe = None
_face_analyzer = None


def _load_image_bytes(content: bytes) -> Tuple[np.ndarray, int, int]:
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    h, w = img.shape[:2]
    return img, w, h


def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def _get_dominant_color(region: np.ndarray) -> str:
    """Extract dominant color from image region."""
    # Convert to RGB and get average color
    avg_color = np.mean(region, axis=(0, 1)).astype(int)
    r, g, b = avg_color
    
    # Simple color classification
    if r < 50 and g < 50 and b < 50:
        return "black"
    elif r > 200 and g > 200 and b > 200:
        return "white"
    elif r > 180 and g < 100 and b < 100:
        return "red"
    elif g > r and g > b:
        return "brown" if g < 150 else "blonde"
    else:
        return "brown"


def _extract_facial_features_with_insightface(img_bgr: np.ndarray, face_data) -> FaceObserved:
    """Extract facial features using InsightFace analysis."""
    h, w = img_bgr.shape[:2]
    
    # Get face bbox for region extraction
    bbox = face_data.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    
    # Extract colors from face regions
    face_img = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    
    # Hair region (top of bbox)
    hair_y = max(0, y1 - int((y2 - y1) * 0.3))
    hair_region = img_bgr[hair_y:y1, x1:x2] if hair_y < y1 else None
    hair_color = _get_dominant_color(hair_region) if hair_region is not None and hair_region.size > 0 else None
    
    # Eye region (upper middle of face)
    eye_y1 = y1 + int((y2 - y1) * 0.3)
    eye_y2 = y1 + int((y2 - y1) * 0.5)
    eye_region = face_img[eye_y1-y1:eye_y2-y1, :] if face_img.size > 0 else None
    
    # Simplified eye color (hard to detect accurately)
    eye_color = "brown"  # Default, would need landmark detection for accuracy
    
    # Skin tone from cheek area
    cheek_y1 = y1 + int((y2 - y1) * 0.4)
    cheek_y2 = y1 + int((y2 - y1) * 0.7)
    cheek_x1 = x1 + int((x2 - x1) * 0.2)
    cheek_x2 = x2 - int((x2 - x1) * 0.2)
    cheek_region = img_bgr[cheek_y1:cheek_y2, cheek_x1:cheek_x2]
    
    skin_avg = np.mean(cheek_region, axis=(0, 1)).astype(int) if cheek_region.size > 0 else [128, 128, 128]
    brightness = np.mean(skin_avg)
    
    if brightness < 80:
        skin_tone = "dark"
    elif brightness < 120:
        skin_tone = "tan"
    elif brightness < 160:
        skin_tone = "medium"
    elif brightness < 200:
        skin_tone = "light"
    else:
        skin_tone = "fair"
    
    # Age from InsightFace (if available)
    age = getattr(face_data, 'age', None)
    if age is not None:
        if age < 30:
            age_appearance = "young"
        elif age < 50:
            age_appearance = "middle-aged"
        else:
            age_appearance = "senior"
    else:
        age_appearance = None
    
    # Gender from InsightFace
    gender = getattr(face_data, 'gender', None)
    if gender is not None:
        gender_presentation = "feminine" if gender == 0 else "masculine"
    else:
        gender_presentation = None
    
    # Face shape (simplified - would need landmarks for accuracy)
    face_width = x2 - x1
    face_height = y2 - y1
    ratio = face_height / face_width if face_width > 0 else 1.0
    
    if ratio > 1.4:
        face_shape = "oval"
    elif ratio < 1.1:
        face_shape = "round"
    else:
        face_shape = "oval"  # Default
    
    return FaceObserved(
        face_shape=face_shape,
        hair_color=hair_color,
        eye_color=eye_color,
        nose_bridge="medium",  # Placeholder
        lip_fullness="medium",  # Placeholder
        skin_tone=skin_tone,
        age_appearance=age_appearance,
        gender_presentation=gender_presentation
    )


@app.on_event("startup")
async def startup() -> None:
    global _pipe, _face_analyzer
    t0 = time.time()
    
    # Load diffusion model
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
    
    # Load InsightFace for feature extraction
    t1 = time.time()
    logger.info("loading_face_analyzer", extra={"extra_fields": {"model": "buffalo_l"}})
    _face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("loaded_face_analyzer", extra={"extra_fields": {"seconds": round(time.time() - t1, 2)}})


def _profile_prompt(features: dict) -> str:
    """Build preset prompt from features dict (not model instance)."""
    observed = features.get("observed", {})
    bits = [f"{k.replace('_',' ')} {v}" for k, v in observed.items() if v]
    desc = ", ".join(bits) if bits else "pleasant face"
    return f"High quality profile portrait, {desc}, Disney 3D animated style, front-facing view, centered face, direct gaze, cinematic lighting, expressive eyes, professional studio quality"


def _extract_features_from_image(content: bytes) -> FaceProfileFeaturesV1:
    if not content or len(content) < 1024:
        error = {
            "code": "BAD_SELFIE",
            "message": "Selfie quality check failed",
            "details": {"reason": "NO_FACE_DETECTED", "quality_score": 0.0, "num_faces": 0},
        }
        raise HTTPException(status_code=422, detail=error)

    try:
        img_bgr, w, h = _load_image_bytes(content)
    except Exception:
        error = {
            "code": "BAD_SELFIE",
            "message": "Selfie quality check failed",
            "details": {"reason": "INVALID_IMAGE", "quality_score": 0.0, "num_faces": 0},
        }
        raise HTTPException(status_code=422, detail=error)

    # Use InsightFace for detection and analysis
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = _face_analyzer.get(img_rgb)
    
    num_faces = len(faces)
    if num_faces != 1:
        error = {
            "code": "BAD_SELFIE",
            "message": "Selfie quality check failed",
            "details": {"reason": "MULTIPLE_FACES" if num_faces > 1 else "NO_FACE_DETECTED", "quality_score": 0.0, "num_faces": num_faces},
        }
        raise HTTPException(status_code=422, detail=error)

    face = faces[0]
    
    # Quality check based on detection score and image metrics
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = _blur_score(gray)
    bright = _brightness(gray)
    
    quality = 0.0
    quality += min(1.0, blur / 150.0) * 0.4
    quality += (1.0 - min(1.0, abs(bright - 128.0) / 128.0)) * 0.3
    quality += min(1.0, float(face.det_score)) * 0.3  # Detection confidence
    
    if quality < 0.35:
        error = {
            "code": "BAD_SELFIE",
            "message": "Selfie quality check failed",
            "details": {"reason": "LOW_QUALITY", "quality_score": quality, "num_faces": 1},
        }
        raise HTTPException(status_code=422, detail=error)

    # Extract facial features using InsightFace
    observed = _extract_facial_features_with_insightface(img_bgr, face)
    
    features = FaceProfileFeaturesV1(
        observed=observed,
        meta=FaceMeta(face_detected=True, num_faces=1, quality_score=float(round(quality, 3))),
    )
    return features


@app.post("/v1/profile/analyze", response_class=JSONResponse)
async def analyze_selfie(selfie: UploadFile = File(None), file: UploadFile = File(None)) -> JSONResponse:
    """Extract features from selfie image."""
    if selfie is None and file is None:
        raise HTTPException(status_code=400, detail="Either 'selfie' or 'file' field required")
    content = await selfie.read() if selfie is not None else await file.read()  # type: ignore[union-attr]
    features = _extract_features_from_image(content)
    return JSONResponse({"avatar_features": features.model_dump()})


@app.post("/v1/profile/generate", response_class=JSONResponse)
async def generate_from_features(avatar_features: dict = Body(...)) -> JSONResponse:
    """Generate profile image from features JSON. Returns JSON with base64 PNG."""
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = _profile_prompt(avatar_features)
    out = _pipe(prompt=prompt, height=512, width=512, num_inference_steps=4, guidance_scale=0.0)
    image = out.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    return JSONResponse({"image_bytes_b64": base64.b64encode(image_bytes).decode()})


@app.post("/")
async def root_compat(selfie: UploadFile = File(None), file: UploadFile = File(None)) -> JSONResponse:
    """Root endpoint for back-compat: analyze + generate."""
    if selfie is None and file is None:
        raise HTTPException(status_code=400, detail="Either 'selfie' or 'file' field required")
    content = await selfie.read() if selfie is not None else await file.read()  # type: ignore[union-attr]
    features = _extract_features_from_image(content)

    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = _profile_prompt(features.model_dump())
    out = _pipe(prompt=prompt, height=512, width=512, num_inference_steps=4, guidance_scale=0.0)
    image = out.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    payload: Dict[str, object] = {
        "avatar_features": features.model_dump(),
        "image_bytes_b64": base64.b64encode(image_bytes).decode(),
    }
    return JSONResponse(payload)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "model_loaded": _pipe is not None})
