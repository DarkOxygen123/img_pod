import base64
import os
import io
import os
import time
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from shared.logging_config import get_logger
from shared.models import FaceMeta, FaceObserved, FaceProfileFeaturesV1
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.profile_worker_settings()

MODEL_ID = os.getenv("PROFILE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEVICE = os.getenv("PROFILE_DEVICE", "cuda")

_pipe = None


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


@app.on_event("startup")
async def startup() -> None:
    global _pipe
    t0 = time.time()
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


def _profile_prompt(features: FaceProfileFeaturesV1) -> str:
    observed = features.observed.model_dump()
    bits = [f"{k.replace('_',' ')} {v}" for k, v in observed.items() if v]
    desc = ", ".join(bits) if bits else "pleasant face"
    return f"High quality profile portrait, {desc}, 3D render, cinematic lighting, eyes visible"


@app.post("/", response_class=JSONResponse)
async def handle_selfie(selfie: UploadFile = File(...)) -> JSONResponse:
    content = await selfie.read()
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

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = _blur_score(gray)
    bright = _brightness(gray)

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6) as fd:
        results = fd.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    detections = results.detections if results and results.detections else []
    num_faces = len(detections)
    if num_faces != 1:
        error = {
            "code": "BAD_SELFIE",
            "message": "Selfie quality check failed",
            "details": {"reason": "MULTIPLE_FACES" if num_faces > 1 else "NO_FACE_DETECTED", "quality_score": 0.0, "num_faces": num_faces},
        }
        raise HTTPException(status_code=422, detail=error)

    # crude quality score: normalize blur+brightness
    quality = 0.0
    quality += min(1.0, blur / 150.0) * 0.6
    quality += (1.0 - min(1.0, abs(bright - 128.0) / 128.0)) * 0.4

    if quality < 0.35:
        error = {
            "code": "BAD_SELFIE",
            "message": "Selfie quality check failed",
            "details": {"reason": "LOW_QUALITY", "quality_score": quality, "num_faces": 1},
        }
        raise HTTPException(status_code=422, detail=error)

    features = FaceProfileFeaturesV1(
        observed=FaceObserved(face_shape="oval"),
        meta=FaceMeta(face_detected=True, num_faces=1, quality_score=float(round(quality, 3))),
    )

    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = _profile_prompt(features)
    out = _pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=9, guidance_scale=0.0)
    image = out.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    payload: Dict[str, object] = {
        "avatar_features": features.model_dump(),
        "image_bytes_b64": base64.b64encode(image_bytes).decode(),
    }
    return JSONResponse(payload)
