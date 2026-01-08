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
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.logging_config import get_logger
from shared.models import FaceMeta, FaceObserved, FaceProfileFeaturesV1
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.profile_worker_settings()

MODEL_ID = os.getenv("PROFILE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEVICE = os.getenv("PROFILE_DEVICE", "cuda")
VLM_MODEL_ID = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"

_pipe = None
_vlm_model = None
_vlm_tokenizer = None


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


def _extract_facial_features_with_vlm(img_bgr: np.ndarray) -> FaceObserved:
    """Use VLM to extract facial features from cropped face."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Ask multiple questions to extract features
    questions = {
        "hair_color": "What is the hair color? Answer with one word: black, brown, blonde, red, gray, or white.",
        "eye_color": "What is the eye color? Answer with one word: brown, blue, green, hazel, or gray.",
        "nose_bridge": "Describe the nose bridge in one word: narrow, medium, or wide.",
        "lip_fullness": "Describe the lip fullness in one word: thin, medium, or full.",
        "skin_tone": "What is the skin tone? Answer with one word: fair, light, medium, tan, or dark.",
        "age_appearance": "What age group does this person appear to be? Answer with one word: young, middle-aged, or senior.",
        "gender_presentation": "What gender presentation? Answer with one word: masculine, feminine, or androgynous.",
        "face_shape": "What is the face shape? Answer with one word: oval, round, square, heart, or diamond."
    }
    
    features = {}
    for key, question in questions.items():
        try:
            enc_image = _vlm_model.encode_image(pil_img)
            answer = _vlm_model.answer_question(enc_image, question, _vlm_tokenizer)
            # Extract first word from answer, lowercase
            words = answer.lower().strip().split()
            if words:
                features[key] = words[0].rstrip('.,;!?')
        except Exception as e:
            logger.warning(f"vlm_feature_extraction_failed", extra={"extra_fields": {"feature": key, "error": str(e)}})
            features[key] = None
    
    return FaceObserved(**features)


@app.on_event("startup")
async def startup() -> None:
    global _pipe, _vlm_model, _vlm_tokenizer
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
    
    # Load VLM for feature extraction
    t1 = time.time()
    logger.info("loading_vlm_model", extra={"extra_fields": {"model_id": VLM_MODEL_ID}})
    _vlm_tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL_ID, revision=VLM_REVISION)
    _vlm_model = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_ID, 
        trust_remote_code=True, 
        revision=VLM_REVISION,
        torch_dtype=torch.float16
    ).to(DEVICE)
    _vlm_model.eval()
    logger.info("loaded_vlm_model", extra={"extra_fields": {"seconds": round(time.time() - t1, 2)}})


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

    # Extract facial features using VLM
    observed = _extract_facial_features_with_vlm(img_bgr)
    
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
