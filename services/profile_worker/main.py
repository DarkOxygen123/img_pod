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
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from shared.logging_config import get_logger
from shared.models import FaceMeta, FaceObserved, FaceProfileFeaturesV1
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.profile_worker_settings()

MODEL_ID = os.getenv("PROFILE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEVICE = os.getenv("PROFILE_DEVICE", "cuda")
VQA_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

_pipe = None
_vqa_model = None
_vqa_processor = None


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


def _extract_facial_features_with_vqa(img_bgr: np.ndarray) -> FaceObserved:
    """Use Qwen2-VL for visual question answering to extract facial features."""
    t_start = time.time()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    questions = {
        "hair_color": "What is the hair color? Answer with one word: black, brown, blonde, red, gray, or white.",
        "hair_type": "What is the hair type? Answer with one word: straight, wavy, curly, or coily.",
        "hair_length": "What is the hair length? Answer with one word: bald, short, medium, or long.",
        "eye_color": "What is the eye color? Answer with one word: brown, blue, green, hazel, or gray.",
        "nose_bridge": "Describe the nose bridge in one word: narrow, medium, or wide.",
        "lip_fullness": "Describe the lip fullness in one word: thin, medium, or full.",
        "skin_tone": "What is the skin tone? Answer with one word: fair, light, medium, tan, or dark.",
        "age_appearance": "What age group does this person appear to be? Answer with one word: young, middle-aged, or senior.",
        "gender_presentation": "What gender presentation? Answer with one word: masculine, feminine, or androgynous.",
        "face_shape": "What is the face shape? Answer with one word: oval, round, square, heart, or diamond.",
        "facial_hair_type": "What type of facial hair? Answer with one word: none, mustache, beard, goatee, or stubble.",
        "facial_hair_length": "If facial hair present, what length? Answer with one word: none, short, medium, or long."
    }
    
    # Process questions one at a time to avoid batch errors
    features = {}
    question_keys = list(questions.keys())
    
    for key in question_keys:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": questions[key]}
                    ]
                }
            ]
            
            text = _vqa_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = _vqa_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                output_ids = _vqa_model.generate(**inputs, max_new_tokens=10)
            
            # Decode answer
            generated_ids = output_ids[0][len(inputs.input_ids[0]):]
            answer = _vqa_processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # Extract first word from answer
            words = answer.lower().strip().split()
            if words:
                features[key] = words[0].rstrip('.,;!?')
            else:
                features[key] = None
                
        except Exception as e:
            logger.warning("vqa_feature_failed", extra={"extra_fields": {"feature": key, "error": str(e)}})
            features[key] = None
    
    t_elapsed = time.time() - t_start
    logger.info("vqa_extraction_complete", extra={"extra_fields": {"seconds": round(t_elapsed, 2), "num_questions": len(questions)}})
    
    return FaceObserved(**features)


@app.on_event("startup")
async def startup() -> None:
    global _pipe, _vqa_model, _vqa_processor
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
    
    # Load Qwen2-VL for VQA
    t1 = time.time()
    logger.info("loading_vqa_model", extra={"extra_fields": {"model_id": VQA_MODEL_ID}})
    _vqa_model = Qwen2VLForConditionalGeneration.from_pretrained(
        VQA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    _vqa_processor = AutoProcessor.from_pretrained(VQA_MODEL_ID)
    logger.info("loaded_vqa_model", extra={"extra_fields": {"seconds": round(time.time() - t1, 2)}})


def _profile_prompt(features: dict) -> str:
    """Build preset prompt from features dict (not model instance)."""
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
    
    # Emphasize adherence to features
    return (f"Highly detailed 3D Disney Pixar style portrait: {desc}. "
            f"IMPORTANT: Must accurately depict ALL specified features including hair texture and color. "
            f"Front-facing view, centered face, direct gaze at camera, cinematic lighting, "
            f"expressive eyes, professional studio quality, photorealistic rendering")


def _extract_features_from_image(content: bytes) -> FaceProfileFeaturesV1:
    t_total = time.time()
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

    # Simple face detection with cv2
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    num_faces = len(faces)
    if num_faces != 1:
        error = {
            "code": "BAD_SELFIE",
            "message": "Selfie quality check failed",
            "details": {"reason": "MULTIPLE_FACES" if num_faces > 1 else "NO_FACE_DETECTED", "quality_score": 0.0, "num_faces": num_faces},
        }
        raise HTTPException(status_code=422, detail=error)
    
    # Quality check
    blur = _blur_score(gray)
    bright = _brightness(gray)
    
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

    # Extract facial features using VQA (timed internally)
    observed = _extract_facial_features_with_vqa(img_bgr)
    
    features = FaceProfileFeaturesV1(
        observed=observed,
        meta=FaceMeta(face_detected=True, num_faces=1, quality_score=float(round(quality, 3))),
    )
    
    t_elapsed = time.time() - t_total
    logger.info("feature_extraction_total", extra={"extra_fields": {"seconds": round(t_elapsed, 2)}})
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
