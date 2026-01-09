import os
import time
from typing import Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
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

DEVICE = os.getenv("DEVICE", "cuda")
VQA_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

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
    
    # Define questions and valid options
    questions_and_options = {
        "hair_color": ("What is this person's hair color? Choose only one: black, brown, blonde, red, gray, white", 
                      ["black", "brown", "blonde", "red", "gray", "white"]),
        "hair_type": ("What is the hair texture? Choose only one: straight, wavy, curly, coily, afro", 
                     ["straight", "wavy", "curly", "coily", "afro"]),
        "hair_length": ("What is the hair length? Choose only one: bald, very-short, short, medium, long", 
                       ["bald", "very-short", "short", "medium", "long"]),
        "eye_color": ("What is the eye color? Choose only one: brown, blue, green, hazel, gray", 
                     ["brown", "blue", "green", "hazel", "gray"]),
        "skin_tone": ("What is the skin color? Choose only one: white, brown, black, tan, beige, olive", 
                     ["white", "brown", "black", "tan", "beige", "olive"]),
        "age_appearance": ("What age group? Choose only one: young, middle-aged, senior", 
                          ["young", "middle-aged", "senior"]),
        "gender": ("Is this person male or female? Answer only: male or female", 
                  ["male", "female"]),
        "facial_hair": ("What facial hair is visible? Choose only one: none, stubble, mustache, beard, goatee", 
                       ["none", "stubble", "mustache", "beard", "goatee"]),
        "face_shape": ("What is the face shape? Choose only one: oval, round, square, heart, diamond", 
                      ["oval", "round", "square", "heart", "diamond"]),
        "dress_color": ("What is the main clothing color? Choose only one: black, white, blue, red, green, yellow, gray, brown, pink, purple", 
                       ["black", "white", "blue", "red", "green", "yellow", "gray", "brown", "pink", "purple"]),
        "dress_type": ("What type of clothing? Choose only one: shirt, t-shirt, dress, suit, jacket, sweater, hoodie, blouse", 
                      ["shirt", "t-shirt", "dress", "suit", "jacket", "sweater", "hoodie", "blouse"])
    }
    
    # Process questions sequentially for reliability
    features = {}
    
    for key, (question, valid_options) in questions_and_options.items():
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": question}
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
                output_ids = _vqa_model.generate(**inputs, max_new_tokens=8)
            
            # Decode answer
            generated_ids = output_ids[0][len(inputs.input_ids[0]):]
            answer = _vqa_processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # Smart extraction: find valid option in the answer
            answer_lower = answer.lower().strip()
            found_option = None
            for option in valid_options:
                if option in answer_lower:
                    found_option = option
                    break
            
            features[key] = found_option
                
        except Exception as e:
            logger.warning("vqa_feature_failed", extra={"extra_fields": {"feature": key, "error": str(e)}})
            features[key] = None
    
    t_elapsed = time.time() - t_start
    logger.info("vqa_extraction_complete", extra={"extra_fields": {"seconds": round(t_elapsed, 2), "num_questions": len(questions_and_options)}})
    
    # Split into face and dress features
    face_features = {k: v for k, v in features.items() if k not in ['dress_color', 'dress_type']}
    dress_features = {k: v for k, v in features.items() if k in ['dress_color', 'dress_type']}
    
    return face_features, dress_features


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

    # Extract facial and dress features using VQA (timed internally)
    face_features, dress_features = _extract_facial_features_with_vqa(img_bgr)
    
    from shared.models import DressObserved
    features = FaceProfileFeaturesV1(
        observed=FaceObserved(**face_features),
        dress=DressObserved(**dress_features),
        meta=FaceMeta(face_detected=True, num_faces=1, quality_score=float(round(quality, 3))),
    )
    
    t_elapsed = time.time() - t_total
    logger.info("feature_extraction_total", extra={"extra_fields": {"seconds": round(t_elapsed, 2)}})
    return features


@app.on_event("startup")
async def startup() -> None:
    global _vqa_model, _vqa_processor
    t0 = time.time()
    
    # Load Qwen2-VL for VQA
    logger.info("loading_vqa_model", extra={"extra_fields": {"model_id": VQA_MODEL_ID}})
    _vqa_model = Qwen2VLForConditionalGeneration.from_pretrained(
        VQA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    _vqa_processor = AutoProcessor.from_pretrained(VQA_MODEL_ID)
    logger.info("loaded_vqa_model", extra={"extra_fields": {"seconds": round(time.time() - t0, 2)}})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "selfie_feature_worker"})


@app.post("/v1/profile/analyze", response_class=JSONResponse)
async def analyze_selfie(selfie: UploadFile = File(None), file: UploadFile = File(None)) -> JSONResponse:
    """Extract features from selfie image."""
    if selfie is None and file is None:
        raise HTTPException(status_code=400, detail="Either 'selfie' or 'file' field required")
    content = await selfie.read() if selfie is not None else await file.read()  # type: ignore[union-attr]
    features = _extract_features_from_image(content)
    return JSONResponse({"avatar_features": features.model_dump()})
