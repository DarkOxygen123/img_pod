import os
import time
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from ultralytics import YOLO

from shared.logging_config import get_logger
from shared.models import FaceMeta, FaceObserved, DressObserved, AccessoriesObserved, FaceProfileFeaturesV1
from shared.settings import config

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None

app = FastAPI()
logger = get_logger(__name__)
settings = config.profile_worker_settings()

DEVICE = os.getenv("DEVICE", "cuda")
VQA_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  # Upgraded to 7B model

_vqa_model = None
_vqa_processor = None
_face_detector = None
_mp_face_detector = None


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

    def _ask(question: str, valid_options: List[str]) -> Optional[str]:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = _vqa_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = _vqa_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                output_ids = _vqa_model.generate(**inputs, max_new_tokens=8)

            generated_ids = output_ids[0][len(inputs.input_ids[0]) :]
            answer = _vqa_processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            answer_lower = answer.lower().strip()
            for option in valid_options:
                if option in answer_lower:
                    return option
            return None
        except Exception as e:
            logger.warning("vqa_feature_failed", extra={"extra_fields": {"error": str(e), "question": question}})
            return None
    
    # Questions and valid options.
    # Notes:
    # - Keep answers discrete to reduce parsing errors.
    # - Skip dependent questions to reduce runtime (hat/glasses/mask details only when present, etc.).
    # - Preserve existing keys while incorporating recommended ones.
    features: dict = {}

    # --- Core identity ---
    features["gender"] = _ask(
        "What is the person's gender presentation? Choose only one: male, female, uncertain",
        ["male", "female", "uncertain"],
    )
    features["age_range"] = _ask(
        "What age range does this person appear to be in? Choose only one: under-25, 25-34, 35-44, 45-54, 55+",
        ["under-25", "25-34", "35-44", "45-54", "55+"],
    )

    # Keep legacy field (used by some downstream prompt logic); derive when possible.
    age_map = {
        "under-25": "young",
        "25-34": "adult",
        "35-44": "adult",
        "45-54": "middle-aged",
        "55+": "senior",
    }
    if features.get("age_range") in age_map:
        features["age_appearance"] = age_map[features["age_range"]]
    else:
        features["age_appearance"] = _ask(
            "What age group? Choose only one: young, adult, middle-aged, senior",
            ["young", "adult", "middle-aged", "senior"],
        )

    # --- Skin ---
    features["skin_tone"] = _ask(
        "What is the skin tone? Choose only one: fair, light-brown, medium-brown, dark-brown, black",
        ["fair", "light-brown", "medium-brown", "dark-brown", "black"],
    )
    features["skin_undertone"] = _ask(
        "What is the skin undertone? Choose only one: cool, neutral, warm, olive",
        ["cool", "neutral", "warm", "olive"],
    )

    # --- Eyes ---
    features["eye_color"] = _ask(
        "What is the eye color? Choose only one: brown, dark-brown, hazel, green, blue, gray",
        ["brown", "dark-brown", "hazel", "green", "blue", "gray"],
    )
    features["eye_shape"] = _ask(
        "What is the eye shape? Choose only one: almond, round, hooded, monolid, upturned, downturned",
        ["almond", "round", "hooded", "monolid", "upturned", "downturned"],
    )

    # --- Face structure ---
    features["face_shape"] = _ask(
        "What is the face shape? Choose only one: oval, round, square, heart, diamond",
        ["oval", "round", "square", "heart", "diamond"],
    )

    # --- Hair ---
    features["hair_color"] = _ask(
        "What is the hair color? Choose only one: black, dark-brown, brown, light-brown, blonde, red, gray, white",
        ["black", "dark-brown", "brown", "light-brown", "blonde", "red", "gray", "white"],
    )
    features["hair_type"] = _ask(
        "What is the hair texture? Choose only one: straight, wavy, curly, coily",
        ["straight", "wavy", "curly", "coily"],
    )
    features["hair_length"] = _ask(
        "What is the hair length? Choose only one: bald, very-short, short, medium, long",
        ["bald", "very-short", "short", "medium", "long"],
    )
    features["hair_style"] = _ask(
        "What is the hairstyle? Choose only one: side-part, middle-part, slicked-back, undercut, fade, quiff, pompadour, short-crop, buzz-cut, curly-open, long-open, bun, top-knot, ponytail, braid",
        [
            "side-part",
            "middle-part",
            "slicked-back",
            "undercut",
            "fade",
            "quiff",
            "pompadour",
            "short-crop",
            "buzz-cut",
            "curly-open",
            "long-open",
            "bun",
            "top-knot",
            "ponytail",
            "braid",
        ],
    )
    features["hairline_type"] = _ask(
        "What is the hairline type? Choose only one: straight, rounded, widow-peak, receding-mild, receding-deep",
        ["straight", "rounded", "widow-peak", "receding-mild", "receding-deep"],
    )
    features["balding_pattern"] = _ask(
        "If balding is visible, what pattern best fits? Choose only one: none, temples, crown, front-top, full-top, horseshoe",
        ["none", "temples", "crown", "front-top", "full-top", "horseshoe"],
    )

    # --- Facial hair ---
    features["facial_hair"] = _ask(
        "What facial hair is visible? Choose only one: none, stubble, mustache, chevron-mustache, handlebar-mustache, goatee, short-boxed-beard, full-beard, long-beard",
        [
            "none",
            "stubble",
            "mustache",
            "chevron-mustache",
            "handlebar-mustache",
            "goatee",
            "short-boxed-beard",
            "full-beard",
            "long-beard",
        ],
    )
    gender = (features.get("gender") or "").strip()
    if gender == "female":
        # Prevent female prompts from accidentally gaining facial hair.
        features["facial_hair"] = "none"
        features["facial_hair_density"] = "none"
    else:
        if features.get("facial_hair") and features.get("facial_hair") != "none":
            features["facial_hair_density"] = _ask(
                "What is the facial hair density? Choose only one: light, medium, dense",
                ["light", "medium", "dense"],
            )
        else:
            features["facial_hair_density"] = "none"

    # --- Facial marks ---
    features["facial_marks"] = _ask(
        "Are there visible facial marks? Choose only one: none, mole, freckles, scar, acne",
        ["none", "mole", "freckles", "scar", "acne"],
    )
    if features.get("facial_marks") and features.get("facial_marks") != "none":
        features["facial_mark_position"] = _ask(
            "If facial marks exist, where are they located? Choose only one: forehead, left-cheek, right-cheek, nose, upper-lip, chin, jawline",
            ["forehead", "left-cheek", "right-cheek", "nose", "upper-lip", "chin", "jawline"],
        )
    else:
        features["facial_mark_position"] = "none"

    # --- Expression ---
    features["expression"] = _ask(
        "What is the facial expression? Choose only one: neutral, slight-smile, smile, serious",
        ["neutral", "slight-smile", "smile", "serious"],
    )

    # --- Dress ---
    features["dress_color"] = _ask(
        "What is the main clothing color? Choose only one: black, white, blue, red, green, yellow, gray, brown, pink, purple",
        ["black", "white", "blue", "red", "green", "yellow", "gray", "brown", "pink", "purple"],
    )
    features["dress_type"] = _ask(
        "What type of clothing? Choose only one: shirt, t-shirt, kurta, saree, dress, suit, jacket, sweater, hoodie, blouse",
        ["shirt", "t-shirt", "kurta", "saree", "dress", "suit", "jacket", "sweater", "hoodie", "blouse"],
    )

    # --- Accessories (ask presence first; ask details only when present) ---
    features["hat_present"] = _ask("Is the person wearing a cap or hat? Answer: yes or no", ["yes", "no"])
    if features.get("hat_present") == "yes":
        features["hat_style"] = _ask(
            "If wearing a cap/hat, what style? Choose one: baseball-cap, beanie, fedora, cowboy-hat, sun-hat, beret",
            ["baseball-cap", "beanie", "fedora", "cowboy-hat", "sun-hat", "beret"],
        )
        features["hat_color"] = _ask(
            "If wearing a cap/hat, what color? Choose one: black, white, blue, red, green, gray, brown, tan",
            ["black", "white", "blue", "red", "green", "gray", "brown", "tan"],
        )
    else:
        features["hat_style"] = "none"
        features["hat_color"] = "none"

    features["glasses_present"] = _ask("Is the person wearing glasses? Answer: yes or no", ["yes", "no"])
    if features.get("glasses_present") == "yes":
        features["glasses_type"] = _ask(
            "If wearing glasses, what type? Choose one: reading-glasses, sunglasses, aviators, round-glasses, square-glasses",
            ["reading-glasses", "sunglasses", "aviators", "round-glasses", "square-glasses"],
        )
        features["glasses_color"] = _ask(
            "If wearing glasses, what frame color? Choose one: black, brown, gold, silver, clear, tortoise",
            ["black", "brown", "gold", "silver", "clear", "tortoise"],
        )
    else:
        features["glasses_type"] = "none"
        features["glasses_color"] = "none"

    features["mask_present"] = _ask("Is the person wearing a face mask? Answer: yes or no", ["yes", "no"])
    if features.get("mask_present") == "yes":
        features["mask_type"] = _ask(
            "If wearing a mask, what type? Choose one: surgical, cloth, n95, bandana",
            ["surgical", "cloth", "n95", "bandana"],
        )
        features["mask_color"] = _ask(
            "If wearing a mask, what color? Choose one: white, blue, black, gray, patterned",
            ["white", "blue", "black", "gray", "patterned"],
        )
    else:
        features["mask_type"] = "none"
        features["mask_color"] = "none"
    
    t_elapsed = time.time() - t_start
    logger.info("vqa_extraction_complete", extra={"extra_fields": {"seconds": round(t_elapsed, 2), "num_questions": len(questions_and_options)}})

    # Post-processing cleanup: ensure dependent fields have usable values.
    if features.get("hat_present") != "yes":
        features["hat_style"] = "none"
        features["hat_color"] = "none"
    if features.get("glasses_present") != "yes":
        features["glasses_type"] = "none"
        features["glasses_color"] = "none"
    if features.get("mask_present") != "yes":
        features["mask_type"] = "none"
        features["mask_color"] = "none"
    
    # Split into face, dress, and accessory features
    face_features = {
        k: v
        for k, v in features.items()
        if k
        not in [
            "dress_color",
            "dress_type",
            "hat_present",
            "hat_style",
            "hat_color",
            "glasses_present",
            "glasses_type",
            "glasses_color",
            "mask_present",
            "mask_type",
            "mask_color",
        ]
    }
    dress_features = {k: v for k, v in features.items() if k in ["dress_color", "dress_type"]}
    accessory_features = {
        k: v
        for k, v in features.items()
        if k
        in [
            "hat_present",
            "hat_style",
            "hat_color",
            "glasses_present",
            "glasses_type",
            "glasses_color",
            "mask_present",
            "mask_type",
            "mask_color",
        ]
    }
    
    return face_features, dress_features, accessory_features


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _nms(boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.5) -> List[Tuple[int, int, int, int]]:
    # Simple NMS by area (largest first).
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for box in boxes_sorted:
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


def _detect_faces_mediapipe(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    global _mp_face_detector
    if _mp_face_detector is None:
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = _mp_face_detector.process(img_rgb)
    if not res or not res.detections:
        return []
    h, w = img_bgr.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    for det in res.detections:
        score = float(det.score[0]) if det.score else 0.0
        if score < 0.6:
            continue
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin * w)
        y1 = int(bb.ymin * h)
        bw = int(bb.width * w)
        bh = int(bb.height * h)
        x1 = max(0, x1)
        y1 = max(0, y1)
        bw = max(1, min(w - x1, bw))
        bh = max(1, min(h - y1, bh))
        boxes.append((x1, y1, bw, bh))
    return _nms(boxes, iou_thresh=0.4)


def _detect_faces_yolo(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using YOLOv8-Face for robust detection."""
    global _face_detector
    if _face_detector is None:
        return []
    
    results = _face_detector(img_bgr, verbose=False)
    faces = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                faces.append((x1, y1, x2 - x1, y2 - y1))  # Convert to (x, y, w, h) format
    
    return faces


def _detect_faces_opencv_fallback(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Fallback face detection using OpenCV."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return [tuple(f) for f in faces]


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

    # Resize large images for faster VQA processing
    max_dimension = 1024
    if max(h, w) > max_dimension:
        orig_w, orig_h = w, h
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        logger.info(
            "image_resized",
            extra={"extra_fields": {"original_size": f"{orig_w}x{orig_h}", "new_size": f"{new_w}x{new_h}"}},
        )

    # Face detection priority:
    # 1) MediaPipe (best for reducing false positives)
    # 2) YOLO face model (only if configured)
    # 3) OpenCV Haar cascade fallback
    faces = _detect_faces_mediapipe(img_bgr)
    if not faces:
        faces = _detect_faces_yolo(img_bgr)
    if not faces:
        logger.info("no_faces_mediapipe_or_yolo_trying_opencv")
        faces = _detect_faces_opencv_fallback(img_bgr)
    
    num_faces = len(faces)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Robust handling of edge cases
    if num_faces == 0:
        # NO FACE DETECTED - use full image with lower quality score
        logger.warning("no_face_detected_using_full_image")
        img_bgr_cropped = img_bgr
        quality_penalty = 0.3  # Lower quality score for no-face images
    elif num_faces == 1:
        # SINGLE FACE - ideal case, crop with padding
        (x, y, fw, fh) = faces[0]
        padding_factor = 1.5  # 50% padding around face
        pad_w = int(fw * (padding_factor - 1) / 2)
        pad_h = int(fh * (padding_factor - 1) / 2)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + fw + pad_w)
        y2 = min(h, y + fh + pad_h)
        
        img_bgr_cropped = img_bgr[y1:y2, x1:x2]
        logger.info("face_cropped", extra={"extra_fields": {"original": f"{w}x{h}", "cropped": f"{x2-x1}x{y2-y1}"}})
        quality_penalty = 1.0  # No penalty for single face
    else:
        # MULTIPLE FACES - pick largest face and warn
        logger.warning("multiple_faces_picking_largest", extra={"extra_fields": {"num_faces": num_faces}})
        largest_face = max(faces, key=lambda f: f[2] * f[3])  # Largest area
        (x, y, fw, fh) = largest_face
        padding_factor = 1.5
        pad_w = int(fw * (padding_factor - 1) / 2)
        pad_h = int(fh * (padding_factor - 1) / 2)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + fw + pad_w)
        y2 = min(h, y + fh + pad_h)
        
        img_bgr_cropped = img_bgr[y1:y2, x1:x2]
        quality_penalty = 0.7  # Penalty for multiple faces
    
    # Quality check on cropped/full image
    if num_faces > 0:
        gray_region = gray[y1:y2, x1:x2]
    else:
        gray_region = gray
    
    blur = _blur_score(gray_region)
    bright = _brightness(gray_region)
    
    quality = 0.0
    quality += min(1.0, blur / 150.0) * 0.6
    quality += (1.0 - min(1.0, abs(bright - 128.0) / 128.0)) * 0.4
    quality *= quality_penalty  # Apply face detection penalty
    
    # Only reject if REALLY bad quality
    if quality < 0.2:
        error = {
            "code": "BAD_SELFIE",
            "message": "Selfie quality check failed",
            "details": {"reason": "VERY_LOW_QUALITY", "quality_score": quality, "num_faces": num_faces},
        }
        raise HTTPException(status_code=422, detail=error)

    # Extract facial, dress, and accessory features using VQA (timed internally)
    face_features, dress_features, accessory_features = _extract_facial_features_with_vqa(img_bgr_cropped)
    
    from shared.models import DressObserved, AccessoriesObserved
    features = FaceProfileFeaturesV1(
        observed=FaceObserved(**face_features),
        dress=DressObserved(**dress_features),
        accessories=AccessoriesObserved(**accessory_features),
        meta=FaceMeta(face_detected=(num_faces > 0), num_faces=num_faces, quality_score=float(round(quality, 3))),
    )
    
    t_elapsed = time.time() - t_total
    logger.info("feature_extraction_total", extra={"extra_fields": {"seconds": round(t_elapsed, 2)}})
    return features


@app.on_event("startup")
async def startup() -> None:
    global _vqa_model, _vqa_processor, _face_detector, _mp_face_detector
    t0 = time.time()
    
    # Load MediaPipe face detector (primary)
    if mp is not None:
        try:
            _mp_face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
            logger.info("loaded_mediapipe_face_detector")
        except Exception as e:
            logger.warning("mediapipe_face_detector_load_failed", extra={"extra_fields": {"error": str(e)}})
            _mp_face_detector = None

    # Optional: Load YOLO face detector if a face-specific weights file is configured.
    try:
        yolo_weights = os.getenv("YOLO_FACE_MODEL_PATH")
        if yolo_weights:
            logger.info("loading_yolo_face", extra={"extra_fields": {"weights": yolo_weights}})
            _face_detector = YOLO(yolo_weights)
            logger.info("loaded_yolo_face", extra={"extra_fields": {"seconds": round(time.time() - t0, 2)}})
        else:
            _face_detector = None
    except Exception as e:
        logger.warning("yolo_face_load_failed", extra={"extra_fields": {"error": str(e)}})
        _face_detector = None
    
    # Load Qwen2-VL-7B for VQA
    logger.info("loading_vqa_model", extra={"extra_fields": {"model_id": VQA_MODEL_ID}})
    t_vqa = time.time()
    _vqa_model = Qwen2VLForConditionalGeneration.from_pretrained(
        VQA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    _vqa_processor = AutoProcessor.from_pretrained(VQA_MODEL_ID)
    logger.info("loaded_vqa_model", extra={"extra_fields": {"seconds": round(time.time() - t_vqa, 2)}})
    logger.info("startup_complete", extra={"extra_fields": {"total_seconds": round(time.time() - t0, 2)}})


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
