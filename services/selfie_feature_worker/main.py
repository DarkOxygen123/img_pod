import os
import time
from typing import Tuple, Optional, List
import json

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

    def _extract_json_from_text(text: str) -> Optional[dict]:
        if not text:
            return None
        s = text.strip()
        # Heuristic: grab the outermost JSON object.
        i = s.find("{")
        j = s.rfind("}")
        if i == -1 or j == -1 or j <= i:
            return None
        candidate = s[i : j + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def _batch_vqa(schema: dict) -> dict:
        """Ask for all fields in one VQA call, returning a validated features dict."""
        # Build a compact instruction prompt.
        lines = [
            "Return ONLY a JSON object.",
            "Rules:",
            "- Keys must exactly match the requested keys.",
            "- Values must be one of the allowed options for that key, or null.",
            "- Do not add extra keys.",
            "- No explanation text.",
            "Keys and allowed options:",
        ]
        for k, opts in schema.items():
            lines.append(f"- {k}: {', '.join(opts)}")
        instruction = "\n".join(lines)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": instruction},
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
            # Single generation for all fields.
            output_ids = _vqa_model.generate(**inputs, max_new_tokens=256)

        generated_ids = output_ids[0][len(inputs.input_ids[0]) :]
        raw = _vqa_processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        parsed = _extract_json_from_text(raw)
        if not isinstance(parsed, dict):
            raise ValueError("batch_vqa_parse_failed")

        validated: dict = {}
        for k, opts in schema.items():
            v = parsed.get(k)
            if v is None:
                validated[k] = None
                continue
            if isinstance(v, str):
                vv = v.strip().lower()
                validated[k] = vv if vv in opts else None
            else:
                validated[k] = None
        return validated
    
    # Prefer a single batched VQA call to avoid proxy/interface timeouts.
    schema = {
        "gender": ["male", "female", "uncertain"],
        "age_range": ["under-25", "25-34", "35-44", "45-54", "55+"],
        "skin_tone": ["fair", "light-brown", "medium-brown", "dark-brown", "black"],
        "skin_undertone": ["cool", "neutral", "warm", "olive"],
        "eye_color": ["brown", "dark-brown", "hazel", "green", "blue", "gray"],
        "eye_shape": ["almond", "round", "hooded", "monolid", "upturned", "downturned"],
        "face_shape": ["oval", "round", "square", "heart", "diamond"],
        "hair_color": ["black", "dark-brown", "brown", "light-brown", "blonde", "red", "gray", "white"],
        "hair_type": ["straight", "wavy", "curly", "coily"],
        "hair_length": ["bald", "very-short", "short", "medium", "long"],
        "hair_style": [
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
        "hairline_type": ["straight", "rounded", "widow-peak", "receding-mild", "receding-deep"],
        "balding_pattern": ["none", "temples", "crown", "front-top", "full-top", "horseshoe"],
        "facial_hair": [
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
        "facial_hair_density": ["none", "light", "medium", "dense"],
        "facial_marks": ["none", "mole", "freckles", "scar", "acne"],
        "facial_mark_position": ["none", "forehead", "left-cheek", "right-cheek", "nose", "upper-lip", "chin", "jawline"],
        "expression": ["neutral", "slight-smile", "smile", "serious"],
        "dress_color": ["black", "white", "blue", "red", "green", "yellow", "gray", "brown", "pink", "purple"],
        "dress_type": ["shirt", "t-shirt", "kurta", "saree", "dress", "suit", "jacket", "sweater", "hoodie", "blouse"],
        "hat_present": ["yes", "no"],
        "hat_style": ["none", "baseball-cap", "beanie", "fedora", "cowboy-hat", "sun-hat", "beret"],
        "hat_color": ["none", "black", "white", "blue", "red", "green", "gray", "brown", "tan"],
        "glasses_present": ["yes", "no"],
        "glasses_type": ["none", "reading-glasses", "sunglasses", "aviators", "round-glasses", "square-glasses"],
        "glasses_color": ["none", "black", "brown", "gold", "silver", "clear", "tortoise"],
        "mask_present": ["yes", "no"],
        "mask_type": ["none", "surgical", "cloth", "n95", "bandana"],
        "mask_color": ["none", "white", "blue", "black", "gray", "patterned"],
    }

    try:
        features: dict = _batch_vqa(schema)
    except Exception as e:
        # Fallback: ask a minimal set only (best-effort) rather than timing out.
        logger.warning("batch_vqa_failed_falling_back_minimal", extra={"extra_fields": {"error": str(e)}})
        features = {}
        features["gender"] = _ask(
            "What is the person's gender presentation? Choose only one: male, female, uncertain",
            ["male", "female", "uncertain"],
        )
        features["age_range"] = _ask(
            "What age range does this person appear to be in? Choose only one: under-25, 25-34, 35-44, 45-54, 55+",
            ["under-25", "25-34", "35-44", "45-54", "55+"],
        )
        features["skin_tone"] = _ask(
            "What is the skin tone? Choose only one: fair, light-brown, medium-brown, dark-brown, black",
            ["fair", "light-brown", "medium-brown", "dark-brown", "black"],
        )
        features["eye_color"] = _ask(
            "What is the eye color? Choose only one: brown, dark-brown, hazel, green, blue, gray",
            ["brown", "dark-brown", "hazel", "green", "blue", "gray"],
        )
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
        features["facial_hair"] = _ask(
            "What facial hair is visible? Choose only one: none, stubble, mustache, goatee, full-beard",
            ["none", "stubble", "mustache", "goatee", "full-beard"],
        )
        features["facial_marks"] = _ask(
            "Are there visible facial marks? Choose only one: none, mole, freckles, scar",
            ["none", "mole", "freckles", "scar"],
        )
        features["dress_type"] = _ask(
            "What type of clothing? Choose only one: shirt, t-shirt, kurta, saree, dress, suit",
            ["shirt", "t-shirt", "kurta", "saree", "dress", "suit"],
        )
        features["hat_present"] = _ask("Is the person wearing a cap or hat? Answer: yes or no", ["yes", "no"])
        features["glasses_present"] = _ask("Is the person wearing glasses? Answer: yes or no", ["yes", "no"])
        features["mask_present"] = _ask("Is the person wearing a face mask? Answer: yes or no", ["yes", "no"])
    
    t_elapsed = time.time() - t_start
    logger.info(
        "vqa_extraction_complete",
        extra={"extra_fields": {"seconds": round(t_elapsed, 2), "num_questions": len(features)}},
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

    # Post-processing cleanup: ensure dependent fields have usable values.
    gender = (features.get("gender") or "").strip()
    if gender == "female":
        features["facial_hair"] = "none"
        features["facial_hair_density"] = "none"

    if features.get("facial_marks") in (None, "none"):
        features["facial_mark_position"] = "none"

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
