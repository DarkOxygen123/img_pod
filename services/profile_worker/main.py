import base64
import io
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from shared.logging_config import get_logger
from shared.models import FaceMeta, FaceObserved, FaceProfileFeaturesV1
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.profile_worker_settings()


async def generate_placeholder_profile() -> bytes:
    img = Image.new("RGB", (512, 512), color=(120, 180, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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

    features = FaceProfileFeaturesV1(
        observed=FaceObserved(face_shape="oval", hair_color="black", eye_color="brown"),
        meta=FaceMeta(face_detected=True, num_faces=1, quality_score=0.85),
    )
    image_bytes = await generate_placeholder_profile()
    payload: Dict[str, object] = {
        "avatar_features": features.model_dump(),
        "image_bytes_b64": base64.b64encode(image_bytes).decode(),
    }
    return JSONResponse(payload)
