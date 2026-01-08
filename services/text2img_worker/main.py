import io
import time
from typing import Dict

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image

from shared.logging_config import get_logger
from shared.models import WorkerText2ImgRequest
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.text2img_worker_settings()


async def generate_placeholder_image(prompt: str, width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 180, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/", response_class=Response)
async def generate(request: WorkerText2ImgRequest = Body(...)) -> Response:
    start = time.time()
    image_bytes = await generate_placeholder_image(request.prompt, request.width, request.height)
    if len(image_bytes) < 100:
        # Simulate garbled output
        raise HTTPException(status_code=500, detail="Garbled output")
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info("text2img_generated", extra_fields={"elapsed_ms": elapsed_ms, "prompt_len": len(request.prompt)})
    return Response(content=image_bytes, media_type="image/png")
