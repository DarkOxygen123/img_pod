import asyncio
import base64
import uuid
from typing import List, Optional

from shared.http_client import post_json, post_multipart_file
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from shared.ajay import ajay_rule
from shared.logging_config import get_logger
from shared.models import (
    ChatPrivateImageGenRequest,
    ErrorResponse,
    GeneralImageGenRequest,
    PolicyCode,
    ProfileCreateResponse,
    ProfileUpdateRequest,
)
from shared.queue_manager import BoundedQueue, SingleWorker
from shared.settings import config
from shared.policy import evaluate_private_chat, sanitize_general

logger = get_logger(__name__)
app = FastAPI()

interface_settings = config.interface_settings()


class QueueItem:
    def __init__(self, payload: dict, future: asyncio.Future):
        self.payload = payload
        self.future = future


profile_queue = BoundedQueue(capacity=interface_settings.profile_queue_max)
text2img_queue = BoundedQueue(capacity=interface_settings.text2img_queue_max)
worker_index = 0


async def call_profile_worker(payload: dict) -> dict:
    resp = await post_multipart_file(
        str(interface_settings.profile_worker_url),
        field_name="selfie",
        filename=payload["filename"],
        file_bytes=payload["content"],
        content_type=payload["content_type"],
        timeout_s=interface_settings.profile_sla_ms / 1000,
    )
    if resp.status_code == 422:
        return {"_passthrough_status": 422, "_passthrough_json": resp.json()}
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.content.decode(errors="replace"))
    return resp.json()


def pick_text2img_worker() -> str:
    global worker_index
    urls = interface_settings.text2img_worker_urls
    if not urls:
        raise HTTPException(status_code=500, detail="No text2img workers configured")
    url = urls[worker_index % len(urls)]
    worker_index = (worker_index + 1) % len(urls)
    return str(url)


async def call_text2img_worker(payload: dict) -> bytes:
    attempts = interface_settings.regen_attempts
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            resp = await post_json(
                str(payload["worker_url"]),
                payload["body"],
                timeout_s=interface_settings.text2img_sla_ms / 1000,
            )
            if resp.status_code >= 400:
                last_error = HTTPException(status_code=resp.status_code, detail=resp.content.decode(errors="replace"))
                continue
            if len(resp.content) < 512:
                last_error = HTTPException(status_code=424, detail="QUALITY_FAILED: tiny output")
                continue
            return resp.content
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue
    if last_error:
        if isinstance(last_error, HTTPException) and last_error.status_code == 424:
            raise HTTPException(status_code=424, detail=ErrorResponse(code=PolicyCode.QUALITY_FAILED, message="QUALITY_FAILED", details={"attempts": attempts}).model_dump())
        raise HTTPException(status_code=424, detail=ErrorResponse(code=PolicyCode.QUALITY_FAILED, message=str(last_error), details={"attempts": attempts}).model_dump())
    raise HTTPException(status_code=424, detail="QUALITY_FAILED")


async def profile_worker_handler(item: QueueItem) -> None:
    result = await call_profile_worker(item.payload)
    item.future.set_result(result)


async def text2img_worker_handler(item: QueueItem) -> None:
    result = await call_text2img_worker(item.payload)
    item.future.set_result(result)


profile_worker = SingleWorker(profile_queue, profile_worker_handler)
text2img_worker = SingleWorker(text2img_queue, text2img_worker_handler)


@app.on_event("startup")
async def startup_event() -> None:
    await profile_worker.start()
    await text2img_worker.start()


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    # Return structured error payloads without FastAPI wrapping.
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


def retry_after_ms(queue_size: int, sla_ms: int) -> int:
    estimate = queue_size * sla_ms
    return min(estimate, interface_settings.retry_after_cap_ms)


async def enqueue_or_429(queue: BoundedQueue, payload: dict, sla_ms: int, worker: SingleWorker) -> asyncio.Future:
    if queue.full():
        retry_ms = retry_after_ms(queue.size, sla_ms)
        error = ErrorResponse(
            code=PolicyCode.OVERLOADED,
            message="Queue full",
            retry_after_ms=retry_ms,
        )
        raise HTTPException(status_code=429, detail=error.model_dump())
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    try:
        await queue.put(QueueItem(payload, future))
    except asyncio.QueueFull:
        retry_ms = retry_after_ms(queue.size, sla_ms)
        error = ErrorResponse(
            code=PolicyCode.OVERLOADED,
            message="Queue full",
            retry_after_ms=retry_ms,
        )
        raise HTTPException(status_code=429, detail=error.model_dump())
    await worker.start()
    return future


@app.post("/v1/profile/create")
async def profile_create(file: UploadFile = File(...)) -> Response:
    content = await file.read()
    payload = {
        "filename": file.filename,
        "content": content,
        "content_type": file.content_type or "image/png",
    }
    future = await enqueue_or_429(profile_queue, payload, interface_settings.profile_sla_ms, profile_worker)
    try:
        result: dict = await asyncio.wait_for(future, timeout=interface_settings.profile_sla_ms / 1000)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Profile worker timed out")

    if result.get("_passthrough_status") == 422:
        return JSONResponse(status_code=422, content=result["_passthrough_json"])

    json_part = ProfileCreateResponse(avatar_features=result["avatar_features"]).model_dump_json()
    image_bytes = base64.b64decode(result["image_bytes_b64"])
    boundary = f"boundary-{uuid.uuid4().hex}"
    parts: List[bytes] = []
    parts.append(f"--{boundary}\r\nContent-Type: application/json\r\n\r\n{json_part}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Type: image/png\r\nContent-Disposition: attachment; filename=profile.png\r\n\r\n".encode()
        + image_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    return Response(content=body, media_type=f"multipart/mixed; boundary={boundary}")


@app.post("/v1/profile/update")
async def profile_update(body: ProfileUpdateRequest = Body(...)) -> Response:
    # For now, reuse text2img worker with identity prompt.
    prompt = f"Profile portrait of user with traits: {body.avatar_features.model_dump_json()}"
    worker_url = pick_text2img_worker()
    payload = {
        "worker_url": worker_url,
        "body": {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
        },
    }
    future = await enqueue_or_429(text2img_queue, payload, interface_settings.text2img_sla_ms, text2img_worker)
    try:
        image_bytes: bytes = await asyncio.wait_for(future, timeout=interface_settings.text2img_sla_ms / 1000)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Text2Img worker timed out")
    return Response(content=image_bytes, media_type="image/png")


@app.post("/v1/chat/private/imagegen")
async def chat_private_imagegen(body: ChatPrivateImageGenRequest) -> Response:
    decision = evaluate_private_chat(body.new_message.text)
    if decision.blocked:
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(code=PolicyCode.POLICY_BLOCKED, message="POLICY_BLOCKED", details={"reason": decision.reason}).model_dump(),
        )

    # Call LLM service to build PromptBundle
    llm_payload = {
        "participants": body.participants.model_dump(),
        "chat_history": body.history.model_dump(),
        "new_message": body.new_message.model_dump(),
        "mood": {"mood": "neutral", "intensity": 0, "confidence": 0.0, "pending_reply": False, "unreplied_count": 0},
        "style_request": body.style_request.model_dump(),
        "is_general": False,
    }
    llm_resp = await post_json(str(interface_settings.llm_service_url) + "/v1/bundle", llm_payload, timeout_s=10.0)
    if llm_resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"LLM error: {llm_resp.content.decode(errors='replace')}")
    bundle = llm_resp.json()["prompt_bundle"]
    prompt = bundle["final_prompt"]
    worker_url = pick_text2img_worker()
    payload = {
        "worker_url": worker_url,
        "body": {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
        },
    }
    future = await enqueue_or_429(text2img_queue, payload, interface_settings.text2img_sla_ms, text2img_worker)
    try:
        image_bytes: bytes = await asyncio.wait_for(future, timeout=interface_settings.text2img_sla_ms / 1000)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Text2Img worker timed out")
    return Response(content=image_bytes, media_type="image/png")


@app.post("/v1/imagegen")
async def general_imagegen(body: GeneralImageGenRequest) -> Response:
    decision = sanitize_general(body.prompt.text)
    if decision.blocked:
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(code=PolicyCode.POLICY_BLOCKED, message="POLICY_BLOCKED", details={"reason": decision.reason}).model_dump(),
        )

    llm_payload = {
        "participants": body.participants.model_dump(),
        "chat_history": {"messages": []},
        "new_message": {"sent_at": body.prompt.sent_at, "text": decision.sanitized_text or body.prompt.text},
        "mood": {"mood": "neutral", "intensity": 0, "confidence": 0.0, "pending_reply": False, "unreplied_count": 0},
        "style_request": body.style_request.model_dump(),
        "is_general": True,
    }
    llm_resp = await post_json(str(interface_settings.llm_service_url) + "/v1/bundle", llm_payload, timeout_s=10.0)
    if llm_resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"LLM error: {llm_resp.content.decode(errors='replace')}")
    bundle = llm_resp.json()["prompt_bundle"]
    prompt = bundle["final_prompt"]
    worker_url = pick_text2img_worker()
    payload = {
        "worker_url": worker_url,
        "body": {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
        },
    }
    future = await enqueue_or_429(text2img_queue, payload, interface_settings.text2img_sla_ms, text2img_worker)
    try:
        image_bytes: bytes = await asyncio.wait_for(future, timeout=interface_settings.text2img_sla_ms / 1000)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Text2Img worker timed out")
    return Response(content=image_bytes, media_type="image/png")
