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
    Chat1to1ImageGenRequest,
    Chat1to1ImageGenResponse,
    ShortsImageGenRequest,
    ShortsImageGenResponse,
    ScenesImageGenRequest,
    ScenesImageGenResponse,
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
profile_generate_queue = BoundedQueue(capacity=interface_settings.profile_queue_max)
text2img_queue = BoundedQueue(capacity=interface_settings.text2img_queue_max)
chat1to1_queue = BoundedQueue(capacity=interface_settings.chat1to1_queue_max)
shorts_queue = BoundedQueue(capacity=interface_settings.shorts_queue_max)
scenes_queue = BoundedQueue(capacity=interface_settings.scenes_queue_max)
selfie_worker_index = 0
profile_worker_index = 0
text2img_worker_index = 0
chat1to1_worker_index = 0
shorts_worker_index = 0
scenes_worker_index = 0


def pick_selfie_feature_worker() -> str:
    """Round-robin load balancing for selfie feature workers."""
    global selfie_worker_index
    urls = interface_settings.selfie_feature_worker_urls
    if not urls:
        raise HTTPException(status_code=500, detail="No selfie feature workers configured")
    url = urls[selfie_worker_index % len(urls)]
    selfie_worker_index = (selfie_worker_index + 1) % len(urls)
    return str(url)


def pick_profile_worker() -> str:
    """Round-robin load balancing for profile workers."""
    global profile_worker_index
    urls = interface_settings.profile_worker_urls
    if not urls:
        raise HTTPException(status_code=500, detail="No profile workers configured")
    url = urls[prof


def pick_chat1to1_worker() -> str:
    """Round-robin load balancing for 1:1 chat workers."""
    global chat1to1_worker_index
    urls = interface_settings.chat1to1_worker_urls
    if not urls:


def pick_shorts_worker() -> str:
    """Round-robin load balancing for shorts workers."""
    global shorts_worker_index
    urls = interface_settings.shorts_worker_urls
    if not urls:
        raise HTTPException(status_code=500, detail="No shorts workers configured")
    url = urls[shorts_worker_index % len(urls)]
    shorts_worker_index = (shorts_worker_index + 1) % len(urls)
    return str(url)


def pick_scenes_worker() -> str:
    """Round-robin load balancing for scenes workers."""
    global scenes_worker_index
    urls = interface_settings.scenes_worker_urls
    if not urls:
        raise HTTPException(status_code=500, detail="No scenes workers configured")
    url = urls[scenes_worker_index % len(urls)]
    scenes_worker_index = (scenes_worker_index + 1) % len(urls)
    return str(url)
        raise HTTPException(status_code=500, detail="No 1:1 chat workers configured")
    url = urls[chat1to1_worker_index % len(urls)]
    chat1to1_worker_index = (chat1to1_worker_index + 1) % len(urls)
    return str(url)ile_worker_index % len(urls)]
    profile_worker_index = (profile_worker_index + 1) % len(urls)
    return str(url)


async def call_profile_worker(payload: dict) -> dict:
    # Step 1: analyze selfie to extract features (dedicated VQA service)
    analyze_url = pick_selfie_feature_worker().rstrip("/") + "/v1/profile/analyze"
    resp = await post_multipart_file(
        analyze_url,
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


async def call_profile_generate(avatar_features: dict) -> bytes:
    """Generate profile image from features via profile GPU."""
    generate_url = pick_profile_worker().rstrip("/") + "/v1/profile/generate"
    resp = await post_json(
        generate_url,
        {"avatar_features": avatar_features},
        timeout_s=interface_settings.profile_sla_ms / 1000,
    )
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.content.decode(errors="replace"))
    image_b64 = resp.json()["image_bytes_b64"]
    return base64.b64decode(image_b64)


def pick_text2img_worker() -> str:
    global text2img_worker_index
    urls = interface_settings.text2img_worker_urls
    if not urls:
        raise HTTPException(status_code=500, detail="No text2img workers configured")
    url = urls[text2img_worker_index % len(urls)]
    text2img_worker_index = (text2img_worker_index + 1) % len(urls)
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


async def profile_generate_handler(item: QueueItem) -> None:
    """Queue handler for profile image generation from features."""
    image_bytes = await call_profile_generate(item.payload["avatar_features"])
    item.future.set_result(image_bytes)


async def text2img_worker_handler(item: QueueItem) -> None:
    result = await call_text2img_worker(item.payload)
    item.future.set_result(result)


async def chat1to1_worker_handler(item: QueueItem) -> None:
    """Queue handler for 1:1 chat image generation."""
    worker_url = item.payload["worker_url"]
    resp = await post_json(
        str(worker_url) + "/v1/chat/1to1/generate",
        item.payload["body"],
        timeout_s=interface_settings.chat1to1_sla_ms / 1000,
    )
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.content.decode(errors="replace"))
    image_b64 = resp.json()["image_bytes_b64"]
    imagechat1to1_worker_urls": [str(u) for u in interface_settings.chat1to1_worker_urls],
        "llm_service_url": str(interface_settings.llm_service_url)
    }})
profile_worker = SingleWorker(profile_queue, profile_worker_handler)
profile_generate_worker = SingleWorker(profile_generate_queue, profile_generate_handler)
text2img_worker = SingleWorker(text2img_queue, text2img_worker_handler)
chat1to1_worker = SingleWorker(chat1to1_queue, chat1to1_worker_handler)
shorts_worker = SingleWorker(shorts_queue, shorts_worker_handler)
scenes_worker = SingleWorker(scenes_queue, scenes_worker_handler)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("interface_startup", extra={"extra_fields": {
        "selfie_feature_worker_urls": [str(u) for u in interface_settings.selfie_feature_worker_urls],
        "profile_worker_urls": [str(u) for u in interface_settings.profile_worker_urls],
        "text2img_worker_urls": [str(u) for u in interface_settings.text2img_worker_urls],
        "chat1to1_worker_urls": [str(u) for u in interface_settings.chat1to1_worker_urls],
        "shorts_worker_urls": [str(u) for u in interface_settings.shorts_worker_urls],
        "scenes_worker_urls": [str(u) for u in interface_settings.scenes_worker_urls],
        "llm_service_url": str(interface_settings.llm_service_url)
    }})
    await profile_worker.start()
    await profile_generate_worker.start()
    await text2img_worker.start()
    await chat1to1_worker.start()
    await shorts_worker.start()
    await scenes_worker.start()
    logger.info("interface_workers_started")


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
    t0 = asyncio.get_event_loop().time()
    logger.info("profile_create_start", extra={"extra_fields": {"filename": file.filename, "content_type": file.content_type}})
    
    # Read input selfie
    content = await file.read()
    logger.info("profile_create_read_file", extra={"extra_fields": {"size_bytes": len(content)}})
    
    payload = {
        "filename": file.filename,
        "content": content,
        "content_type": file.content_type or "image/png",
    }
    
    # Queue to profile worker for analysis (features only)
    logger.info("profile_create_enqueue_analyze", extra={"extra_fields": {"queue_size": profile_queue.size}})
    future = await enqueue_or_429(profile_queue, payload, interface_settings.profile_sla_ms, profile_worker)
    
    try:
        logger.info("profile_create_wait_analyze")
        result: dict = await asyncio.wait_for(future, timeout=interface_settings.profile_sla_ms / 1000)
        logger.info("profile_create_analyze_complete", extra={"extra_fields": {"has_features": "avatar_features" in result}})
    except asyncio.TimeoutError:
        logger.error(
            "profile_create_analyze_timeout",
            extra={"extra_fields": {"seconds": round(asyncio.get_event_loop().time() - t0, 2)}},
        )
        raise HTTPException(status_code=504, detail="Profile worker timed out")
    except Exception as e:
        logger.error("profile_create_analyze_error", extra={"extra_fields": {"error": str(e)}})
        raise

    # BAD_SELFIE pass-through
    if result.get("_passthrough_status") == 422:
        logger.info("profile_create_bad_selfie")
        return JSONResponse(status_code=422, content=result["_passthrough_json"])

    # Queue to profile GPU generate (features → PNG)
    gen_payload = {"avatar_features": result["avatar_features"]}
    logger.info("profile_create_enqueue_generate", extra={"extra_fields": {"queue_size": profile_generate_queue.size}})
    gen_future = await enqueue_or_429(profile_generate_queue, gen_payload, interface_settings.profile_sla_ms, profile_generate_worker)
    
    try:
        logger.info("profile_create_wait_generate")
        image_bytes: bytes = await asyncio.wait_for(gen_future, timeout=interface_settings.profile_sla_ms / 1000)
        logger.info("profile_create_generate_complete", extra={"extra_fields": {"image_size": len(image_bytes)}})
    except asyncio.TimeoutError:
        logger.error(
            "profile_create_generate_timeout",
            extra={"extra_fields": {"seconds": round(asyncio.get_event_loop().time() - t0, 2)}},
        )
        raise HTTPException(status_code=504, detail="Profile generate timed out")
    except Exception as e:
        logger.error("profile_create_generate_error", extra={"extra_fields": {"error": str(e)}})
        raise

    # Multipart response: JSON features + PNG image
    logger.info("profile_create_build_response")
    json_part = ProfileCreateResponse(avatar_features=result["avatar_features"]).model_dump_json()
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
    logger.info(
        "profile_create_success",
        extra={
            "extra_fields": {
                "response_size": len(body),
                "seconds": round(asyncio.get_event_loop().time() - t0, 2),
            }
        },
    )
    return Response(content=body, media_type=f"multipart/mixed; boundary={boundary}")

@app.get("/healthz")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "selfie_feature_worker_urls": [str(u) for u in interface_settings.selfie_feature_worker_urls],
            "profile_worker_urls": [str(u) for u in interface_settings.profile_worker_urls],
            "text2img_worker_urls": [str(u) for u in interface_settings.text2img_worker_urls],
            "chat1to1_worker_urls": [str(u) for u in interface_settings.chat1to1_worker_urls],
            "shorts_worker_urls": [str(u) for u in interface_settings.shorts_worker_urls],
            "scenes_worker_urls": [str(u) for u in interface_settings.scenes_worker_urls],
            "llm_service_url": str(interface_settings.llm_service_url),
        }
    )


@app.post("/v1/profile/update")
async def profile_update(body: ProfileUpdateRequest = Body(...)) -> Response:
    # Queue features to profile GPU generate
    gen_payload = {"avatar_features": body.avatar_features.model_dump()}
    gen_future = await enqueue_or_429(profile_generate_queue, gen_payload, interface_settings.profile_sla_ms, profile_generate_worker)
    try:
        image_bytes: bytes = await asyncio.wait_for(gen_future, timeout=interface_settings.profile_sla_ms / 1000)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Profile generate timed out")
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


@app.post("/v1/chat/1to1/imagegen")
async def chat_1to1_imagegen(body: Chat1to1ImageGenRequest = Body(...)) -> JSONResponse:
    """
    1:1 chat contextual image generation.
    
    NO NSFW restrictions - generates any content based on chat context.
    
    Flow:
    1. Call LLM to expand chat context into detailed prompt
    2. Generate image via 1:1 chat worker
    3. Generate caption via LLM
    4. Return image (base64), caption, and prompt (shortened if needed)
    """
    t0 = asyncio.get_event_loop().time()
    logger.info(
        "chat1to1_imagegen_start",
        extra={
            "extra_fields": {
                "num_messages": len(body.chat_messages),
                "num_participants": len(body.participants),
                "style": body.style,
            }
        },
    )
    
    # Step 1: Expand chat context via LLM
    logger.info("chat1to1_calling_llm_expand")
    expand_payload = {
        "chat_messages": [msg.model_dump() for msg in body.chat_messages],
        "style": body.style,
        "participants": [p.model_dump() for p in body.participants],
        "target_message": body.target_message,
    }
    expand_resp = await post_json(
        str(interface_settings.llm_service_url) + "/v1/chat/1to1/expand",
        expand_payload,
        timeout_s=15.0,
    )
    if expand_resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"LLM expand error: {expand_resp.content.decode(errors='replace')}"
        )
    
    expand_data = expand_resp.json()
    expanded_prompt = expand_data["expanded_prompt"]
    shortened_target = expand_data.get("shortened_target")
    
    logger.info(
        "chat1to1_llm_expand_complete",
        extra={"extra_fields": {"prompt_length": len(expanded_prompt), "shortened": shortened_target is not None}}
    )
    
    # Step 2: Generate image via 1:1 chat worker
    logger.info("chat1to1_enqueue_image_gen", extra={"extra_fields": {"queue_size": chat1to1_queue.size}})
    worker_url = pick_chat1to1_worker()
    gen_payload = {
        "worker_url": worker_url,
        "body": {
            "prompt": expanded_prompt,
            "negative_prompt": None,
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
        },
    }
    
    future = await enqueue_or_429(
        chat1to1_queue,
        gen_payload,
        interface_settings.chat1to1_sla_ms,
        chat1to1_worker
    )
    
    try:
        logger.info("chat1to1_wait_image_gen")
        image_bytes: bytes = await asyncio.wait_for(
            future,
            timeout=interface_settings.chat1to1_sla_ms / 1000
        )
        logger.info("chat1to1_image_gen_complete", extra={"extra_fields": {"image_size": len(image_bytes)}})
    except asyncio.TimeoutError:
        logger.error(
            "chat1to1_image_gen_timeout",
            extra={"extra_fields": {"seconds": round(asyncio.get_event_loop().time() - t0, 2)}}
        )
        raise HTTPException(status_code=504, detail="1:1 chat worker timed out")
    except Exception as e:
        logger.error("chat1to1_image_gen_error", extra={"extra_fields": {"error": str(e)}})
        raise
    
    # Step 3: Generate caption via LLM
    logger.info("chat1to1_calling_llm_caption")
    caption_payload = {"prompt": expanded_prompt}
    caption_resp = await post_json(
        str(interface_settings.llm_service_url) + "/v1/chat/1to1/caption",
        caption_payload,
        timeout_s=10.0,
    )
    if caption_resp.status_code >= 400:
        logger.warning("chat1to1_caption_failed", extra={"extra_fields": {"error": caption_resp.content.decode(errors='replace')}})
        caption = "✨ AI-generated moment"  # Fallback caption
    else:
        caption = caption_resp.json()["caption"]
    
    logger.info("chat1to1_caption_complete", extra={"extra_fields": {"caption": caption}})
    
    # Step 4: Return response
    image_base64 = base64.b64encode(image_bytes).decode()
    prompt_to_return = shortened_target if shortened_target else body.target_message
    
    t_elapsed = asyncio.get_event_loop().time() - t0
    logger.info(
        "chat1to1_imagegen_success",
        extra={
            "extra_fields": {
                "seconds": round(t_elapsed, 2),
                "response_size": len(image_base64),
            }
        },
    )
    
    return JSONResponse(
        content=Chat1to1ImageGenResponse(
            image_base64=image_base64,
            caption=caption,
            prompt_used=prompt_to_return,
        ).model_dump()
    )


@app.post("/v1/chat/shorts/generate")
async def chat_shorts_generate(body: ShortsImageGenRequest = Body(...)) -> JSONResponse:
    """
    Shorts image generation with NSFW moderation.
    
    No chat history - just user message, style, and participants.
    NSFW moderated: Avoids straight nudity via clothing/occlusions.
    
    Flow:
    1. Call LLM to expand prompt with NSFW moderation
    2. Generate image via shorts worker
    3. Generate caption via LLM
    4. Return image (base64), caption, and prompt (shortened if needed)
    """
    t0 = asyncio.get_event_loop().time()
    logger.info(
        "shorts_generate_start",
        extra={
            "extra_fields": {
                "num_participants": len(body.participants),
                "style": body.style,
            }
        },
    )
    
    # Step 1: Expand prompt via LLM with NSFW moderation
    logger.info("shorts_calling_llm_expand")
    expand_payload = {
        "style": body.style,
        "participants": [p.model_dump() for p in body.participants],
        "user_message": body.user_message,
    }
    expand_resp = await post_json(
        str(interface_settings.llm_service_url) + "/v1/chat/shorts/expand",
        expand_payload,
        timeout_s=15.0,
    )
    if expand_resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"LLM expand error: {expand_resp.content.decode(errors='replace')}"
        )
    
    expand_data = expand_resp.json()
    expanded_prompt = expand_data["expanded_prompt"]
    shortened_message = expand_data.get("shortened_message")
    
    logger.info(
        "shorts_llm_expand_complete",
        extra={"extra_fields": {"prompt_length": len(expanded_prompt), "shortened": shortened_message is not None}}
    )
    
    # Step 2: Generate image via shorts worker
    logger.info("shorts_enqueue_image_gen", extra={"extra_fields": {"queue_size": shorts_queue.size}})
    worker_url = pick_shorts_worker()
    gen_payload = {
        "worker_url": worker_url,
        "body": {
            "prompt": expanded_prompt,
            "negative_prompt": "nudity, explicit, nsfw, naked, nude",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
        },
    }
    
    future = await enqueue_or_429(
        shorts_queue,
        gen_payload,
        interface_settings.shorts_sla_ms,
        shorts_worker
    )
    
    try:
        logger.info("shorts_wait_image_gen")
        image_bytes: bytes = await asyncio.wait_for(
            future,
            timeout=interface_settings.shorts_sla_ms / 1000
        )
        logger.info("shorts_image_gen_complete", extra={"extra_fields": {"image_size": len(image_bytes)}})
    except asyncio.TimeoutError:
        logger.error(
            "shorts_image_gen_timeout",
            extra={"extra_fields": {"seconds": round(asyncio.get_event_loop().time() - t0, 2)}}
        )
        raise HTTPException(status_code=504, detail="Shorts worker timed out")
    except Exception as e:
        logger.error("shorts_image_gen_error", extra={"extra_fields": {"error": str(e)}})
        raise
    
    # Step 3: Generate caption via LLM
    logger.info("shorts_calling_llm_caption")
    caption_payload = {"prompt": expanded_prompt}
    caption_resp = await post_json(
        str(interface_settings.llm_service_url) + "/v1/chat/1to1/caption",
        caption_payload,
        timeout_s=10.0,
    )
    if caption_resp.status_code >= 400:
        logger.warning("shorts_caption_failed", extra={"extra_fields": {"error": caption_resp.content.decode(errors='replace')}})
        caption = "✨ AI-generated shorts"  # Fallback caption
    else:
        caption = caption_resp.json()["caption"]
    
    logger.info("shorts_caption_complete", extra={"extra_fields": {"caption": caption}})
    
    # Step 4: Return response
    image_base64 = base64.b64encode(image_bytes).decode()
    prompt_to_return = shortened_message if shortened_message else body.user_message
    
    t_elapsed = asyncio.get_event_loop().time() - t0
    logger.info(
        "shorts_generate_success",
        extra={
            "extra_fields": {
                "seconds": round(t_elapsed, 2),
                "response_size": len(image_base64),
            }
        },
    )
    
    return JSONResponse(
        content=ShortsImageGenResponse(
            image_base64=image_base64,
            caption=caption,
            prompt_used=prompt_to_return,
        ).model_dump()
    )


@app.post("/v1/chat/scenes/generate")
async def chat_scenes_generate(body: ScenesImageGenRequest = Body(...)) -> JSONResponse:
    """
    Scenes image generation with NSFW moderation.
    
    No chat history - just user message, style, and participants.
    NSFW moderated: Avoids straight nudity via clothing/occlusions.
    
    Flow:
    1. Call LLM to expand prompt with NSFW moderation
    2. Generate image via scenes worker
    3. Generate caption via LLM
    4. Return image (base64), caption, and prompt (shortened if needed)
    """
    t0 = asyncio.get_event_loop().time()
    logger.info(
        "scenes_generate_start",
        extra={
            "extra_fields": {
                "num_participants": len(body.participants),
                "style": body.style,
            }
        },
    )
    
    # Step 1: Expand prompt via LLM with NSFW moderation
    logger.info("scenes_calling_llm_expand")
    expand_payload = {
        "style": body.style,
        "participants": [p.model_dump() for p in body.participants],
        "user_message": body.user_message,
    }
    expand_resp = await post_json(
        str(interface_settings.llm_service_url) + "/v1/chat/scenes/expand",
        expand_payload,
        timeout_s=15.0,
    )
    if expand_resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"LLM expand error: {expand_resp.content.decode(errors='replace')}"
        )
    
    expand_data = expand_resp.json()
    expanded_prompt = expand_data["expanded_prompt"]
    shortened_message = expand_data.get("shortened_message")
    
    logger.info(
        "scenes_llm_expand_complete",
        extra={"extra_fields": {"prompt_length": len(expanded_prompt), "shortened": shortened_message is not None}}
    )
    
    # Step 2: Generate image via scenes worker
    logger.info("scenes_enqueue_image_gen", extra={"extra_fields": {"queue_size": scenes_queue.size}})
    worker_url = pick_scenes_worker()
    gen_payload = {
        "worker_url": worker_url,
        "body": {
            "prompt": expanded_prompt,
            "negative_prompt": "nudity, explicit, nsfw, naked, nude",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
        },
    }
    
    future = await enqueue_or_429(
        scenes_queue,
        gen_payload,
        interface_settings.scenes_sla_ms,
        scenes_worker
    )
    
    try:
        logger.info("scenes_wait_image_gen")
        image_bytes: bytes = await asyncio.wait_for(
            future,
            timeout=interface_settings.scenes_sla_ms / 1000
        )
        logger.info("scenes_image_gen_complete", extra={"extra_fields": {"image_size": len(image_bytes)}})
    except asyncio.TimeoutError:
        logger.error(
            "scenes_image_gen_timeout",
            extra={"extra_fields": {"seconds": round(asyncio.get_event_loop().time() - t0, 2)}}
        )
        raise HTTPException(status_code=504, detail="Scenes worker timed out")
    except Exception as e:
        logger.error("scenes_image_gen_error", extra={"extra_fields": {"error": str(e)}})
        raise
    
    # Step 3: Generate caption via LLM
    logger.info("scenes_calling_llm_caption")
    caption_payload = {"prompt": expanded_prompt}
    caption_resp = await post_json(
        str(interface_settings.llm_service_url) + "/v1/chat/1to1/caption",
        caption_payload,
        timeout_s=10.0,
    )
    if caption_resp.status_code >= 400:
        logger.warning("scenes_caption_failed", extra={"extra_fields": {"error": caption_resp.content.decode(errors='replace')}})
        caption = "✨ AI-generated scene"  # Fallback caption
    else:
        caption = caption_resp.json()["caption"]
    
    logger.info("scenes_caption_complete", extra={"extra_fields": {"caption": caption}})
    
    # Step 4: Return response
    image_base64 = base64.b64encode(image_bytes).decode()
    prompt_to_return = shortened_message if shortened_message else body.user_message
    
    t_elapsed = asyncio.get_event_loop().time() - t0
    logger.info(
        "scenes_generate_success",
        extra={
            "extra_fields": {
                "seconds": round(t_elapsed, 2),
                "response_size": len(image_base64),
            }
        },
    )
    
    return JSONResponse(
        content=ScenesImageGenResponse(
            image_base64=image_base64,
            caption=caption,
            prompt_used=prompt_to_return,
        ).model_dump()
    )


