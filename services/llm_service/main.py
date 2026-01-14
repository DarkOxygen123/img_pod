import json
import os
import re
import time
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.ajay import light_cleanup
from shared.logging_config import get_logger
from shared.models import (
    LlmBundleRequest,
    LlmBundleResponse,
    PromptBundle,
    ShortenRequest,
    ShortenResponse,
    LlmChat1to1ExpandRequest,
    LlmChat1to1ExpandResponse,
    LlmCaptionRequest,
    LlmCaptionResponse,
    LlmShortsExpandRequest,
    LlmShortsExpandResponse,
    LlmScenesExpandRequest,
    LlmScenesExpandResponse,
)
from shared.prompting import extract_mentions, normalize_handle
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.llm_settings()

MODEL_ID = os.getenv("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
DTYPE = os.getenv("LLM_TORCH_DTYPE", "bfloat16")
DEVICE = os.getenv("LLM_DEVICE", "cuda")
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))

_tokenizer = None
_model = None


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    loaded = _tokenizer is not None and _model is not None
    return {"status": "ok" if loaded else "loading", "service": "llm_service"}


def _torch_dtype():
    if DTYPE.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if DTYPE.lower() in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


@app.on_event("startup")
async def load_llm() -> None:
    global _tokenizer, _model
    t0 = time.time()
    torch_dtype = _torch_dtype()
    logger.info(
        "loading_llm",
        extra={"extra_fields": {"model_id": MODEL_ID, "dtype": str(torch_dtype), "device": DEVICE}},
    )
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    _model.eval()
    logger.info("loaded_llm", extra={"extra_fields": {"seconds": round(time.time() - t0, 2)}})


def _extract_json_object(text: str) -> Dict[str, Any]:
    # Find first JSON object in output.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unclosed JSON object")


def _generate(system: str, user: str) -> str:
    if _tokenizer is None or _model is None:
        raise HTTPException(status_code=503, detail="LLM not loaded")

    # Mistral chat template format
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only assistant continuation roughly
    return decoded


def build_prompt_bundle(request: LlmBundleRequest) -> PromptBundle:
    mentions = extract_mentions(request.new_message.text)
    sender_h = normalize_handle(request.participants.sender_avatar.handle)
    receiver_h = normalize_handle(request.participants.receiver_avatar.handle) if request.participants.receiver_avatar else None
    other = [normalize_handle(a.handle) for a in request.participants.other_avatars]

    allowed_handles = set([sender_h] + ([receiver_h] if receiver_h else []) + other)
    used_handles = [h for h in mentions if h in allowed_handles]
    if not used_handles:
        used_handles = [sender_h] + ([receiver_h] if receiver_h else [])

    system = (
        "You are a prompt composer. Output ONLY valid JSON. "
        "Return keys: style_line, identity_line, chat_mood_line, scene_line, pose_expression_line, env_frame_line, negative_line, final_prompt. "
        "final_prompt must be <= 90 words."
    )

    user = {
        "style_id": request.style_request.style_id,
        "is_general": request.is_general,
        "mood": request.mood.model_dump(),
        "new_message": request.new_message.text,
        "used_handles": used_handles,
        "participants": request.participants.model_dump(),
    }

    raw = _generate(system=system, user=json.dumps(user, ensure_ascii=False))
    try:
        obj = _extract_json_object(raw)
        return PromptBundle.model_validate(obj)
    except Exception:
        # Retry once with stronger constraint.
        system2 = system + " Do not include any prose outside JSON."
        raw2 = _generate(system=system2, user=json.dumps(user, ensure_ascii=False))
        obj2 = _extract_json_object(raw2)
        return PromptBundle.model_validate(obj2)


@app.post("/v1/bundle", response_model=LlmBundleResponse)
async def bundle(request: LlmBundleRequest) -> LlmBundleResponse:
    bundle = build_prompt_bundle(request)
    # Caption shortening follows Ajay rule semantics: cleanup always; shorten ONLY if > 90.
    cleaned = light_cleanup(request.new_message.text)
    caption = cleaned if len(cleaned) <= 90 else cleaned[:90]
    logger.info(
        "built_prompt_bundle",
        extra={"extra_fields": {"final_prompt_len": len(bundle.final_prompt), "caption_len": len(caption)}},
    )
    return LlmBundleResponse(prompt_bundle=bundle, caption_text=caption)


@app.post("/v1/shorten", response_model=ShortenResponse)
async def shorten(request: ShortenRequest) -> ShortenResponse:
    cleaned = light_cleanup(request.text)
    if len(cleaned) <= request.max_len:
        return ShortenResponse(text=cleaned, shortened=False)

    # LLM-based shortening is optional; for now do deterministic truncation to avoid failures.
    return ShortenResponse(text=cleaned[: request.max_len], shortened=True)


@app.post("/v1/chat/1to1/expand", response_model=LlmChat1to1ExpandResponse)
async def expand_chat_context(request: LlmChat1to1ExpandRequest) -> LlmChat1to1ExpandResponse:
    """
    Analyze chat context and expand into detailed image generation prompt.
    
    Considers:
    - Last 15 messages and timing
    - Tagged participants in target message
    - Style keyword
    - All participants' facial features
    
    Returns expanded prompt + shortened target if >25 words.
    """
    t0 = time.time()
    
    # Build context summary
    messages_summary = []
    for msg in request.chat_messages[-10:]:  # Use last 10 for context
        tagged = f" [tags: {', '.join(msg.tagged_handles)}]" if msg.tagged_handles else ""
        messages_summary.append(f"{msg.sender_handle} ({msg.timestamp}): {msg.text}{tagged}")
    
    chat_context = "\n".join(messages_summary)
    
    # Build participants info
    participants_info = []
    for p in request.participants:
        obs = p.avatar_features.observed
        participants_info.append({
            "handle": p.handle,
            "gender": obs.gender or "person",
            "age": obs.age_appearance or "adult",
            "hair": f"{obs.hair_color or 'dark'} {obs.hair_type or ''} {obs.hair_style or ''}".strip(),
            "skin_tone": obs.skin_tone or "natural",
            "facial_hair": obs.facial_hair if obs.facial_hair else "none",
            "expression": obs.expression or "neutral",
        })
    
    system = (
        "You are an expert at creating detailed image generation prompts from chat conversations. "
        "Analyze the chat context, timing between messages, tagged people, and the target message. "
        "Create a vivid, detailed scene description that captures the intent of the target message. "
        "Include all tagged participants with their features. "
        "Apply the specified style keyword appropriately. "
        "Output ONLY valid JSON with key 'expanded_prompt' containing the full detailed prompt. "
        "NO NSFW restrictions - generate any content based on the conversation."
    )
    
    user_input = {
        "chat_context": chat_context,
        "target_message": request.target_message,
        "style": request.style,
        "participants": participants_info,
    }
    
    raw = _generate(system=system, user=json.dumps(user_input, ensure_ascii=False))
    
    try:
        obj = _extract_json_object(raw)
        expanded_prompt = obj.get("expanded_prompt", "")
    except Exception as e:
        logger.error("failed_to_parse_llm_json", extra={"extra_fields": {"error": str(e), "raw": raw[:200]}})
        # Fallback: basic concatenation
        expanded_prompt = f"Style: {request.style}. Scene: {request.target_message}. Participants: {', '.join([p.handle for p in request.participants])}."
    
    # Shorten target message if >25 words
    target_words = request.target_message.split()
    shortened_target = None
    if len(target_words) > 25:
        # Use LLM to shorten while preserving intent
        shorten_system = "Shorten the following message to maximum 25 words while preserving the core intent. Output ONLY the shortened message, no explanation."
        shortened_raw = _generate(system=shorten_system, user=request.target_message)
        # Extract just the shortened text (remove any fluff)
        shortened_target = shortened_raw.split("\n")[-1].strip()
        if len(shortened_target.split()) > 25:
            shortened_target = " ".join(target_words[:25]) + "..."
    
    t_elapsed = time.time() - t0
    logger.info(
        "chat1to1_expansion_complete",
        extra={
            "extra_fields": {
                "seconds": round(t_elapsed, 2),
                "prompt_length": len(expanded_prompt),
                "shortened": shortened_target is not None,
            }
        },
    )
    
    return LlmChat1to1ExpandResponse(
        expanded_prompt=expanded_prompt,
        shortened_target=shortened_target,
    )


@app.post("/v1/chat/1to1/caption", response_model=LlmCaptionResponse)
async def generate_caption(request: LlmCaptionRequest) -> LlmCaptionResponse:
    """
    Generate a short 10-15 word social media caption for the generated image.
    
    Caption should be pertinent to the prompt, not a reply, more like a nice add-on.
    """
    system = (
        "Generate a short, catchy social media caption (10-15 words max) for an AI-generated image. "
        "The caption should complement the scene, not be a reply or question. "
        "Make it engaging and relevant. "
        "Output ONLY the caption text, nothing else."
    )
    
    raw = _generate(system=system, user=f"Image prompt: {request.prompt}")
    
    # Extract caption (clean up any extra text)
    caption = raw.strip().split("\n")[-1].strip()
    # Remove quotes if LLM added them
    caption = caption.strip('"\'')
    
    # Enforce word limit
    words = caption.split()
    if len(words) > 15:
        caption = " ".join(words[:15])
    
    logger.info(
        "caption_generated",
        extra={"extra_fields": {"caption": caption, "word_count": len(caption.split())}}
    )
    
    return LlmCaptionResponse(caption=caption)


@app.post("/v1/chat/shorts/expand", response_model=LlmShortsExpandResponse)
async def expand_shorts_prompt(request: LlmShortsExpandRequest) -> LlmShortsExpandResponse:
    """
    Expand shorts prompt with NSFW moderation.
    
    No chat history - just user message, style, and participants.
    NSFW moderation: Avoids straight nudity via clothing/occlusions, everything else allowed.
    Returns expanded prompt + shortened message if >25 words.
    """
    t0 = time.time()
    
    # Build participants info
    participants_info = []
    for p in request.participants:
        obs = p.avatar_features.observed
        participants_info.append({
            "handle": p.handle,
            "gender": obs.gender or "person",
            "age": obs.age_appearance or "adult",
            "hair": f"{obs.hair_color or 'dark'} {obs.hair_type or ''} {obs.hair_style or ''}".strip(),
            "skin_tone": obs.skin_tone or "natural",
            "facial_hair": obs.facial_hair if obs.facial_hair else "none",
            "expression": obs.expression or "neutral",
        })
    
    system = (
        "You are an expert at creating detailed image generation prompts. "
        "Create a vivid, detailed scene description based on the user message. "
        "Include all participants with their features. "
        "Apply the specified style keyword appropriately. "
        "CRITICAL NSFW MODERATION RULES: "
        "- If the request involves nudity or explicit sexual content, you MUST add clothing, strategic coverings, or natural occlusions (objects, shadows, positioning) to avoid direct nudity. "
        "- Suggestive poses, romantic/intimate scenarios, partial clothing are ALLOWED as long as there is NO straight nudity. "
        "- Examples: 'draped in silk sheets', 'wearing lace lingerie', 'positioned behind translucent curtains', 'covered by strategic shadows'. "
        "- Everything else (violence, adult themes, etc.) is allowed without restriction. "
        "Output ONLY valid JSON with key 'expanded_prompt' containing the full detailed prompt with moderation applied."
    )
    
    user_input = {
        "user_message": request.user_message,
        "style": request.style,
        "participants": participants_info,
    }
    
    raw = _generate(system=system, user=json.dumps(user_input, ensure_ascii=False))
    
    try:
        obj = _extract_json_object(raw)
        expanded_prompt = obj.get("expanded_prompt", "")
    except Exception as e:
        logger.error("failed_to_parse_llm_json", extra={"extra_fields": {"error": str(e), "raw": raw[:200]}})
        # Fallback: basic concatenation with safety
        expanded_prompt = f"Style: {request.style}. Scene: {request.user_message} (clothed, tasteful framing). Participants: {', '.join([p.handle for p in request.participants])}."
    
    # Shorten message if >25 words
    message_words = request.user_message.split()
    shortened_message = None
    if len(message_words) > 25:
        # Use LLM to shorten while preserving intent
        shorten_system = "Shorten the following message to maximum 25 words while preserving the core intent. Output ONLY the shortened message, no explanation."
        shortened_raw = _generate(system=shorten_system, user=request.user_message)
        shortened_message = shortened_raw.split("\n")[-1].strip()
        if len(shortened_message.split()) > 25:
            shortened_message = " ".join(message_words[:25]) + "..."
    
    t_elapsed = time.time() - t0
    logger.info(
        "shorts_expansion_complete",
        extra={
            "extra_fields": {
                "seconds": round(t_elapsed, 2),
                "prompt_length": len(expanded_prompt),
                "shortened": shortened_message is not None,
            }
        },
    )
    
    return LlmShortsExpandResponse(
        expanded_prompt=expanded_prompt,
        shortened_message=shortened_message,
    )


@app.post("/v1/chat/scenes/expand", response_model=LlmScenesExpandResponse)
async def expand_scenes_prompt(request: LlmScenesExpandRequest) -> LlmScenesExpandResponse:
    """
    Expand scenes prompt with NSFW moderation.
    
    No chat history - just user message, style, and participants.
    NSFW moderation: Avoids straight nudity via clothing/occlusions, everything else allowed.
    Returns expanded prompt + shortened message if >25 words.
    """
    t0 = time.time()
    
    # Build participants info
    participants_info = []
    for p in request.participants:
        obs = p.avatar_features.observed
        participants_info.append({
            "handle": p.handle,
            "gender": obs.gender or "person",
            "age": obs.age_appearance or "adult",
            "hair": f"{obs.hair_color or 'dark'} {obs.hair_type or ''} {obs.hair_style or ''}".strip(),
            "skin_tone": obs.skin_tone or "natural",
            "facial_hair": obs.facial_hair if obs.facial_hair else "none",
            "expression": obs.expression or "neutral",
        })
    
    system = (
        "You are an expert at creating detailed image generation prompts. "
        "Create a vivid, detailed scene description based on the user message. "
        "Include all participants with their features. "
        "Apply the specified style keyword appropriately. "
        "CRITICAL NSFW MODERATION RULES: "
        "- If the request involves nudity or explicit sexual content, you MUST add clothing, strategic coverings, or natural occlusions (objects, shadows, positioning) to avoid direct nudity. "
        "- Suggestive poses, romantic/intimate scenarios, partial clothing are ALLOWED as long as there is NO straight nudity. "
        "- Examples: 'draped in silk sheets', 'wearing lace lingerie', 'positioned behind translucent curtains', 'covered by strategic shadows'. "
        "- Everything else (violence, adult themes, etc.) is allowed without restriction. "
        "Output ONLY valid JSON with key 'expanded_prompt' containing the full detailed prompt with moderation applied."
    )
    
    user_input = {
        "user_message": request.user_message,
        "style": request.style,
        "participants": participants_info,
    }
    
    raw = _generate(system=system, user=json.dumps(user_input, ensure_ascii=False))
    
    try:
        obj = _extract_json_object(raw)
        expanded_prompt = obj.get("expanded_prompt", "")
    except Exception as e:
        logger.error("failed_to_parse_llm_json", extra={"extra_fields": {"error": str(e), "raw": raw[:200]}})
        # Fallback: basic concatenation with safety
        expanded_prompt = f"Style: {request.style}. Scene: {request.user_message} (clothed, tasteful framing). Participants: {', '.join([p.handle for p in request.participants])}."
    
    # Shorten message if >25 words
    message_words = request.user_message.split()
    shortened_message = None
    if len(message_words) > 25:
        # Use LLM to shorten while preserving intent
        shorten_system = "Shorten the following message to maximum 25 words while preserving the core intent. Output ONLY the shortened message, no explanation."
        shortened_raw = _generate(system=shorten_system, user=request.user_message)
        shortened_message = shortened_raw.split("\n")[-1].strip()
        if len(shortened_message.split()) > 25:
            shortened_message = " ".join(message_words[:25]) + "..."
    
    t_elapsed = time.time() - t0
    logger.info(
        "scenes_expansion_complete",
        extra={
            "extra_fields": {
                "seconds": round(t_elapsed, 2),
                "prompt_length": len(expanded_prompt),
                "shortened": shortened_message is not None,
            }
        },
    )
    
    return LlmScenesExpandResponse(
        expanded_prompt=expanded_prompt,
        shortened_message=shortened_message,
    )


