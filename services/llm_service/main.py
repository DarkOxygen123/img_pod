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


def _build_character_description(handle: str, avatar_features) -> str:
    """Build comprehensive character description matching profile generation format."""
    obs = avatar_features.observed
    dress = avatar_features.dress
    accessories = avatar_features.accessories
    
    # Normalize gender
    gender_raw = (obs.gender or "person").strip()
    if gender_raw == "male":
        gender = "man"
    elif gender_raw == "female":
        gender = "woman"
    else:
        gender = "person"
    
    # Core features (matching profile_worker._profile_prompt)
    age = obs.age_appearance or "adult"
    skin_tone = obs.skin_tone or "natural"
    skin_undertone = obs.skin_undertone
    hair_color = obs.hair_color or "dark"
    hair_type = obs.hair_type or "natural"
    hair_style = obs.hair_style
    hair_length = obs.hair_length or "medium"
    hairline_type = obs.hairline_type
    balding_pattern = obs.balding_pattern
    eye_color = obs.eye_color or "brown"
    eye_shape = obs.eye_shape
    face_shape = obs.face_shape or "oval"
    
    # Facial hair (comprehensive)
    facial_hair = obs.facial_hair
    facial_hair_density = obs.facial_hair_density
    beard_style = obs.beard_style
    mustache_style = obs.mustache_style
    
    # Facial marks
    facial_marks = obs.facial_marks
    facial_mark_position = obs.facial_mark_position
    
    # Clothing
    dress_color = dress.dress_color if dress else None
    dress_type = dress.dress_type if dress else None
    
    # Accessories
    hat_present = accessories.hat_present == "yes" if accessories else False
    hat_style = accessories.hat_style if accessories else None
    hat_color = accessories.hat_color if accessories else None
    glasses_present = accessories.glasses_present == "yes" if accessories else False
    glasses_type = accessories.glasses_type if accessories else None
    glasses_color = accessories.glasses_color if accessories else None
    mask_present = accessories.mask_present == "yes" if accessories else False
    mask_type = accessories.mask_type if accessories else None
    mask_color = accessories.mask_color if accessories else None
    
    # Build comprehensive description
    parts = [f"{handle}: {age} {gender}"]
    
    # Skin
    if skin_undertone and skin_undertone != "none":
        parts.append(f"{skin_tone} skin ({skin_undertone} undertone)")
    else:
        parts.append(f"{skin_tone} skin")
    
    # Hair (detailed)
    hair_desc = f"{hair_color} {hair_type} hair"
    if hair_length:
        hair_desc += f" ({hair_length} length)"
    if hair_style and hair_style != "none":
        hair_desc += f", {hair_style} style"
    if hairline_type and hairline_type != "none":
        hair_desc += f", {hairline_type} hairline"
    if balding_pattern and balding_pattern not in ("none", "no"):
        hair_desc += f", {balding_pattern} balding"
    parts.append(hair_desc)
    
    # Eyes
    if eye_shape and eye_shape != "none":
        parts.append(f"{eye_color} eyes ({eye_shape} shape)")
    else:
        parts.append(f"{eye_color} eyes")
    
    # Face shape
    parts.append(f"{face_shape} face")
    
    # Facial hair (detailed, gender-aware)
    if gender == "woman":
        # Women: explicitly no facial hair
        facial_hair_desc = "no facial hair"
    else:
        # Men: comprehensive facial hair handling
        beard = (beard_style or "").strip()
        mustache = (mustache_style or "").strip()
        
        if beard and beard != "none" and mustache and mustache != "none":
            if facial_hair_density and facial_hair_density != "none":
                facial_hair_desc = f"{facial_hair_density} {mustache} mustache and {beard} beard"
            else:
                facial_hair_desc = f"{mustache} mustache and {beard} beard"
        elif beard and beard != "none":
            if facial_hair_density and facial_hair_density != "none":
                facial_hair_desc = f"{facial_hair_density} {beard} beard"
            else:
                facial_hair_desc = f"{beard} beard"
        elif mustache and mustache != "none":
            if facial_hair_density and facial_hair_density != "none":
                facial_hair_desc = f"{facial_hair_density} {mustache} mustache"
            else:
                facial_hair_desc = f"{mustache} mustache"
        elif facial_hair and facial_hair != "none":
            # Fallback to legacy field
            if facial_hair_density and facial_hair_density != "none":
                facial_hair_desc = f"{facial_hair_density} {facial_hair}"
            else:
                facial_hair_desc = facial_hair
        else:
            # Explicitly clean-shaven when no facial hair detected
            facial_hair_desc = "clean-shaven"
    
    parts.append(facial_hair_desc)
    
    # Facial marks
    if facial_marks and facial_marks != "none":
        if facial_mark_position and facial_mark_position != "none":
            parts.append(f"{facial_marks} on {facial_mark_position}")
        else:
            parts.append(f"{facial_marks}")
    
    # Clothing
    if dress_color and dress_type:
        parts.append(f"wearing {dress_color} {dress_type}")
    
    # Accessories
    if hat_present and hat_style and hat_style != "none":
        if hat_color and hat_color != "none":
            parts.append(f"{hat_color} {hat_style} hat")
        else:
            parts.append(f"{hat_style} hat")
    
    if glasses_present and glasses_type and glasses_type != "none":
        if glasses_color and glasses_color != "none":
            parts.append(f"{glasses_color} {glasses_type}")
        else:
            parts.append(f"{glasses_type}")
    
    if mask_present and mask_type and mask_type != "none":
        if mask_color and mask_color != "none":
            parts.append(f"{mask_color} {mask_type} mask")
        else:
            parts.append(f"{mask_type} mask")
    
    return "; ".join(parts)


def _generate(system: str, user: str) -> str:
    if _tokenizer is None or _model is None:
        raise HTTPException(status_code=503, detail="LLM not loaded")

    # Mistral chat template format
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    logger.info("llm_prompt_template", extra={"extra_fields": {
        "prompt_length": len(prompt),
        "prompt_preview": prompt[-300:]  # Last 300 chars to see the generation prompt marker
    }})
    
    inputs = _tokenizer(prompt, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    
    # Decode only the generated tokens (exclude the input prompt)
    generated_ids = out[0][input_length:]
    response = _tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    logger.info("llm_generation_debug", extra={"extra_fields": {
        "input_length": input_length,
        "output_length": out[0].shape[0],
        "generated_tokens": generated_ids.shape[0],
        "response_length": len(response),
        "response_preview": response[:500]
    }})
    
    return response


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
    
    # Build comprehensive participants descriptions (matching profile format)
    participants_info = []
    for p in request.participants:
        char_desc = _build_character_description(p.handle, p.avatar_features)
        participants_info.append(char_desc)
    
    system = (
        "You are an expert at creating detailed image generation prompts from chat conversations. "
        "Analyze the chat context and create a prompt following this EXACT structure:\n\n"
        "For 3d_cartoon style, use this template:\n"
        "Style & rendering: High-quality stylized 3D cartoon render (NOT photorealistic). Toon shading with smooth clean materials. "
        "Simplified geometry but HIGH DETAIL in faces - proper eyes with pupils, defined features, expressive faces. "
        "Animated character/avatar look similar to Pixar/Disney quality 3D animation. Clean topology, soft subsurface skin scattering. "
        "Non-photoreal CG characters. NOT clay models, NOT low-poly, NOT oversimplified.\n"
        "Scene description: [Describe the scene action/activity based on target message]\n"
        "Character details: [Use the complete participant descriptions provided - they contain all physical features] "
        "Clothing & attire: [Specify appropriate clothing based on scene context - casual/formal/athletic/etc. Match the setting and activity]. "
        "Facial expression: [HIGH EMPHASIS - derive from chat mood and target message]. Pose & body language: [Based on scene context].\n"
        "Environment & background: [Setting details from target message]\n"
        "Lighting & color: [Describe lighting mood matching the scene]\n"
        "Composition & framing: [Camera angle and framing. CRITICAL: Characters should fill 60-70% of frame height - proper adult proportions, "
        "NOT tiny/child-sized figures. Medium shot or closer framing. Characters are the main focus, clearly visible with defined features].\n\n"
        "For anime style, use this template:\n"
        "Style & rendering: High-quality anime/manga art style with cel-shaded coloring. Clean linework, vibrant saturated colors. "
        "Expressive large eyes with detailed highlights and reflections. Stylized proportions typical of anime character design. "
        "Smooth gradients and sharp shadows (cel shading). NOT sketch/rough, NOT chibi/super-deformed unless specified.\n"
        "Scene description: [Describe the scene action/activity based on target message]\n"
        "Character details: [Use the complete participant descriptions - adapt features to anime aesthetics while preserving key characteristics] "
        "Clothing & attire: [Specify appropriate anime-style clothing based on scene context]. "
        "Facial expression: [HIGH EMPHASIS - anime-style expressive emotions]. Pose & body language: [Dynamic anime posing].\n"
        "Environment & background: [Setting with anime art style elements]\n"
        "Lighting & color: [Vibrant anime color palette and dramatic lighting]\n"
        "Composition & framing: [CRITICAL: Characters fill 60-70% of frame - proper proportions, clearly visible detailed features].\n\n"
        "For realistic style, use this template:\n"
        "Style & rendering: Photorealistic quality with natural skin textures, lifelike details. Proper subsurface scattering, pore detail, "
        "realistic hair strands. Natural proportions and anatomy. Professional photography quality lighting and depth of field. "
        "NOT painted/illustrated, NOT CG/rendered look - actual photographic realism.\n"
        "Scene description: [Describe the scene action/activity based on target message]\n"
        "Character details: [Use the complete participant descriptions - translate to realistic photographic appearance] "
        "Clothing & attire: [Specify realistic clothing with fabric textures and natural wear]. "
        "Facial expression: [Subtle realistic human expressions]. Pose & body language: [Natural human posing].\n"
        "Environment & background: [Realistic setting with environmental details]\n"
        "Lighting & color: [Natural or professional photography lighting setup]\n"
        "Composition & framing: [CRITICAL: Characters fill 60-70% of frame - photographic composition, sharp focus on subjects].\n\n"
        "CRITICAL: Use the style-specific template above. Include ALL participants mentioned or tagged with their complete descriptions. "
        "Characters must be properly sized - NOT miniature/child-like scale. "
        "Output ONLY valid JSON with key 'expanded_prompt' containing the complete structured prompt. "
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
        # LLM returns nested object, flatten it into a single string
        if isinstance(obj.get("expanded_prompt"), dict):
            # Flatten the structured sections into a single prompt string
            sections = obj["expanded_prompt"]
            expanded_prompt = "\n\n".join([f"{k}: {v}" for k, v in sections.items()])
        else:
            expanded_prompt = obj.get("expanded_prompt", "")
    except Exception as e:
        logger.error("failed_to_parse_llm_json", extra={"extra_fields": {"error": str(e), "raw": raw[:200]}})
        # Fallback: basic concatenation using handle names from participants_info strings
        participant_names = ", ".join([p.handle for p in request.participants])
        expanded_prompt = f"Style: {request.style}. Scene: {request.target_message}. Participants: {participant_names}."
    
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
                "FULL_EXPANDED_PROMPT": expanded_prompt,  # Log complete prompt for debugging
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

    Caption style varies by context:
    - 1to1: Intimate, 1st/2nd person ("us", "we", "our") - NOT 3rd person
    - shorts/scenes: Emotion-focused, captures feeling not description
    """
    if request.context_type == "1to1":
        system = (
            "Generate a short, intimate caption (10-15 words max) for a 1:1 chat image. "
            "Use 1st or 2nd person perspective (us, we, our, you, me). "
            "NEVER use 3rd person (they, Alice and Bob, etc). "
            "Express the shared moment/emotion between the two people. "
            "Output ONLY the caption text, nothing else."
        )
    else:  # shorts or scenes
        system = (
            "Generate a short, emotion-focused caption (10-15 words max) for a social media image. "
            "Capture the FEELING or MOOD, not just describe what's shown. "
            "Make it evocative and engaging. "
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
    
    # Build comprehensive participants descriptions (matching profile format)
    participants_info = []
    for p in request.participants:
        char_desc = _build_character_description(p.handle, p.avatar_features)
        participants_info.append(char_desc)
    
    system = (
        "You are an expert at creating detailed image generation prompts. "
        "Create a prompt following this EXACT structure:\n\n"
        "For 3d_cartoon style, use this template:\n"
        "Style & rendering: High-quality stylized 3D cartoon render (NOT photorealistic). Toon shading with smooth clean materials. "
        "Simplified geometry but HIGH DETAIL in faces - proper eyes with pupils, defined features, expressive faces. "
        "Animated character/avatar look similar to Pixar/Disney quality 3D animation. Clean topology, soft subsurface skin scattering. "
        "Non-photoreal CG characters. NOT clay models, NOT low-poly, NOT oversimplified.\n"
        "Scene description: [Describe the scene action/activity based on user message]\n"
        "Character details: [Use the complete participant descriptions provided - they contain all physical features] "
        "Clothing & attire: [Specify appropriate clothing based on scene context - casual/formal/athletic/streetwear/etc. Match the setting and activity]. "
        "Facial expression: [HIGH EMPHASIS - derive from user message mood]. Pose & body language: [Based on scene context].\n"
        "Environment & background: [Setting details from user message]\n"
        "Lighting & color: [Describe lighting mood matching the scene]\n"
        "Composition & framing: [Camera angle and framing. CRITICAL: Character should fill 60-70% of frame height - proper adult proportions, "
        "NOT tiny/child-sized figure. Medium shot or closer framing. Character is the main focus, clearly visible with defined features].\n\n"
        "For anime style, use this template:\n"
        "Style & rendering: High-quality anime/manga art style with cel-shaded coloring. Clean linework, vibrant saturated colors. "
        "Expressive large eyes with detailed highlights and reflections. Stylized proportions typical of anime character design. "
        "Smooth gradients and sharp shadows (cel shading). NOT sketch/rough, NOT chibi/super-deformed unless specified.\n"
        "Scene description: [Describe the scene action/activity based on user message]\n"
        "Character details: [Use the complete participant descriptions - adapt features to anime aesthetics while preserving key characteristics] "
        "Clothing & attire: [Specify appropriate anime-style clothing based on scene context]. "
        "Facial expression: [HIGH EMPHASIS - anime-style expressive emotions]. Pose & body language: [Dynamic anime posing].\n"
        "Environment & background: [Setting with anime art style elements]\n"
        "Lighting & color: [Vibrant anime color palette and dramatic lighting]\n"
        "Composition & framing: [CRITICAL: Character fills 60-70% of frame - proper proportions, clearly visible detailed features].\n\n"
        "For realistic style, use this template:\n"
        "Style & rendering: Photorealistic quality with natural skin textures, lifelike details. Proper subsurface scattering, pore detail, "
        "realistic hair strands. Natural proportions and anatomy. Professional photography quality lighting and depth of field. "
        "NOT painted/illustrated, NOT CG/rendered look - actual photographic realism.\n"
        "Scene description: [Describe the scene action/activity based on user message]\n"
        "Character details: [Use the complete participant descriptions - translate to realistic photographic appearance] "
        "Clothing & attire: [Specify realistic clothing with fabric textures and natural wear]. "
        "Facial expression: [Subtle realistic human expressions]. Pose & body language: [Natural human posing].\n"
        "Environment & background: [Realistic setting with environmental details]\n"
        "Lighting & color: [Natural or professional photography lighting setup]\n"
        "Composition & framing: [CRITICAL: Character fills 60-70% of frame - photographic composition, sharp focus on subjects].\n\n"
        "CRITICAL NSFW MODERATION: If nudity/explicit content, ADD clothing/coverings/occlusions. Suggestive poses OK if clothed. No straight nudity.\n\n"
        "CRITICAL: Use the style-specific template above. Include ALL participants with their complete descriptions. "
        "Character must be properly sized - NOT miniature/child-like scale. "
        "Output ONLY valid JSON with key 'expanded_prompt' containing the complete structured prompt with moderation applied."
    )
    
    user_input = {
        "user_message": request.user_message,
        "style": request.style,
        "participants": participants_info,
    }
    
    raw = _generate(system=system, user=json.dumps(user_input, ensure_ascii=False))
    
    try:
        obj = _extract_json_object(raw)
        # LLM returns nested object, flatten it into a single string
        if isinstance(obj.get("expanded_prompt"), dict):
            # Flatten the structured sections into a single prompt string
            sections = obj["expanded_prompt"]
            expanded_prompt = "\n\n".join([f"{k}: {v}" for k, v in sections.items()])
        else:
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
                "FULL_EXPANDED_PROMPT": expanded_prompt,  # Log complete prompt for debugging
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
    
    # Build comprehensive participants descriptions (matching profile format)
    participants_info = []
    for p in request.participants:
        char_desc = _build_character_description(p.handle, p.avatar_features)
        participants_info.append(char_desc)
    
    system = (
        "You are an expert at creating detailed image generation prompts. "
        "Create a prompt following this EXACT structure:\n\n"
        "For 3d_cartoon style, use this template:\n"
        "Style & rendering: High-quality stylized 3D cartoon render (NOT photorealistic). Toon shading with smooth clean materials. "
        "Simplified geometry but HIGH DETAIL in faces - proper eyes with pupils, defined features, expressive faces. "
        "Animated character/avatar look similar to Pixar/Disney quality 3D animation. Clean topology, soft subsurface skin scattering. "
        "Non-photoreal CG characters. NOT clay models, NOT low-poly, NOT oversimplified.\n"
        "Scene description: [Describe the scene action/activity based on user message]\n"
        "Character details: [Use the complete participant descriptions provided - they contain all physical features] "
        "Clothing & attire: [Specify appropriate clothing for each character based on scene context - casual/formal/athletic/etc. Match the setting and activity]. "
        "Facial expression: [HIGH EMPHASIS - derive from user message mood]. Pose & body language: [Based on scene context].\n"
        "Environment & background: [Setting details from user message]\n"
        "Lighting & color: [Describe lighting mood matching the scene]\n"
        "Composition & framing: [Camera angle and framing. CRITICAL: Characters should fill 60-70% of frame height - proper adult proportions, "
        "NOT tiny/child-sized figures. Medium shot or closer framing. Characters are the main focus, clearly visible with defined features].\n\n"
        "For anime style, use this template:\n"
        "Style & rendering: High-quality anime/manga art style with cel-shaded coloring. Clean linework, vibrant saturated colors. "
        "Expressive large eyes with detailed highlights and reflections. Stylized proportions typical of anime character design. "
        "Smooth gradients and sharp shadows (cel shading). NOT sketch/rough, NOT chibi/super-deformed unless specified.\n"
        "Scene description: [Describe the scene action/activity based on user message]\n"
        "Character details: [Use the complete participant descriptions - adapt features to anime aesthetics while preserving key characteristics] "
        "Clothing & attire: [Specify appropriate anime-style clothing for each character based on scene context]. "
        "Facial expression: [HIGH EMPHASIS - anime-style expressive emotions]. Pose & body language: [Dynamic anime posing].\n"
        "Environment & background: [Setting with anime art style elements]\n"
        "Lighting & color: [Vibrant anime color palette and dramatic lighting]\n"
        "Composition & framing: [CRITICAL: Characters fill 60-70% of frame - proper proportions, clearly visible detailed features].\n\n"
        "For realistic style, use this template:\n"
        "Style & rendering: Photorealistic quality with natural skin textures, lifelike details. Proper subsurface scattering, pore detail, "
        "realistic hair strands. Natural proportions and anatomy. Professional photography quality lighting and depth of field. "
        "NOT painted/illustrated, NOT CG/rendered look - actual photographic realism.\n"
        "Scene description: [Describe the scene action/activity based on user message]\n"
        "Character details: [Use the complete participant descriptions - translate to realistic photographic appearance] "
        "Clothing & attire: [Specify realistic clothing for each character with fabric textures and natural wear]. "
        "Facial expression: [Subtle realistic human expressions]. Pose & body language: [Natural human posing].\n"
        "Environment & background: [Realistic setting with environmental details]\n"
        "Lighting & color: [Natural or professional photography lighting setup]\n"
        "Composition & framing: [CRITICAL: Characters fill 60-70% of frame - photographic composition, sharp focus on subjects].\n\n"
        "CRITICAL NSFW MODERATION: If nudity/explicit content, ADD clothing/coverings/occlusions. Suggestive poses OK if clothed. No straight nudity.\n\n"
        "CRITICAL: Use the style-specific template above. Include ALL participants with their complete descriptions. "
        "Characters must be properly sized - NOT miniature/child-like scale. "
        "Output ONLY valid JSON with key 'expanded_prompt' containing the complete structured prompt with moderation applied."
    )
    
    user_input = {
        "user_message": request.user_message,
        "style": request.style,
        "participants": participants_info,
    }
    
    raw = _generate(system=system, user=json.dumps(user_input, ensure_ascii=False))
    
    try:
        obj = _extract_json_object(raw)
        # LLM returns nested object, flatten it into a single string
        if isinstance(obj.get("expanded_prompt"), dict):
            # Flatten the structured sections into a single prompt string
            sections = obj["expanded_prompt"]
            expanded_prompt = "\n\n".join([f"{k}: {v}" for k, v in sections.items()])
        else:
            expanded_prompt = obj.get("expanded_prompt", "")
    except Exception as e:
        logger.error("failed_to_parse_llm_json", extra={"extra_fields": {"error": str(e), "raw": raw[:500]}})
        # Fallback: basic concatenation with NSFW moderation
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
                "FULL_EXPANDED_PROMPT": expanded_prompt,  # Log complete prompt for debugging
            }
        },
    )
    
    return LlmScenesExpandResponse(
        expanded_prompt=expanded_prompt,
        shortened_message=shortened_message,
    )


