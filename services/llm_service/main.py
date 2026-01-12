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
from shared.models import LlmBundleRequest, LlmBundleResponse, PromptBundle, ShortenRequest, ShortenResponse
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
