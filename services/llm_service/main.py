from fastapi import FastAPI

from shared.ajay import ajay_rule
from shared.logging_config import get_logger
from shared.models import LlmBundleRequest, LlmBundleResponse, PromptBundle, ShortenRequest, ShortenResponse
from shared.settings import config

app = FastAPI()
logger = get_logger(__name__)
settings = config.llm_settings()


def build_prompt_bundle(request: LlmBundleRequest) -> PromptBundle:
    participants_text = request.participants.model_dump_json()
    style_line = f"Style: {request.style_request.style_id}"
    identity_line = f"Participants: {participants_text}"
    chat_mood_line = "" if request.is_general else f"Mood: {request.mood.mood} intensity {request.mood.intensity}"
    scene_line = f"Scene from message: {request.new_message.text}"
    pose_expression_line = "Natural expressions"
    env_frame_line = "Medium shot, balanced lighting"
    negative_line = "blurry, low quality, text, watermark"
    final_prompt = " ".join(
        [style_line, identity_line, chat_mood_line, scene_line, pose_expression_line, env_frame_line]
    ).strip()
    return PromptBundle(
        style_line=style_line,
        identity_line=identity_line,
        chat_mood_line=chat_mood_line,
        scene_line=scene_line,
        pose_expression_line=pose_expression_line,
        env_frame_line=env_frame_line,
        negative_line=negative_line,
        final_prompt=final_prompt,
    )


@app.post("/v1/bundle", response_model=LlmBundleResponse)
async def bundle(request: LlmBundleRequest) -> LlmBundleResponse:
    bundle = build_prompt_bundle(request)
    caption, shortened = ajay_rule(request.new_message.text)
    logger.info(
        "built_prompt_bundle",
        extra_fields={"shortened": shortened, "final_prompt_len": len(bundle.final_prompt)},
    )
    return LlmBundleResponse(prompt_bundle=bundle, caption_text=caption)


@app.post("/v1/shorten", response_model=ShortenResponse)
async def shorten(request: ShortenRequest) -> ShortenResponse:
    text, shortened = ajay_rule(request.text, request.max_len)
    return ShortenResponse(text=text, shortened=shortened)
