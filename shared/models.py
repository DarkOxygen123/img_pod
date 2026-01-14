from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class FaceObserved(BaseModel):
    face_shape: Optional[str] = None
    hair_color: Optional[str] = None
    hair_type: Optional[str] = None
    hair_style: Optional[str] = None
    hair_length: Optional[str] = None
    hairline_type: Optional[str] = None
    balding_pattern: Optional[str] = None
    eye_color: Optional[str] = None
    eye_shape: Optional[str] = None
    skin_tone: Optional[str] = None
    skin_undertone: Optional[str] = None
    age_appearance: Optional[str] = None
    age_range: Optional[str] = None
    gender: Optional[str] = None
    facial_hair: Optional[str] = None
    facial_hair_density: Optional[str] = None
    beard_style: Optional[str] = None
    mustache_style: Optional[str] = None
    facial_marks: Optional[str] = None
    facial_mark_position: Optional[str] = None
    expression: Optional[str] = None


class DressObserved(BaseModel):
    dress_color: Optional[str] = None
    dress_type: Optional[str] = None


class AccessoriesObserved(BaseModel):
    hat_present: Optional[str] = None  # yes, no
    hat_style: Optional[str] = None
    hat_color: Optional[str] = None
    glasses_present: Optional[str] = None  # yes, no
    glasses_type: Optional[str] = None
    glasses_color: Optional[str] = None
    mask_present: Optional[str] = None  # yes, no
    mask_type: Optional[str] = None
    mask_color: Optional[str] = None


class FaceRenderControls(BaseModel):
    lighting_preference: Optional[str] = None
    angle_preference: Optional[str] = None


class FaceMeta(BaseModel):
    face_detected: bool = False
    num_faces: int = 0
    quality_score: float = 0.0


class FaceProfileFeaturesV1(BaseModel):
    observed: FaceObserved = Field(default_factory=FaceObserved)
    dress: DressObserved = Field(default_factory=DressObserved)
    accessories: AccessoriesObserved = Field(default_factory=AccessoriesObserved)
    meta: FaceMeta = Field(default_factory=FaceMeta)


class Avatar(BaseModel):
    handle: str
    avatar_features: FaceProfileFeaturesV1


class Participants(BaseModel):
    sender_avatar: Avatar
    receiver_avatar: Optional[Avatar] = None
    other_avatars: List[Avatar] = Field(default_factory=list)


class MoodEnum(str, Enum):
    neutral = "neutral"
    playful = "playful"
    warm = "warm"
    tense = "tense"
    sad = "sad"
    awkward = "awkward"
    flirty = "flirty"


class Mood(BaseModel):
    mood: MoodEnum = MoodEnum.neutral
    intensity: Literal[0, 1, 2] = 0
    confidence: float = 0.0
    pending_reply: bool = False
    unreplied_count: int = 0


class StyleRequest(BaseModel):
    style_id: str = "suko_3d"


class Message(BaseModel):
    sent_at: Optional[str] = None
    text: str


class MessageHistory(BaseModel):
    messages: List[Message] = Field(default_factory=list)


class PromptBundle(BaseModel):
    style_line: str
    identity_line: str
    chat_mood_line: str
    scene_line: str
    pose_expression_line: str
    env_frame_line: str
    negative_line: str
    final_prompt: str


class PolicyCode(str, Enum):
    BAD_SELFIE = "BAD_SELFIE"
    OVERLOADED = "OVERLOADED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    QUALITY_FAILED = "QUALITY_FAILED"
    BAD_INPUT = "BAD_INPUT"


class ErrorResponse(BaseModel):
    code: PolicyCode
    message: str
    details: Optional[dict] = None
    retry_after_ms: Optional[int] = None


class ProfileCreateResponse(BaseModel):
    avatar_features: FaceProfileFeaturesV1


class ProfileCreate422Response(BaseModel):
    code: PolicyCode = PolicyCode.BAD_SELFIE
    message: str
    details: dict


class ProfileCreateRequest(BaseModel):
    # Multipart handled at endpoint; placeholder for typing.
    pass


class ProfileUpdateRequest(BaseModel):
    avatar_features: FaceProfileFeaturesV1


class ChatPrivateImageGenRequest(BaseModel):
    sender_id: str
    receiver_id: str
    style_request: StyleRequest
    new_message: Message
    history: MessageHistory
    participants: Participants


class GeneralImageGenRequest(BaseModel):
    sender_id: str
    style_request: StyleRequest
    prompt: Message
    participants: Participants


class LlmBundleRequest(BaseModel):
    participants: Participants
    chat_history: MessageHistory
    new_message: Message
    mood: Mood
    style_request: StyleRequest
    is_general: bool = False


class LlmBundleResponse(BaseModel):
    prompt_bundle: PromptBundle
    caption_text: Optional[str] = None


class ShortenRequest(BaseModel):
    text: str
    max_len: int = 90


class ShortenResponse(BaseModel):
    text: str
    shortened: bool


class WorkerText2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0


class WorkerText2ImgResponse(BaseModel):
    image_bytes: bytes
    attempt: int = 1


class WorkerProfileRequest(BaseModel):
    # Multipart handled at endpoint.
    pass


class WorkerProfileResponse(BaseModel):
    avatar_features: FaceProfileFeaturesV1
    image_bytes: bytes


# 1:1 Chat Image Generation Models

class ChatMessage(BaseModel):
    """Single chat message with sender info and timestamp."""
    sender_handle: str
    text: str
    timestamp: str  # ISO 8601 format
    tagged_handles: List[str] = Field(default_factory=list)


class Chat1to1Participant(BaseModel):
    """Participant in 1:1 chat with their profile features."""
    handle: str
    avatar_features: FaceProfileFeaturesV1


class Chat1to1ImageGenRequest(BaseModel):
    """Request for 1:1 chat contextual image generation."""
    chat_messages: List[ChatMessage] = Field(..., min_length=1, max_length=15)
    style: str  # e.g., "anime", "realistic", "3d_cartoon", "oil_painting"
    participants: List[Chat1to1Participant] = Field(..., min_length=2)  # At least 2 chatters
    target_message: str  # The specific message to visualize


class Chat1to1ImageGenResponse(BaseModel):
    """Response with generated image, caption, and final prompt."""
    image_base64: str
    caption: str  # 10-15 word social media caption
    prompt_used: str  # Original or shortened user intent


# LLM Service - Chat Context Expansion Models

class LlmChat1to1ExpandRequest(BaseModel):
    """Request for LLM to expand chat context into detailed prompt."""
    chat_messages: List[ChatMessage]
    style: str
    participants: List[Chat1to1Participant]
    target_message: str


class LlmChat1to1ExpandResponse(BaseModel):
    """LLM expanded prompt with context."""
    expanded_prompt: str
    shortened_target: Optional[str] = None  # If target_message was >25 words


class LlmCaptionRequest(BaseModel):
    """Request for short social media caption."""
    prompt: str
    

class LlmCaptionResponse(BaseModel):
    """Short caption for the generated image."""
    caption: str  # 10-15 words


# Worker - 1:1 Chat Image Generation

class WorkerChat1to1Request(BaseModel):
    """Worker request for 1:1 chat image generation."""
    prompt: str
    negative_prompt: Optional[str] = None
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0


class WorkerChat1to1Response(BaseModel):
    """Worker response with generated image."""
    image_bytes: bytes


# Shorts/Scenes Image Generation Models (No chat history, NSFW moderated)

class ShortsParticipant(BaseModel):
    """Participant in shorts generation with their profile features."""
    handle: str
    avatar_features: FaceProfileFeaturesV1


class ShortsImageGenRequest(BaseModel):
    """Request for shorts image generation (no chat history)."""
    style: str  # e.g., "anime", "realistic", "3d_cartoon", "oil_painting"
    participants: List[ShortsParticipant] = Field(..., min_length=1)
    user_message: str  # The message to visualize


class ShortsImageGenResponse(BaseModel):
    """Response with generated image, caption, and prompt."""
    image_base64: str
    caption: str  # 10-15 word social media caption
    prompt_used: str  # Original or shortened user message


class ScenesParticipant(BaseModel):
    """Participant in scenes generation with their profile features."""
    handle: str
    avatar_features: FaceProfileFeaturesV1


class ScenesImageGenRequest(BaseModel):
    """Request for scenes image generation (no chat history)."""
    style: str  # e.g., "anime", "realistic", "3d_cartoon", "oil_painting"
    participants: List[ScenesParticipant] = Field(..., min_length=1)
    user_message: str  # The message to visualize


class ScenesImageGenResponse(BaseModel):
    """Response with generated image, caption, and prompt."""
    image_base64: str
    caption: str  # 10-15 word social media caption
    prompt_used: str  # Original or shortened user message


# LLM Service - Shorts/Scenes Expansion Models

class LlmShortsExpandRequest(BaseModel):
    """Request for LLM to expand shorts prompt with NSFW moderation."""
    style: str
    participants: List[ShortsParticipant]
    user_message: str


class LlmShortsExpandResponse(BaseModel):
    """LLM expanded prompt with NSFW moderation applied."""
    expanded_prompt: str
    shortened_message: Optional[str] = None  # If user_message was >25 words


class LlmScenesExpandRequest(BaseModel):
    """Request for LLM to expand scenes prompt with NSFW moderation."""
    style: str
    participants: List[ScenesParticipant]
    user_message: str


class LlmScenesExpandResponse(BaseModel):
    """LLM expanded prompt with NSFW moderation applied."""
    expanded_prompt: str
    shortened_message: Optional[str] = None  # If user_message was >25 words


# Worker - Shorts/Scenes Image Generation

class WorkerShortsRequest(BaseModel):
    """Worker request for shorts image generation."""
    prompt: str
    negative_prompt: Optional[str] = None
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0


class WorkerShortsResponse(BaseModel):
    """Worker response with generated image."""
    image_bytes: bytes


class WorkerScenesRequest(BaseModel):
    """Worker request for scenes image generation."""
    prompt: str
    negative_prompt: Optional[str] = None
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0


class WorkerScenesResponse(BaseModel):
    """Worker response with generated image."""
    image_bytes: bytes

