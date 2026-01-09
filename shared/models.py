from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class FaceObserved(BaseModel):
    face_shape: Optional[str] = None
    hair_color: Optional[str] = None
    hair_type: Optional[str] = None
    hair_length: Optional[str] = None
    eye_color: Optional[str] = None
    skin_tone: Optional[str] = None
    age_appearance: Optional[str] = None
    gender: Optional[str] = None
    facial_hair: Optional[str] = None


class DressObserved(BaseModel):
    dress_color: Optional[str] = None
    dress_type: Optional[str] = None


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
