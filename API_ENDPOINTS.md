# API Endpoints Documentation

**Base URL**: `https://bw77wupwq7k752-8000.proxy.runpod.net`

All endpoints return JSON responses. Timestamps use ISO 8601 format (e.g., `2026-01-14T12:30:00Z`).

---

## Table of Contents
1. [Profile Management](#1-profile-management)
2. [1:1 Chat Image Generation](#2-11-chat-image-generation)
3. [Shorts Image Generation](#3-shorts-image-generation)
4. [Scenes Image Generation](#4-scenes-image-generation)
5. [Private Chat Image Generation](#5-private-chat-image-generation)
6. [General Image Generation](#6-general-image-generation)
7. [Error Responses](#error-responses)

---

## 1. Profile Management

### POST /v1/profile/create
**Description**: Creates a user profile by analyzing a selfie image.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: Image file (JPEG/PNG)

**Success Response** (200):
```json
{
  "avatar_features": {
    "observed": {
      "face_shape": "oval",
      "hair_color": "brown",
      "hair_type": "straight",
      "hair_style": "short",
      "hair_length": "medium",
      "eye_color": "brown",
      "eye_shape": "almond",
      "skin_tone": "medium",
      "skin_undertone": "warm",
      "age_appearance": "young adult",
      "age_range": "20-30",
      "gender": "male",
      "facial_hair": "none",
      "expression": "neutral"
    },
    "dress": {
      "dress_color": "blue",
      "dress_type": "casual shirt"
    },
    "accessories": {
      "hat_present": "no",
      "glasses_present": "yes",
      "glasses_type": "rectangular",
      "glasses_color": "black",
      "mask_present": "no"
    },
    "meta": {
      "face_detected": true,
      "num_faces": 1,
      "quality_score": 0.95
    }
  }
}
```

**Error Response** (422):
```json
{
  "code": "BAD_SELFIE",
  "message": "No face detected in image",
  "details": {
    "num_faces": 0
  }
}
```

---

### POST /v1/profile/update
**Description**: Updates specific profile fields (hair, skin, accessories) and regenerates avatar.

**Request**:
```json
{
  "hair_color": "blonde",
  "hair_type": "wavy",
  "hair_style": "long",
  "hair_length": "shoulder-length",
  "skin_tone": "fair",
  "skin_undertone": "cool",
  "hat_present": "yes",
  "hat_style": "fedora",
  "hat_color": "black",
  "mask_present": "no"
}
```

**Request Schema** (all fields optional):
- **Hair**: `hair_color`, `hair_type`, `hair_style`, `hair_length`
- **Skin**: `skin_tone`, `skin_undertone`
- **Accessories**: `hat_present`, `hat_style`, `hat_color`, `mask_present`, `mask_type`, `mask_color`

**Success Response** (200):
- Content-Type: `image/png`
- Binary PNG image of regenerated avatar

---

## 2. 1:1 Chat Image Generation

### POST /v1/chat/1to1/imagegen
**Description**: Generates a contextual image based on 1:1 chat conversation history. **NO NSFW RESTRICTIONS** - generates any content based on context.

**Important Notes**:
- **Tagged Handles**: Use `tagged_handles` to reference people NOT in the chat (e.g., friends mentioned in conversation). The 2 main participants are already included in `participants` array.
- **Participants**: Must include complete `avatar_features` JSON from profile creation for both chatters.
- **Captions**: Generated in 1st/2nd person ("us", "we", "our") - NOT 3rd person.

**Request**:
```json
{
  "chat_messages": [
    {
      "sender_handle": "alice",
      "text": "What do you think about the sunset?",
      "timestamp": "2026-01-14T18:00:00Z",
      "tagged_handles": []
    },
    {
      "sender_handle": "bob",
      "text": "It's beautiful! Reminds me of our trip with @charlie",
      "timestamp": "2026-01-14T18:01:30Z",
      "tagged_handles": ["charlie"]
    },
    {
      "sender_handle": "alice",
      "text": "Let's imagine us walking on the beach during sunset",
      "timestamp": "2026-01-14T18:03:00Z",
      "tagged_handles": []
    }
  ],
  "style": "realistic",
  "participants": [
    {
      "handle": "alice",
      "avatar_features": {
        "observed": {
          "face_shape": "oval",
          "hair_color": "blonde",
          "hair_type": "wavy",
          "hair_style": "long",
          "hair_length": "shoulder-length",
          "eye_color": "blue",
          "eye_shape": "almond",
          "skin_tone": "fair",
          "skin_undertone": "cool",
          "age_appearance": "young adult",
          "age_range": "20-30",
          "gender": "female",
          "expression": "happy"
        },
        "dress": {
          "dress_color": "white",
          "dress_type": "casual"
        },
        "accessories": {
          "hat_present": "no",
          "glasses_present": "no",
          "mask_present": "no"
        },
        "meta": {
          "face_detected": true,
          "num_faces": 1,
          "quality_score": 0.95
        }
      }
    },
    {
      "handle": "bob",
      "avatar_features": {
        "observed": {
          "face_shape": "square",
          "hair_color": "black",
          "hair_type": "straight",
          "hair_style": "short",
          "eye_color": "brown",
          "skin_tone": "medium",
          "age_range": "25-35",
          "gender": "male"
        },
        "dress": {},
        "accessories": {
          "hat_present": "no",
          "glasses_present": "no",
          "mask_present": "no"
        },
        "meta": {
          "face_detected": true,
          "num_faces": 1,
          "quality_score": 0.93
        }
      }
    }
  ],
  "target_message": "Let's imagine us walking on the beach during sunset"
}
```

**Request Schema**:
- `chat_messages` (array, 1-15 messages): Conversation history
  - `sender_handle` (string): Username of sender (must match a participant)
  - `text` (string): Message content
  - `timestamp` (string): ISO 8601 timestamp
  - `tagged_handles` (array): Usernames mentioned who are NOT participants (e.g., friends referenced in conversation)
- `style` (string): Visual style - "anime", "realistic", "3d_cartoon", "oil_painting", "watercolor", "comic_book", "pixel_art"
- `participants` (array, exactly 2): The 2 people chatting with COMPLETE avatar features from profile creation
- `target_message` (string): The specific message to visualize

**Success Response** (200):
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "caption": "Our perfect sunset moment together 🌅",
  "prompt_used": "walking on the beach during sunset"
}
```

**Response Schema**:
- `image_base64` (string): Base64-encoded PNG image (1024x1024)
- `caption` (string): 10-15 word intimate caption using 1st/2nd person perspective
- `prompt_used` (string): Final prompt used (may be shortened if original >25 words)

**Error Response** (429):
```json
{
  "code": "OVERLOADED",
  "message": "Queue is full, please retry later",
  "retry_after_ms": 5000
}
```

---

## 3. Shorts Image Generation

### POST /v1/chat/shorts/generate
**Description**: Generates short-form content images (similar to Instagram Stories/TikTok). **NSFW MODERATED** - avoids explicit nudity by adding clothing/occlusions.

**Important Notes**:
- **Participants**: Must include complete `avatar_features` JSON from profile creation.
- **Captions**: Emotion-focused, capturing the FEELING/MOOD rather than describing the image.

**Request**:
```json
{
  "style": "anime",
  "participants": [
    {
      "handle": "alice",
      "avatar_features": {
        "observed": {
          "face_shape": "oval",
          "hair_color": "blonde",
          "hair_type": "wavy",
          "hair_style": "long",
          "hair_length": "shoulder-length",
          "eye_color": "blue",
          "skin_tone": "fair",
          "age_range": "20-30",
          "gender": "female",
          "expression": "confident"
        },
        "dress": {
          "dress_type": "elegant"
        },
        "accessories": {
          "hat_present": "no",
          "glasses_present": "no",
          "mask_present": "no"
        },
        "meta": {
          "face_detected": true,
          "num_faces": 1,
          "quality_score": 0.95
        }
      }
    }
  ],
  "user_message": "A confident woman posing elegantly in a luxurious bedroom"
}
```

**Request Schema**:
- `style` (string): Visual style - "anime", "realistic", "3d_cartoon", "oil_painting", "watercolor", "comic_book", "pixel_art"
- `participants` (array, minimum 1): People to include with COMPLETE avatar features from profile creation
- `user_message` (string): Description of the scene to generate

**Success Response** (200):
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "caption": "Confidence radiating through every pose ✨",
  Important Notes**:
- **Participants**: Must include complete `avatar_features` JSON from profile creation.
- **Captions**: Emotion-focused, capturing the FEELING/MOOD rather than describing the image.

**Request**:
```json
{
  "style": "realistic",
  "participants": [
    {
      "handle": "bob",
      "avatar_features": {
        "observed": {
          "face_shape": "square",
          "hair_color": "black",
          "hair_type": "straight",
          "eye_color": "brown",
          "skin_tone": "medium",
          "age_range": "25-35",
          "gender": "male",
          "expression": "excited"
        },
        "dress": {
          "dress_type": "casual"
        },
        "accessories": {
          "hat_present": "no",
          "glasses_present": "no",
          "mask_present": "no"
        },
        "meta": {
          "face_detected": true,
          "num_faces": 1,
          "quality_score": 0.92
        }
      }
    },
    {
      "handle": "charlie",
      "avatar_features": {
        "observed": {
          "face_shape": "round",
          "hair_color": "brown",
          "skin_tone": "tan",
          "age_range": "30-40",
          "gender": "male"
        },
        "dress": {},
        "accessories": {
          "hat_present": "no",
          "glasses_present": "yes",
          "glasses_type": "round",
          "mask_present": "no"
        },
        "meta": {
          "face_detected": true,
          "num_faces": 1,
          "quality_score": 0.90
        }
      }
    }
  ],
  "user_message": "Two friends celebrating a victory at a sports bar"
}
```

**Request Schema**: (Same as Shorts)
- `style` (string): Visual style
- `participants` (array, minimum 1): People to include with COMPLETE avatar features from profile creation
- `user_message` (string): Scene description

**Success Response** (200):
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "caption": "Victory tastes sweeter when shared with your crew 🍻",
  "prompt_used": "Two friends celebrating a victory at a sports bar"
}
```

**Response Schema**: (Same as Shorts)
- `image_base64` (string): Base64-encoded PNG image
- `caption` (string): 10-15 word emotion-focused caption (captures feeling)

**Request Schema**: (Same as Shorts)
- `style` (string): Visual style
- `participants` (array, minimum 1): People to include
- `user_message` (string): Scene description

**Success Response** (200):
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "caption": "Bob and Charlie cheering their team's win together",
  "prompt_used": "Two friends celebrating a victory at a sports bar"
}
```

**Response Schema**: (Same as Shorts)
- `image_base64` (string): Base64-encoded PNG image
- `caption` (string): Social media caption
- `prompt_used` (string): Final prompt

**NSFW Handling**: Same moderation as Shorts endpoint.

---

## 5. Private Chat Image Generation

### POST /v1/chat/private/imagegen
**Description**: Generates images for private chats with policy filtering.

**Request**:
```json
{
  "sender_id": "user123",
  "receiver_id": "user456",
  "style_request": {
    "style_id": "suko_3d"
  },
  "new_message": {
    "sent_at": "2026-01-14T15:30:00Z",
    "text": "Let's meet at the coffee shop"
  },
  "history": {
    "messages": [
      {
        "sent_at": "2026-01-14T15:25:00Z",
        "text": "Hey, what are you up to?"
      }
    ]
  },
  "participants": {
    "sender_avatar": {
      "handle": "user123",
      "avatar_features": {
        "observed": {
          "face_shape": "oval"
        },
        "meta": {
          "face_detected": true,
          "num_faces": 1,
          "quality_score": 0.95
        }
      }
    },
    "receiver_avatar": {
      "handle": "user456",
      "avatar_features": {
        "observed": {
          "face_shape": "round"
        },
        "meta": {
          "face_detected": true,
          "num_faces": 1,
          "quality_score": 0.92
        }
      }
    }
  }
}
```

**Success Response** (200):
- Content-Type: `image/png`
- Binary PNG image data

**Error Response** (422):
```json
{
  "code": "POLICY_BLOCKED",
  "message": "Content violates policy",
  "details": {
    "matched_keywords": ["inappropriate_word"]
  }
}
```

---

## 6. General Image Generation

### POST /v1/imagegen
**Description**: General-purpose image generation with content sanitization.

**Request**:
```json
{
  "sender_id": "user123",
  "style_request": {
    "style_id": "suko_3d"
  },
  "prompt": {
    "sent_at": "2026-01-14T16:00:00Z",
    "text": "A serene mountain landscape at dawn"
  },
  "participants": {
    "sender_avatar": {
      "handle": "user123",
      "avatar_features": {
        "observed": {
          "face_shape": "oval"
        },
        "meta": {
          "face_detected": true,
          "num_faces": 1,
          "quality_score": 0.95
        }
      }
    }
  }
}
```

**Success Response** (200):
- Content-Type: `image/png`
- Binary PNG image data

---

## Error Responses

All endpoints may return these error codes:

### 422 - Validation/Policy Error
```json
{
  "code": "BAD_SELFIE" | "POLICY_BLOCKED" | "BAD_INPUT",
  "message": "Human-readable error description",
  "details": {
    "additional": "context"
  }
}
```

### 429 - Rate Limited / Queue Full
```json
{
  "code": "OVERLOADED",
  "message": "Queue is full, please retry later",
  "retry_after_ms": 5000
}
```

### 500 - Internal Server Error
```json
{
  "detail": "Internal server error description"
}
```

---

## Common Data Types

### FaceProfileFeaturesV1
Complete schema for avatar features:
```json
{
  "observed": {
    "face_shape": "oval|round|square|heart|diamond",
    "hair_color": "black|brown|blonde|red|gray|white",
    "hair_type": "straight|wavy|curly|coily",
    "hair_style": "short|long|bald|tied|loose",
    "hair_length": "short|medium|long",
    "eye_color": "brown|blue|green|hazel|gray",
    "eye_shape": "almond|round|hooded|monolid",
    "skin_tone": "fair|light|medium|olive|tan|brown|dark",
    "skin_undertone": "warm|cool|neutral",
    "age_appearance": "child|teen|young adult|middle-aged|senior",
    "age_range": "0-12|13-19|20-30|31-45|46-60|60+",
    "gender": "male|female|non-binary",
    "facial_hair": "none|light|medium|full",
    "beard_style": "goatee|full beard|stubble|clean shaven",
    "expression": "neutral|happy|sad|angry|surprised"
  },
  "dress": {
    "dress_color": "color name",
    "dress_type": "casual|formal|sportswear|traditional"
  },
  "accessories": {
    "hat_present": "yes|no",
    "hat_style": "cap|fedora|beanie",
    "glasses_present": "yes|no",
    "glasses_type": "rectangular|round|cat-eye",
    "mask_present": "yes|no"
  },
  "meta": {
    "face_detected": true|false,
    "num_faces": 0-10,
    "quality_score": 0.0-1.0
  }
}
```

### Style Options
Available styles for all generation endpoints:
- `"anime"` - Japanese animation style
- `"realistic"` - Photorealistic rendering
- `"3d_cartoon"` - 3D animated character style
- `"oil_painting"` - Traditional oil painting aesthetic
- `"watercolor"` - Watercolor painting style
- `"comic_book"` - Comic book/graphic novel style
- `"pixel_art"` - Retro pixel art style

---

## Rate Limits & Performance

- **Profile Creation**: ~90 seconds (includes VQA analysis + image generation)
- **1:1 Chat Generation**: ~15 seconds (LLM expansion + image generation + caption)
- **Shorts/Scenes Generation**: ~15 seconds (LLM expansion + moderation + generation)
- **Queue Capacity**:
  - Profile: 10 requests
  - 1:1 Chat: 10 requests
  - Shorts: 10 requests
  - Scenes: 10 requests

When queues are full, requests return HTTP 429 with `retry_after_ms` header.

---

## Authentication

**Current**: No authentication required (pods accessible via RunPod proxy URLs).

**Production**: Add API key authentication in your backend before exposing to clients.

---

## Testing

**Health Check**:
```bash
curl https://bw77wupwq7k752-8000.proxy.runpod.net/healthz
# Returns: {"status":"healthy"}
```

**Example cURL - Profile Creation**:
```bash
curl -X POST https://bw77wupwq7k752-8000.proxy.runpod.net/v1/profile/create \
  -F "file=@selfie.jpg"
```

**Example cURL - 1:1 Chat**:
```bash
curl -X POST https://bw77wupwq7k752-8000.proxy.runpod.net/v1/chat/1to1/imagegen \
  -H "Content-Type: application/json" \
  -d '{
    "chat_messages": [
      {
        "sender_handle": "alice",
        "text": "Let's go to the beach!",
        "timestamp": "2026-01-14T12:00:00Z",
        "tagged_handles": []
      }
    ],
    "style": "realistic",
    "participants": [
      {
        "handle": "alice",
        "avatar_features": {...}
      }
    ],
    "target_message": "Let's go to the beach!"
  }'
```
