"""
API Examples - How to Use Image Generation Endpoints

This script demonstrates all available endpoints:
1. Profile Creation - Upload selfie to extract features
2. Profile Update - Update specific profile fields
3. 1:1 Chat Image Generation - Contextual chat images (NO NSFW restrictions)
4. Shorts Generation - Short-form content (NSFW moderated)
5. Scenes Generation - Scene-based images (NSFW moderated)

Requirements:
    pip install requests pillow

Usage:
    python testing/api_examples.py
"""

import base64
import json
import os
from datetime import datetime
from email import message_from_bytes
from email.policy import default
from pathlib import Path

import requests


# Configuration
BASE_URL = "https://bw77wupwq7k752-8000.proxy.runpod.net"
OUTPUT_DIR = Path("testing/api_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _parse_multipart_mixed(content_type: str, body: bytes) -> tuple[dict | None, bytes | None]:
    """Parse multipart/mixed HTTP body and return (json_obj, image_bytes)."""
    if not content_type or "multipart/mixed" not in content_type:
        return None, None

    # The email parser expects headers; prepend minimal headers so it can parse.
    mime_bytes = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).encode("utf-8") + body

    msg = message_from_bytes(mime_bytes, policy=default)
    if not msg.is_multipart():
        return None, None

    json_obj: dict | None = None
    image_bytes: bytes | None = None
    for part in msg.iter_parts():
        part_type = part.get_content_type()
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if part_type == "application/json" and json_obj is None:
            try:
                json_obj = json.loads(payload.decode("utf-8"))
            except Exception:
                json_obj = None
        elif part_type.startswith("image/") and image_bytes is None:
            image_bytes = payload

    return json_obj, image_bytes


def save_image(image_base64: str, filename: str) -> str:
    """Save base64 encoded image to file."""
    image_bytes = base64.b64decode(image_base64)
    filepath = OUTPUT_DIR / filename
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    print(f"✅ Saved: {filepath}")
    return str(filepath)


def save_json(data: dict, filename: str) -> str:
    """Save JSON data to file."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {filepath}")
    return str(filepath)


# ============================================================================
# 1. PROFILE CREATION - Extract features from selfie
# ============================================================================


def example_profile_create(name: str, selfie_path: str):
    """
    Upload a selfie image to extract facial features.

    Args:
        name: Person's name (e.g., "Alice", "Bob")
        selfie_path: Path to selfie image

    Returns complete avatar_features JSON needed for other endpoints.
    """
    print("\n" + "=" * 70)
    print(f"PROFILE CREATION - Analyzing {name}'s Selfie")
    print("=" * 70)

    # Use provided selfie path
    # selfie_path provided as parameter

    if not os.path.exists(selfie_path):
        print(f"❌ Selfie not found: {selfie_path}")
        print("   Generate test selfies first: python testing/generate_test_selfies.py")
        return None

    with open(selfie_path, "rb") as f:
        files = {"file": ("selfie.jpg", f, "image/jpeg")}

        print(f"📤 Uploading: {selfie_path}")
        response = requests.post(
            f"{BASE_URL}/v1/profile/create",
            files=files,
            timeout=120,  # Profile creation takes ~90 seconds
        )

    if response.status_code == 200:
        content_type = response.headers.get("content-type", "")
        if "multipart/mixed" in content_type:
            result, image_bytes = _parse_multipart_mixed(content_type, response.content)
            if image_bytes:
                save_image(
                    base64.b64encode(image_bytes).decode("utf-8"),
                    f"profile_image_{name.lower()}.png",
                )
            if result is None:
                print("❌ Could not parse multipart response JSON")
                print(f"   Content-Type: {content_type}")
                return None
        else:
            result = response.json()
        print(f"✅ {name}'s profile created successfully!")
        print(f"   Face detected: {result['avatar_features']['meta']['face_detected']}")
        print(f"   Quality score: {result['avatar_features']['meta']['quality_score']}")

        # Save full features JSON for reference
        save_json(result, f"profile_features_{name.lower()}.json")

        return result["avatar_features"]
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None


# ============================================================================
# 2. PROFILE UPDATE - Update specific fields
# ============================================================================


def example_profile_update():
    """
    Update specific profile fields (hair, skin, accessories).

    Only fields you provide will be updated - no need to send full avatar_features.
    Returns: Updated avatar image (PNG)
    """
    print("\n" + "=" * 70)
    print("2. PROFILE UPDATE - Modifying Hair & Accessories")
    print("=" * 70)

    # Update request - only specify fields you want to change
    update_data = {
        # Hair updates
        "hair_color": "platinum blonde",
        "hair_type": "wavy",
        "hair_style": "long flowing",
        "hair_length": "waist-length",
        # Skin updates
        "skin_tone": "fair",
        "skin_undertone": "cool",
        # Accessories
        "hat_present": "yes",
        "hat_style": "wide-brim sun hat",
        "hat_color": "white",
        "mask_present": "no",
    }

    print(f"📤 Updating profile with: {list(update_data.keys())}")
    response = requests.post(
        f"{BASE_URL}/v1/profile/update", json=update_data, timeout=120
    )

    if response.status_code == 200:
        print("✅ Profile updated successfully!")

        # Save the updated avatar image
        image_path = OUTPUT_DIR / "profile_updated.png"
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved updated avatar: {image_path}")

        return True
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return False


# ============================================================================
# 3. 1:1 CHAT IMAGE GENERATION - Contextual chat images
# ============================================================================


def example_1to1_chat(alice_features: dict, bob_features: dict, charlie_features: dict):
    """
    Generate contextual image based on 1:1 chat conversation.

    - NO NSFW restrictions
    - Captions in 1st/2nd person ("us", "we", "our")
    - tagged_handles for people NOT in chat (referenced friends)
    - Include avatar_features for all tagged people - LLM decides if they appear in image
    """
    print("\n" + "=" * 70)
    print("3. 1:1 CHAT IMAGE GENERATION")
    print("=" * 70)

    request_data = {
        "chat_messages": [
            {
                "sender_handle": "alice",
                "text": "Remember that sunset at the beach?",
                "timestamp": "2026-01-14T18:00:00Z",
                "tagged_handles": [],  # No one mentioned
            },
            {
                "sender_handle": "bob",
                "text": "How could I forget? That was magical with @charlie's guitar playing",
                "timestamp": "2026-01-14T18:01:30Z",
                "tagged_handles": [
                    "charlie"
                ],  # Charlie is NOT in this chat, just mentioned
            },
            {
                "sender_handle": "alice",
                "text": "Let's imagine us back there, walking hand in hand along the shore",
                "timestamp": "2026-01-14T18:03:00Z",
                "tagged_handles": [],
            },
        ],
        "style": "realistic",  # anime, realistic, 3d_cartoon, oil_painting, watercolor
        "participants": [
            {
                "handle": "alice",
                "avatar_features": alice_features,  # Full features from profile creation
            },
            {"handle": "bob", "avatar_features": bob_features},
            {
                "handle": "charlie",
                "avatar_features": charlie_features,  # Tagged in message - LLM decides if he appears
            },
        ],
        "target_message": "Let's imagine us back there, walking hand in hand along the shore",
    }

    print("📤 Generating 1:1 chat image...")
    print(f"   Style: {request_data['style']}")
    print(f"   Participants: {len(request_data['participants'])}")
    print(f"   Chat history: {len(request_data['chat_messages'])} messages")

    response = requests.post(
        f"{BASE_URL}/v1/chat/1to1/imagegen",
        json=request_data,
        timeout=30,  # ~15 seconds expected
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Image generated successfully!")
        print(f"   Caption: {result['caption']}")
        print(f"   Prompt used: {result['prompt_used']}")

        # Save image and metadata
        save_image(result["image_base64"], "1to1_chat_image.png")
        save_json(result, "1to1_chat_response.json")

        return result
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None


# ============================================================================
# 4. SHORTS GENERATION - Short-form content
# ============================================================================


def example_shorts(alice_features: dict):
    """
    Generate shorts/stories style images.

    - NSFW moderated (adds clothing/occlusions)
    - Emotion-focused captions
    - No chat history needed
    """
    print("\n" + "=" * 70)
    print("4. SHORTS GENERATION - Short-form Content")
    print("=" * 70)

    request_data = {
        "style": "anime",
        "participants": [
            {
                "handle": "alice",
                "avatar_features": alice_features,  # Full features from profile creation
            }
        ],
        "user_message": "A confident woman striking a powerful pose in an elegant studio with dramatic lighting",
    }

    print("📤 Generating shorts image...")
    print(f"   Style: {request_data['style']}")
    print(f"   Message: {request_data['user_message']}")

    response = requests.post(
        f"{BASE_URL}/v1/chat/shorts/generate", json=request_data, timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Image generated successfully!")
        print(f"   Caption: {result['caption']}")
        print(f"   Prompt used: {result['prompt_used']}")

        # Save image and metadata
        save_image(result["image_base64"], "shorts_image.png")
        save_json(result, "shorts_response.json")

        return result
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None


# ============================================================================
# 5. SCENES GENERATION - Multi-person scenes
# ============================================================================


def example_scenes(alice_features: dict, bob_features: dict):
    """
    Generate scene-based images with multiple people.

    - NSFW moderated (adds clothing/occlusions)
    - Emotion-focused captions
    - Identical to shorts but separate GPU tracking
    """
    print("\n" + "=" * 70)
    print("5. SCENES GENERATION - Multi-Person Scene")
    print("=" * 70)

    request_data = {
        "style": "3d_cartoon",
        "participants": [
            {"handle": "alice", "avatar_features": alice_features},
            {"handle": "bob", "avatar_features": bob_features},
        ],
        "user_message": "Two friends celebrating a big win at a vibrant sports bar, cheering with drinks raised",
    }

    print("📤 Generating scene image...")
    print(f"   Style: {request_data['style']}")
    print(f"   Participants: {len(request_data['participants'])}")
    print(f"   Message: {request_data['user_message']}")

    response = requests.post(
        f"{BASE_URL}/v1/chat/scenes/generate", json=request_data, timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Image generated successfully!")
        print(f"   Caption: {result['caption']}")
        print(f"   Prompt used: {result['prompt_used']}")

        # Save image and metadata
        save_image(result["image_base64"], "scenes_image.png")
        save_json(result, "scenes_response.json")

        return result
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None


# ============================================================================
# MAIN - Run all examples
# ============================================================================


def main():
    """Run all API examples in sequence."""
    print("\n" + "🚀 " + "=" * 66)
    print("🚀  API EXAMPLES - Image Generation Endpoints")
    print("🚀 " + "=" * 66)
    print(f"   Base URL: {BASE_URL}")
    print(f"   Output Dir: {OUTPUT_DIR}")

    # Selfie paths - UPDATE THESE WITH YOUR ACTUAL IMAGE PATHS
    alice_selfie = "D:/Freelance/SUKO/Selfie_imgs/img_pod/testing/selfies/selfie_2.jpeg"  # Female selfie
    bob_selfie = "D:/Freelance/SUKO/Selfie_imgs/img_pod/testing/selfies/selfie_2.jpg"  # Male selfie
    charlie_selfie = "D:/Freelance/SUKO/Selfie_imgs/img_pod/testing/selfies/selfie_4.jpeg"  # Another male selfie

    # 1. Create profiles for all participants
    print("\n" + "=" * 70)
    print("STEP 1: Creating Profiles for All Participants")
    print("=" * 70)

    alice_features = example_profile_create("Alice", alice_selfie)
    if not alice_features:
        print("\n⚠️  Skipping remaining examples - Alice's profile creation failed")
        return

    bob_features = example_profile_create("Bob", bob_selfie)
    if not bob_features:
        print("\n⚠️  Skipping remaining examples - Bob's profile creation failed")
        return

    charlie_features = example_profile_create("Charlie", charlie_selfie)
    if not charlie_features:
        print("\n⚠️  Skipping remaining examples - Charlie's profile creation failed")
        return

    # 2. Update profile
    example_profile_update()

    # 3. 1:1 Chat image (Charlie tagged but not in chat - LLM decides if he appears)
    example_1to1_chat(alice_features, bob_features, charlie_features)

    # 4. Shorts image
    example_shorts(alice_features)

    # 5. Scenes image
    example_scenes(alice_features, bob_features)

    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED!")
    print(f"   Check outputs in: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
