from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import httpx


def load_config() -> Dict[str, Any]:
    base = Path(__file__).resolve().parent
    cfg_path = base / "config.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"Missing {cfg_path}. Copy testing/config.example.json to testing/config.json and set interface_base_url."
        )
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def parse_boundary(content_type: str) -> str:
    m = re.search(r"boundary=([^;]+)", content_type, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"No boundary in Content-Type: {content_type}")
    return m.group(1).strip().strip('"')


def parse_multipart_mixed(body: bytes, *, boundary: str) -> Tuple[Dict[str, Any], bytes]:
    boundary_bytes = ("--" + boundary).encode()
    parts = body.split(boundary_bytes)

    json_obj: Dict[str, Any] | None = None
    image_bytes: bytes | None = None

    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue

        # part headers/body split
        header_blob, _, content = part.partition(b"\r\n\r\n")
        headers = header_blob.decode(errors="replace").lower()

        if "content-type: application/json" in headers:
            json_obj = json.loads(content.decode("utf-8"))
        elif "content-type: image/png" in headers:
            image_bytes = content

    if json_obj is None or image_bytes is None:
        raise ValueError("Failed to parse multipart response (missing JSON or PNG part)")
    return json_obj, image_bytes


async def profile_create(interface_base_url: str, selfie_path: Path) -> Tuple[Dict[str, Any], bytes]:
    url = interface_base_url.rstrip("/") + "/v1/profile/create"
    file_bytes = selfie_path.read_bytes()

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, files={"selfie": (selfie_path.name, file_bytes, "image/png")})

    if resp.status_code == 422:
        # BAD_SELFIE pass-through
        raise RuntimeError(f"422 BAD_SELFIE: {resp.text}")
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    boundary = parse_boundary(content_type)
    json_part, image_bytes = parse_multipart_mixed(resp.content, boundary=boundary)
    return json_part, image_bytes


async def main() -> None:
    cfg = load_config()
    interface = cfg["interface_base_url"]

    base = Path(__file__).resolve().parent
    input_dir = base / "input_selfies"
    out_dir = base / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    selfies = [input_dir / "selfie_1.png", input_dir / "selfie_2.png"]

    for selfie in selfies:
        if not selfie.exists():
            raise SystemExit(f"Missing selfie: {selfie}. Run generate_test_selfies.py first.")

        json_part, image_bytes = await profile_create(interface, selfie)

        stem = selfie.stem
        (out_dir / f"{stem}_features.json").write_text(json.dumps(json_part, indent=2), encoding="utf-8")
        (out_dir / f"{stem}_profile.png").write_bytes(image_bytes)

        print(f"OK {stem}: wrote output/{stem}_features.json and output/{stem}_profile.png")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
