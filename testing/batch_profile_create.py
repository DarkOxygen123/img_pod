#!/usr/bin/env python3
"""Batch profile-create tester.

Goal:
- Your friend puts images in a folder.
- This script calls the Interface `/v1/profile/create` for each image.
- It creates an output folder next to the input folder and saves:
  - <stem>_features.json
  - <stem>_profile.png
  - summary.jsonl (one line per input with status + timings if present)

Works with the existing backend response format:
- Interface returns `multipart/mixed` containing JSON + PNG.

Usage examples:
  python testing/batch_profile_create.py \
    --input-dir /path/to/selfies \
    --base-url https://<interface-proxy>.proxy.runpod.net

  # Optional: choose output dir explicitly
  python testing/batch_profile_create.py --input-dir ./my_imgs --output-dir ./my_imgs_output --base-url http://localhost:8000

Notes:
- This script uses only stdlib + httpx (already used in this repo).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Optional, Tuple

import httpx


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ParsedResponse:
    features_json: dict
    image_bytes: bytes


def _default_output_dir(input_dir: Path) -> Path:
    # Output folder next to the input folder (same parent).
    return input_dir.parent / f"{input_dir.name}_output"


def _guess_content_type(path: Path) -> str:
    ct, _ = mimetypes.guess_type(str(path))
    return ct or "application/octet-stream"


def _parse_multipart_mixed(resp: httpx.Response) -> ParsedResponse:
    ct = resp.headers.get("content-type") or resp.headers.get("Content-Type")
    if not ct:
        raise ValueError("Missing Content-Type")
    if "multipart/mixed" not in ct:
        raise ValueError(f"Unexpected Content-Type: {ct}")

    # Parse with stdlib email module.
    # We prepend headers so the parser sees a complete MIME message.
    raw = (
        f"Content-Type: {ct}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).encode("utf-8") + resp.content

    msg = BytesParser(policy=policy.default).parsebytes(raw)
    if not msg.is_multipart():
        raise ValueError("Response is not multipart")

    json_part: Optional[bytes] = None
    png_part: Optional[bytes] = None

    for part in msg.iter_parts():
        part_ct = (part.get_content_type() or "").lower()
        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        if part_ct == "application/json" and json_part is None:
            json_part = payload
        elif part_ct in {"image/png", "application/octet-stream"} and png_part is None:
            # interface uses image/png
            png_part = payload

    if json_part is None:
        raise ValueError("Missing JSON part")
    if png_part is None:
        raise ValueError("Missing image part")

    features = json.loads(json_part.decode("utf-8"))
    # The Interface sends: {"avatar_features": {...}}
    return ParsedResponse(features_json=features, image_bytes=png_part)


def _post_profile_create(
    client: httpx.Client,
    base_url: str,
    image_path: Path,
    timeout_s: float,
) -> Tuple[ParsedResponse, float]:
    url = base_url.rstrip("/") + "/v1/profile/create"
    file_bytes = image_path.read_bytes()

    t0 = time.time()
    resp = client.post(
        url,
        files={
            "file": (
                image_path.name,
                file_bytes,
                _guess_content_type(image_path),
            )
        },
        timeout=timeout_s,
    )
    elapsed = time.time() - t0

    # 422 is a valid structured error (BAD_SELFIE). Save it as json output.
    if resp.status_code == 422:
        try:
            data = resp.json()
        except Exception:
            data = {"message": resp.text}
        raise httpx.HTTPStatusError(
            f"422 BAD_SELFIE for {url}",
            request=resp.request,
            response=resp,
        )

    resp.raise_for_status()
    parsed = _parse_multipart_mixed(resp)
    return parsed, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch test /v1/profile/create using a folder of images")
    ap.add_argument("--input-dir", required=True, help="Folder containing input images")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output folder (defaults to a sibling folder named <input>_output)",
    )
    ap.add_argument(
        "--base-url",
        default=os.getenv("INTERFACE_BASE_URL", ""),
        help="Interface base URL (e.g. https://...proxy.runpod.net or http://localhost:8000)",
    )
    ap.add_argument("--timeout", type=float, default=90.0, help="Per-request timeout in seconds")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = no limit)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip files that already have outputs")

    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    base_url = (args.base_url or "").strip()
    if not base_url:
        raise SystemExit("Missing --base-url (or set INTERFACE_BASE_URL env var)")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    if not images:
        print(f"No images found in {input_dir} (supported: {sorted(IMAGE_EXTS)})")
        return 0

    summary_path = output_dir / "summary.jsonl"

    ok = 0
    failed = 0

    with httpx.Client(follow_redirects=True) as client:
        for idx, img in enumerate(images, start=1):
            stem = img.stem
            out_json = output_dir / f"{stem}_features.json"
            out_png = output_dir / f"{stem}_profile.png"

            if args.skip_existing and out_json.exists() and out_png.exists():
                print(f"[{idx}/{len(images)}] SKIP {img.name} (outputs exist)")
                continue

            print(f"[{idx}/{len(images)}] Processing {img.name}...")

            row = {
                "input": str(img),
                "output_json": str(out_json),
                "output_png": str(out_png),
                "status": "error",
                "http_seconds": None,
                "error": None,
            }

            try:
                parsed, elapsed = _post_profile_create(client, base_url, img, timeout_s=float(args.timeout))
                row["status"] = "ok"
                row["http_seconds"] = round(elapsed, 3)

                out_json.write_text(json.dumps(parsed.features_json, indent=2, ensure_ascii=False))
                out_png.write_bytes(parsed.image_bytes)

                ok += 1
                print(f"    OK in {row['http_seconds']}s -> {out_png.name}")

            except httpx.HTTPStatusError as e:
                failed += 1
                row["error"] = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
                print(f"    FAIL: {row['error']}")

                # Save error payload if any
                try:
                    err_path = output_dir / f"{stem}_error.json"
                    err_path.write_text(json.dumps(e.response.json(), indent=2, ensure_ascii=False))
                except Exception:
                    pass

            except Exception as e:
                failed += 1
                row["error"] = str(e)
                print(f"    FAIL: {row['error']}")

            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone")
    print(f"- Input:  {input_dir}")
    print(f"- Output: {output_dir}")
    print(f"- OK: {ok}")
    print(f"- Failed: {failed}")
    print(f"- Summary: {summary_path}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
