from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def make_selfie(path: Path, *, base_color: tuple[int, int, int], accent: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (768, 1024), base_color)
    draw = ImageDraw.Draw(img)

    # Simple "face-like" shapes just to vary bytes deterministically.
    draw.ellipse((220, 260, 540, 580), fill=(240, 210, 180))  # face
    draw.ellipse((310, 360, 360, 410), fill=accent)  # left eye
    draw.ellipse((400, 360, 450, 410), fill=accent)  # right eye
    draw.arc((330, 430, 430, 520), start=10, end=170, fill=(140, 60, 60), width=6)  # smile

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "input_selfies"
    make_selfie(out_dir / "selfie_1.png", base_color=(120, 180, 220), accent=(30, 30, 30))
    make_selfie(out_dir / "selfie_2.png", base_color=(220, 170, 120), accent=(10, 80, 10))
    print(f"Wrote: {out_dir / 'selfie_1.png'}")
    print(f"Wrote: {out_dir / 'selfie_2.png'}")


if __name__ == "__main__":
    main()
