"""Generate KodaQuant app icons using PIL ImageDraw.

Produces:
  build/icon.png   - 1024x1024 PNG
  build/icon.ico   - multi-resolution ICO (16,24,32,48,64,96,128,256)
  frontend/public/favicon.ico - copy of ICO for web use

Zero external dependencies beyond Pillow.
Builds proper multi-resolution ICO binary format directly.
"""
from __future__ import annotations

import io
import os
import struct
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
PUBLIC_DIR = PROJECT_ROOT / "frontend" / "public"

CYAN = (0, 229, 255, 255)
DARK = (5, 6, 8, 255)


def _draw_diamond(draw, size, margin, fill):
    cx = size / 2
    points = [
        (cx, margin),
        (size - margin, cx),
        (cx, size - margin),
        (margin, cx),
    ]
    draw.polygon(points, fill=fill)


def render_image(size=512):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    corner = int(size / 8)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=corner, fill=DARK)

    _draw_diamond(draw, size, margin=size / 8, fill=CYAN)
    _draw_diamond(draw, size, margin=size / 4, fill=DARK)

    return img


def _render_to_png_bytes(size):
    img = render_image(size)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def generate_png():
    img = render_image(1024)
    path = BUILD_DIR / "icon.png"
    img.save(path, "PNG")
    print(f"  [OK] {path}  ({img.size})")


def generate_ico():
    sizes = [16, 24, 32, 48, 64, 96, 128, 256]
    png_buffers = []
    for s in sizes:
        png_buf = _render_to_png_bytes(s)
        png_buffers.append(png_buf)

    header_size = 6
    dir_entry_size = 16
    dir_size = dir_entry_size * len(png_buffers)
    data_offset = header_size + dir_size

    parts = []
    parts.append(struct.pack("<HHH", 0, 1, len(png_buffers)))

    current_offset = data_offset
    for i, buf in enumerate(png_buffers):
        s = sizes[i]
        width = 0 if s == 256 else s
        height = 0 if s == 256 else s
        entry = struct.pack(
            "<BBBBHHII",
            width, height, 0, 0,
            1, 32, len(buf), current_offset,
        )
        parts.append(entry)
        current_offset += len(buf)

    for buf in png_buffers:
        parts.append(buf)

    ico_data = b"".join(parts)

    path = BUILD_DIR / "icon.ico"
    with open(path, "wb") as f:
        f.write(ico_data)
    print(f"  [OK] {path}  ({len(ico_data)} bytes, sizes: {sizes})")


def generate_favicon_ico():
    src = BUILD_DIR / "icon.ico"
    dst = PUBLIC_DIR / "favicon.ico"
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)
    print(f"  [OK] {dst}")


def main():
    print("[generate_icons] KodaQuant icon pipeline")
    generate_png()
    generate_ico()
    generate_favicon_ico()
    print("[generate_icons] Done.")


if __name__ == "__main__":
    main()
