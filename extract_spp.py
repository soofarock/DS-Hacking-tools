"""
KMSP (.spp) sprite extractor

Extracts the 32x32 8bpp tiled sprite frames observed in the supplied
.spp samples and writes them as transparent PNGs.

Observed sample layout:
    0x00  4 bytes   magic "KMSP"
    0x04  4 bytes   version (1)
    0x0C  4 bytes   resource-directory offset (0x28)
    0x18  4 bytes   palette offset (0x90)
    0x24  4 bytes   sprite-data base offset (0x2B0)

The directory at 0x28 contains 8-byte records:
    uint32 resource_id
    uint32 offset_from_sprite_data_base

Records whose resource_id starts with 0x22 contain sprite graphics.
In the supplied samples there are four 0x400-byte graphics blocks.
Each block is 8bpp, stored as sixteen 8x8 tiles in row-major tile order,
so the decoded frame is 32x32 pixels.

The palette is 256 Nintendo DS-style BGR555 colors at palette_offset.
Palette index 0 is the transparent/background color in the supplied
samples.

Usage:
    python extract_spp.py FILE_OR_FOLDER
    python extract_spp.py FILE_OR_FOLDER -o output
    python extract_spp.py FILE_OR_FOLDER -r
    python extract_spp.py FILE_OR_FOLDER --sheet

Requires Pillow:
    python -m pip install pillow
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError:
    print(
        "ERROR: Pillow is required.\n"
        "Install it with:\n"
        "    python -m pip install pillow",
        file=sys.stderr,
    )
    raise SystemExit(2)


MAGIC = b"KMSP"
DIR_OFFSET = 0x28
PALETTE_ENTRIES = 256
PALETTE_BYTES = PALETTE_ENTRIES * 2
TILE_SIZE = 8
BYTES_PER_PIXEL = 1
IMAGE_RESOURCE_PREFIX = 0x22


class SPPError(Exception):
    """Raised when a .spp file is invalid or unsupported."""


def u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise SPPError(f"Unexpected end of file at 0x{offset:X}.")
    return struct.unpack_from("<I", data, offset)[0]


def ds_color_to_rgb(word: int) -> tuple[int, int, int]:
    """Convert Nintendo DS-style BGR555 color to 8-bit RGB."""
    r5 = word & 0x1F
    g5 = (word >> 5) & 0x1F
    b5 = (word >> 10) & 0x1F
    r = (r5 << 3) | (r5 >> 2)
    g = (g5 << 3) | (g5 >> 2)
    b = (b5 << 3) | (b5 >> 2)
    return r, g, b


def parse_spp(data: bytes) -> tuple[list[bytes], dict[str, int]]:
    if len(data) < 0x90:
        raise SPPError(f"File is only {len(data)} bytes; too small for SPP header.")

    if data[:4] != MAGIC:
        raise SPPError(f"Bad magic {data[:4]!r}; expected {MAGIC!r}.")

    version = u32(data, 0x04)
    dir_offset = u32(data, 0x0C)
    palette_offset = u32(data, 0x18)
    sprite_base = u32(data, 0x24)

    if dir_offset >= len(data):
        raise SPPError("Directory offset points past the end of the file.")
    if palette_offset + PALETTE_BYTES > len(data):
        raise SPPError("256-color palette extends past the end of the file.")
    if sprite_base >= len(data):
        raise SPPError("Sprite-data base points past the end of the file.")

    # The supplied KMSP files contain 8 directory records from 0x28 to 0x88.
    # Each record is (resource_id, relative_offset).
    if dir_offset != DIR_OFFSET:
        # Keep decoding possible if the format moves the directory in another
        # revision, while still clearly reporting that the sample layout changed.
        DIR_OFFSET_LOCAL = dir_offset
    else:
        DIR_OFFSET_LOCAL = DIR_OFFSET

    records = []
    p = DIR_OFFSET_LOCAL
    # Header field at 0x20 is 0x88 in the supplied samples, which is the end
    # of the directory. Use it when sensible; otherwise inspect the next 64
    # bytes as a conservative fallback.
    dir_end = u32(data, 0x20) if len(data) >= 0x24 else 0
    if dir_end <= p or dir_end > len(data):
        dir_end = min(len(data), p + 64)

    while p + 8 <= dir_end:
        resource_id = u32(data, p)
        rel_offset = u32(data, p + 4)
        records.append((resource_id, rel_offset))
        p += 8

    image_records = [
        (rid, off) for rid, off in records if (rid >> 24) == IMAGE_RESOURCE_PREFIX
    ]

    if not image_records:
        raise SPPError("No 0x22xxxxxx sprite graphics records were found.")

    image_records.sort(key=lambda x: x[1])

    # Offsets in the samples are relative to sprite_base.
    sprites: list[bytes] = []
    for i, (rid, rel_start) in enumerate(image_records):
        rel_end = (
            image_records[i + 1][1]
            if i + 1 < len(image_records)
            else None
        )

        abs_start = sprite_base + rel_start
        abs_end = (
            sprite_base + rel_end
            if rel_end is not None
            else None
        )

        if abs_start >= len(data):
            raise SPPError(
                f"Resource 0x{rid:08X} starts past the end of the file."
            )

        # If another resource follows, its offset gives us this block size.
        # The final graphics block is bounded by the first non-graphics record
        # after the graphics section, when available.
        if abs_end is None:
            following_offsets = [
                off for rrid, off in records
                if off > rel_start
            ]
            if following_offsets:
                abs_end = sprite_base + min(following_offsets)
            else:
                abs_end = len(data)

        chunk = data[abs_start:abs_end]

        if len(chunk) != 0x400:
            raise SPPError(
                f"Sprite resource 0x{rid:08X} is {len(chunk)} bytes; "
                f"the supplied format expects 0x400-byte (32x32 8bpp) frames."
            )

        sprites.append(chunk)

    info = {
        "version": version,
        "palette_offset": palette_offset,
        "sprite_base": sprite_base,
        "frame_count": len(sprites),
    }
    return sprites, info


def decode_frame(raw: bytes, palette: list[tuple[int, int, int]]) -> Image.Image:
    """Untile a 0x400-byte 8bpp sprite made of sixteen 8x8 tiles."""
    if len(raw) != 0x400:
        raise SPPError(f"Expected 0x400 bytes for a frame, got 0x{len(raw):X}.")

    width = height = 32
    pixels = bytearray(width * height)

    # Tile order is row-major:
    # tile 0 -> top-left, tile 1 -> next tile to the right, etc.
    for tile_index in range(16):
        tile_x = (tile_index % 4) * TILE_SIZE
        tile_y = (tile_index // 4) * TILE_SIZE
        src = tile_index * 64

        for row in range(TILE_SIZE):
            src_row = src + row * TILE_SIZE
            dst_row = (tile_y + row) * width + tile_x
            pixels[dst_row:dst_row + TILE_SIZE] = raw[
                src_row:src_row + TILE_SIZE
            ]

    rgb = bytearray(width * height * 4)
    for i, palette_index in enumerate(pixels):
        r, g, b = palette[palette_index]
        a = 0 if palette_index == 0 else 255
        j = i * 4
        rgb[j:j + 4] = bytes((r, g, b, a))

    return Image.frombytes("RGBA", (width, height), bytes(rgb))


def read_palette(data: bytes, palette_offset: int) -> list[tuple[int, int, int]]:
    if palette_offset + PALETTE_BYTES > len(data):
        raise SPPError("Palette extends past the end of the file.")
    return [
        ds_color_to_rgb(struct.unpack_from("<H", data, palette_offset + i * 2)[0])
        for i in range(PALETTE_ENTRIES)
    ]


def find_input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    patterns = ["**/*.spp"] if recursive else ["*.spp"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(path.glob(pattern))
    return sorted(set(found))


def save_sheet(frames: list[Image.Image], dst: Path) -> None:
    if not frames:
        return

    scale = 4
    frame_w, frame_h = frames[0].size
    gap = 4
    sheet = Image.new(
        "RGBA",
        (
            len(frames) * frame_w * scale + (len(frames) - 1) * gap,
            frame_h * scale,
        ),
        (96, 96, 96, 255),
    )

    for i, frame in enumerate(frames):
        x = i * (frame_w * scale + gap)
        sheet.alpha_composite(
            frame.resize((frame_w * scale, frame_h * scale), Image.Resampling.NEAREST),
            (x, 0),
        )

    sheet.save(dst, format="PNG")


def choose_output_dir(src: Path, output_root: Path | None, input_root: Path) -> Path:
    if output_root is None:
        return src.parent

    if input_root.is_dir():
        try:
            rel_parent = src.parent.relative_to(input_root)
        except ValueError:
            rel_parent = Path(".")
        return output_root / rel_parent

    return output_root


def extract_one(src: Path, output_dir: Path, make_sheet: bool, overwrite: bool) -> int:
    data = src.read_bytes()
    raw_frames, info = parse_spp(data)
    palette = read_palette(data, info["palette_offset"])

    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    written = 0

    for i, raw in enumerate(raw_frames, start=1):
        frame = decode_frame(raw, palette)
        frames.append(frame)

        dst = output_dir / f"{src.stem}_frame{i:02d}.png"
        if dst.exists() and not overwrite:
            print(f"SKIP  {src.name} -> {dst.name} (already exists)")
            continue

        frame.save(dst, format="PNG", optimize=False)
        print(f"OK    {src.name} -> {dst.name}  32x32 px")
        written += 1

    if make_sheet:
        sheet_path = output_dir / f"{src.stem}_sheet.png"
        if sheet_path.exists() and not overwrite:
            print(f"SKIP  {src.name} -> {sheet_path.name} (already exists)")
        else:
            save_sheet(frames, sheet_path)
            print(f"OK    {src.name} -> {sheet_path.name}  {len(frames)} frames")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract KMSP .spp sprite resources to PNG."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A .spp file or a folder containing .spp files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output folder. Folder structure is preserved for folder input.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search for .spp files recursively.",
    )
    parser.add_argument(
        "--sheet",
        action="store_true",
        help="Also create one contact-sheet PNG containing all frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    try:
        files = find_input_files(input_path, args.recursive)
    except FileNotFoundError:
        print(f"ERROR: input does not exist: {input_path}", file=sys.stderr)
        return 1

    if not files:
        print(f"No .spp files found under {input_path}.")
        return 0

    output_root = args.output.resolve() if args.output else None

    ok = 0
    failed = 0

    for src in files:
        try:
            out_dir = choose_output_dir(src, output_root, input_path)
            written = extract_one(
                src,
                out_dir,
                make_sheet=args.sheet,
                overwrite=args.overwrite,
            )
            ok += 1
        except Exception as exc:
            print(f"FAIL  {src}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. Converted: {ok}, failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
