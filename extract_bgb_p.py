#!/usr/bin/env python3
"""
KMBG (.bgb_p) extractor

This script converts the proprietary KMBG background/graphic resources used 
into PNG files. Supports both compressed (.bgb_p with LZSS10 header) and 
uncompressed (raw KMBG) formats.

Usage:
    python extract_bgb_p.py FILE_OR_FOLDER
    python extract_bgb_p.py FILE_OR_FOLDER -o output
    python extract_bgb_p.py FILE_OR_FOLDER -r

Examples:
    python extract_bgb_p.py "data\\graphic"
    python extract_bgb_p.py "data\\graphic" -o extracted -r

The file format observed is:

Uncompressed KMBG (root-level .bgb_p files):
  0x00  4 bytes   magic "KMBG"
  0x04  4 bytes   version
  0x08  4 bytes   header size
  0x0C  4 bytes   palette size (normally 0x200 = 256 x 16-bit colors)
  0x10  4 bytes   graphics/tile data offset
  0x14  4 bytes   graphics/tile data size
  0x18  4 bytes   map data offset
  0x1C  4 bytes   map data size
  0x20  4 bytes   bits per pixel (observed: 8)
  0x24  4 bytes   map width, in 8x8 tiles
  0x28  4 bytes   map height, in 8x8 tiles

  The palette is Nintendo DS-style 15-bit BGR555/RGB555 color data.
  Graphics are linear 8bpp pixels, grouped into 8x8 tiles (64 bytes each).
  The map contains little-endian 16-bit tile indices.

Compressed .bgb_p (graphic/graphic/bg/ folder):
  0x00  1 byte    LZSS10 marker (0x10)
  0x01  3 bytes   decompressed size (LE u24)
  0x04  ...       LZSS10 bitstream (8 flags per byte, MSB-first):
    - flag=0: literal byte
    - flag=1: back-reference: count = (sh>>12)+3, disp = (sh&0xFFF)+1
  After LZSS10: standard KMBG Nintendo DS 8bpp/4bpp tile graphics
  with BGR555 palette (magic KMBG at start of decompressed data)

Transparency support:
  - Palette index 0 is made transparent in output PNGs
  - Output images have transparency via tRNS chunk
"""

from __future__ import annotations

import argparse
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


HEADER_SIZE = 0x30
MAGIC = b"KMBG"
PALETTE_ENTRIES = 256
PALETTE_BYTES = PALETTE_ENTRIES * 2
TILE_W = 8
TILE_H = 8
BYTES_PER_TILE_8BPP = 64


class KMBGError(Exception):
    """Raised when a .bgb_p file is invalid or unsupported."""


def u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise KMBGError(f"Unexpected end of file at 0x{offset:X}.")
    return struct.unpack_from("<I", data, offset)[0]


def u16(data: bytes, offset: int) -> int:
    if offset + 2 > len(data):
        raise KMBGError(f"Unexpected end of file at 0x{offset:X}.")
    return struct.unpack_from("<H", data, offset)[0]


def ds_color_to_rgb(word: int) -> tuple[int, int, int]:
    """
    Convert Nintendo DS 15-bit color to 8-bit RGB.

    DS color layout:
        bits  0.. 4 = R
        bits  5.. 9 = G
        bits 10..14 = B
        bit  15     = unused/format flag
    """
    r5 = word & 0x1F
    g5 = (word >> 5) & 0x1F
    b5 = (word >> 10) & 0x1F

    # Expand 5-bit -> 8-bit with bit replication.
    r = (r5 << 3) | (r5 >> 2)
    g = (g5 << 3) | (g5 >> 2)
    b = (b5 << 3) | (b5 >> 2)
    return r, g, b


def lzss10_decompress(data: bytes, decompressed_size: int) -> bytes:
    """
    Decompress LZSS10 bitstream.

    Format: 8 flags per byte, MSB-first.
    flag=0: copy 1 literal byte
    flag=1: back-reference using 2-byte short header:
      count = (sh >> 12) + 3
      disp  = (sh & 0xFFF) + 1
    """
    output = bytearray()
    pos = 0
    out_pos = 0

    while out_pos < decompressed_size and pos < len(data):
        if pos >= len(data):
            break
        flags_byte = data[pos]
        pos += 1

        for bit_idx in range(7, -1, -1):
            if out_pos >= decompressed_size:
                break
            flag = (flags_byte >> bit_idx) & 1

            if flag == 0:
                # Literal byte
                if pos >= len(data):
                    break
                output.append(data[pos])
                pos += 1
                out_pos += 1
            else:
                # Back-reference
                if pos + 1 >= len(data):
                    break
                sh = u16(data, pos)
                count = ((sh >> 12) & 0xFFF) + 3
                disp = (sh & 0xFFF) + 1
                pos += 2

                # Copy from current position - disp bytes back
                src_start = out_pos - disp
                # Clamp to 0 if reference would go before start of output
                if src_start < 0:
                    src_start = 0
                # Copy count bytes from src_start
                for _ in range(count):
                    if out_pos >= decompressed_size:
                        break
                    # Only copy if src_start is within output bounds
                    if src_start < len(output):
                        output.append(output[src_start])
                    out_pos += 1
                # Note: if src_start >= len(output), we just increment out_pos
                # without copying (the literal bytes will fill in later)

    if out_pos < decompressed_size:
        raise KMBGError(
            f"LZSS10 decompressed only {out_pos} of expected "
            f"{decompressed_size} bytes."
        )

    return bytes(output)


def detect_format(data: bytes) -> str:
    """
    Detect whether a .bgb_p file is compressed or uncompressed.

    Returns 'compressed' if the LZSS10 marker (0x10) is found,
    'uncompressed' if KMBG magic is at the start.
    """
    if len(data) < 4:
        raise KMBGError("File too small to determine format.")

    if data[:4] == MAGIC:
        return "uncompressed"

    if len(data) >= 4 and data[0] == 0x10:
        return "compressed"

    raise KMBGError(
        f"Unknown format; first 4 bytes: {data[:4]!r}. "
        "Expected 'KMBG' (uncompressed) or 0x10 (compressed LZSS10)."
    )


def parse_kmbg_header(data: bytes) -> dict[str, int]:
    if len(data) < HEADER_SIZE:
        raise KMBGError(f"File is only {len(data)} bytes; header is {HEADER_SIZE} bytes.")

    if data[:4] != MAGIC:
        raise KMBGError(
            f"Bad magic {data[:4]!r}; expected {MAGIC!r}."
        )

    header = {
        "version": u32(data, 0x04),
        "header_size": u32(data, 0x08),
        "palette_size": u32(data, 0x0C),
        "graphics_offset": u32(data, 0x10),
        "graphics_size": u32(data, 0x14),
        "map_offset": u32(data, 0x18),
        "map_size": u32(data, 0x1C),
        "bpp": u32(data, 0x20),
        "width_tiles": u32(data, 0x24),
        "height_tiles": u32(data, 0x28),
    }

    # Sanity checks.
    if header["header_size"] < HEADER_SIZE:
        raise KMBGError(
            f"Unsupported header size 0x{header['header_size']:X}."
        )

    for key in ("graphics_offset", "map_offset"):
        if header[key] > len(data):
            raise KMBGError(f"{key} points past end of file.")

    if header["graphics_offset"] + header["graphics_size"] > len(data):
        raise KMBGError("Graphics section extends past end of file.")

    if header["map_offset"] + header["map_size"] > len(data):
        raise KMBGError("Map section extends past end of file.")

    if header["bpp"] != 8:
        raise KMBGError(
            f"Unsupported BPP {header['bpp']}; this decoder expects 8bpp."
        )

    if header["palette_size"] < PALETTE_BYTES:
        raise KMBGError(
            f"Palette section is only 0x{header['palette_size']:X} bytes; "
            f"expected at least 0x{PALETTE_BYTES:X}."
        )

    if header["width_tiles"] == 0 or header["height_tiles"] == 0:
        raise KMBGError("Image dimensions are zero.")

    if header["width_tiles"] * header["height_tiles"] * 2 != header["map_size"]:
        raise KMBGError(
            "Map size does not match width * height * 16-bit tile indices: "
            f"0x{header['map_size']:X} != "
            f"{header['width_tiles']}*{header['height_tiles']}*2."
        )

    if header["graphics_size"] % BYTES_PER_TILE_8BPP != 0:
        raise KMBGError(
            "Graphics size is not divisible by 64 bytes (one 8x8 tile at 8bpp)."
        )

    return header


def decode_kmbg(data: bytes) -> tuple[Image.Image, dict[str, int]]:
    h = parse_kmbg_header(data)

    # Palette immediately follows the header.
    palette_base = h["header_size"]
    palette_end = palette_base + PALETTE_BYTES
    if palette_end > len(data):
        raise KMBGError("Palette extends past end of file.")

    palette = [
        ds_color_to_rgb(u16(data, palette_base + i * 2))
        for i in range(PALETTE_ENTRIES)
    ]

    graphics = data[
        h["graphics_offset"] : h["graphics_offset"] + h["graphics_size"]
    ]
    map_data = data[
        h["map_offset"] : h["map_offset"] + h["map_size"]
    ]

    tile_count = len(graphics) // BYTES_PER_TILE_8BPP
    width_tiles = h["width_tiles"]
    height_tiles = h["height_tiles"]

    # Create an indexed image first. This avoids accumulating color-rounding
    # differences and preserves the original 0..255 palette indices.
    image = Image.new("P", (width_tiles * TILE_W, height_tiles * TILE_H))
    flat_pixels = bytearray(image.width * image.height)

    for tile_y in range(height_tiles):
        for tile_x in range(width_tiles):
            map_index = tile_y * width_tiles + tile_x
            tile_index = u16(map_data, map_index * 2)

            # Current KMBG files use plain 16-bit tile indices.
            # Reject values that cannot reference the tile section instead
            # of silently producing corrupt images.
            if tile_index >= tile_count:
                raise KMBGError(
                    f"Tile index {tile_index} at map entry {map_index} "
                    f"references tile {tile_index}, but only {tile_count} "
                    f"tiles exist."
                )

            src = tile_index * BYTES_PER_TILE_8BPP
            dst_x = tile_x * TILE_W
            dst_y = tile_y * TILE_H

            for row in range(TILE_H):
                src_row = src + row * TILE_W
                dst_row = (dst_y + row) * image.width + dst_x
                flat_pixels[dst_row : dst_row + TILE_W] = graphics[
                    src_row : src_row + TILE_W
                ]

    image.frombytes(bytes(flat_pixels))

    # Install the original 256-color palette into the PNG.
    pal_bytes = b"".join(bytes(rgb) for rgb in palette)
    image.putpalette(pal_bytes, rawmode="RGB")

    # Add transparency: make palette index 0 transparent
    img_info = {
        "width": image.width,
        "height": image.height,
        "tile_count": tile_count,
        "map_entries": width_tiles * height_tiles,
        "bpp": h["bpp"],
        "graphics_offset": h["graphics_offset"],
        "graphics_size": h["graphics_size"],
        "map_offset": h["map_offset"],
        "map_size": h["map_size"],
    }
    image.info["transparency"] = 0

    return image, img_info


def decompress_and_decode_kmbg(src_path: Path) -> tuple[Image.Image, dict[str, int]]:
    """
    Detect format and decompress if needed, then decode KMBG data.

    Handles both:
    - Compressed .bgb_p: LZSS10 header (0x10) + bitstream + KMBG data
    - Uncompressed .bgb_p: raw KMBG with KMBG magic header
    """
    data = src_path.read_bytes()
    fmt = detect_format(data)

    if fmt == "uncompressed":
        # Raw KMBG - decode directly
        image, info = decode_kmbg(data)
    else:
        # Compressed - need LZSS10 decompression first
        # Parse the 4-byte LZSS10 header: 0x10 + 3-byte LE decompressed size
        if len(data) < 4:
            raise KMBGError("Compressed file too small for LZSS10 header.")

        decompressed_size = struct.unpack_from("<I", data, 1)[0]
        # But it's a 3-byte size, so mask to get just 3 bytes
        # Actually, looking at the data: 10f03300 -> bytes 1-3 are f0, 33, 00
        # As LE u24: 0x0033f0 = 21008... let me check

        # Re-read: the header is 4 bytes total, byte 0 is 0x10, bytes 1-3 are size
        # So we need to read 3 bytes as LE u24
        decompressed_size = 0
        for i in range(3):
            if 1 + i < len(data):
                decompressed_size |= (data[1 + i] << (8 * i))

        # Decompress LZSS10
        compressed_data = data[4:]  # Skip 4-byte header
        decompressed = lzss10_decompress(compressed_data, decompressed_size)

        # Now the decompressed data should start with KMBG magic
        if decompressed[:4] != MAGIC:
            raise KMBGError(
                f"Decompressed data does not start with KMBG magic; "
                f"got {decompressed[:4]!r}."
            )

        # Decode the KMBG data (skip the 4-byte KMBG header since it's now at start)
        # But parse_kmbg_header expects the header to start at offset 0 with KMBG magic
        # So we need to adjust: the decompressed data has KMBG at offset 0,
        # which is exactly what parse_kmbg_header expects
        image, info = decode_kmbg(decompressed)

    return image, info


def find_input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(path)

    pattern = "**/*.bgb_p" if recursive else "*.bgb_p"
    return sorted(path.glob(pattern))


def choose_output_path(src: Path, output_root: Path | None, input_root: Path) -> Path:
    if output_root is None:
        return src.with_suffix(".png")

    if input_root.is_dir():
        try:
            rel = src.relative_to(input_root)
        except ValueError:
            rel = Path(src.name)
        return (output_root / rel).with_suffix(".png")

    return (output_root / src.name).with_suffix(".png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract KMBG .bgb_p resources to PNG."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A .bgb_p file or a folder containing .bgb_p files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG file/folder. For a folder input, subfolders are preserved.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search for .bgb_p files recursively.",
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
        print(f"No .bgb_p files found under {input_path}.")
        return 0

    output_root = args.output.resolve() if args.output else None
    if output_root is not None and len(files) == 1 and input_path.is_file():
        # When one file is supplied and -o looks like a filename, honor it.
        if output_root.suffix.lower() == ".png":
            output_path_for_single = output_root
        else:
            output_path_for_single = output_root / files[0].with_suffix(".png").name
    else:
        output_path_for_single = None

    ok = 0
    failed = 0

    for src in files:
        if output_path_for_single is not None:
            dst = output_path_for_single
        else:
            dst = choose_output_path(src, output_root, input_path)

        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite:
            print(f"SKIP  {src}  ->  {dst}  (already exists)")
            continue

        try:
            image, info = decompress_and_decode_kmbg(src)
            # Save with transparency
            save_kwargs = {"format": "PNG", "optimize": False}
            if "transparency" in image.info:
                # Pillow will add tRNS chunk
                save_kwargs["transparency"] = image.info["transparency"]
            image.save(dst, **save_kwargs)

            print(
                f"OK    {src.name:45s} -> {dst}  "
                f"{info['width']}x{info['height']} px, "
                f"{info['tile_count']} tiles"
            )
            ok += 1
        except Exception as exc:
            print(f"FAIL  {src}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. Converted: {ok}, failed: {failed}, skipped: {len(files) - ok - failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
