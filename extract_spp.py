#!/usr/bin/env python3
"""
File .spp sprite sheet extractor.
All directory record types are image resources; dimensions come from the
chunk size (8x16, 16x16, 32x32, 64x32, 64x64; 4bpp or 8bpp).
"""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from pathlib import Path

from PIL import Image

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
HEADER_SIZE = 0x90
DIR_OFFSET = 0x28
DIR_END_OFFSET = 0x20
PALETTE_ENTRIES = 256
PALETTE_BYTES = PALETTE_ENTRIES * 2
TILE_SIZE = 8


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


# Sprite dimensions come from the record TYPE byte (high byte of the rid):
#   type = (height_class << 4) | width_class,  class: 0=8, 1=16, 2=32, 3=64
# e.g. 0x10=8x16, 0x11=16x16, 0x12=32x16, 0x21=16x32, 0x22=32x32,
#      0x23=64x32, 0x32=32x64, 0x33=64x64.
# The chunk size then gives the bit depth: chunk == w*h -> 8bpp,
# chunk == w*h/2 -> 4bpp. All user-verified 2026-08-20.


def sprite_dims_for_type(rid: int) -> tuple[int, int]:
    t = rid >> 24
    return (8 << (t & 0x0F), 8 << (t >> 4))


def group_palette_needs(
    recs: list[tuple[int, int]], sprite_base: int, palette_offset: int
) -> list[tuple[int, int]] | None:
    """Reconstruct palette blocks from each group's color need.

    recs: (rid, chunk_len) for every non-table directory record.
    A group whose records are 4bpp needs a 16-color block; 8bpp needs
    256.  Type 0x10 with a 0x80 chunk is ambiguous (16x16 4bpp or 8x16
    8bpp) - the ambiguity is resolved by brute-force over the sum check:
    the needs must total exactly (sprite_base - palette_offset)/2 colors.
    Returns [(offset, num_colors), ...] in group first-appearance order,
    or None when no assignment fits."""
    region_colors = (sprite_base - palette_offset) // 2
    order: list[int] = []
    need: dict[int, int | None] = {}
    ambig: list[int] = []
    for rid, clen in recs:
        g = (rid >> 16) & 0xFF
        if g not in need:
            need[g] = None
            order.append(g)
        t = rid >> 24
        bw = 8 << (t & 0x0F)
        bh = 8 << (t >> 4)
        if t == 0x10 and clen == 0x80:
            if g not in ambig:
                ambig.append(g)
        elif clen * 2 == bw * bh:
            if need[g] not in (None, 16):
                return None
            need[g] = 16
        elif clen == bw * bh:
            if need[g] not in (None, 256):
                return None
            need[g] = 256

    for flip in range(1 << len(ambig)):
        for i, g in enumerate(ambig):
            need[g] = 256 if (flip >> i) & 1 else 16
        total = sum(need[g] if need[g] is not None else 16 for g in order)
        if total != region_colors:
            continue
        blocks = []
        off = palette_offset
        for g in order:
            n = need[g] if need[g] is not None else 16
            blocks.append((off, n))
            off += n * 2
        return blocks
    return None


def find_palette_blocks(
    data: bytes,
    dir_offset: int,
    dir_end: int,
    palette_offset: int,
    sprite_base: int,
    n_blocks: int,
) -> list[tuple[int, int]] | None:
    """Return [(offset, num_colors), ...] for each palette block.

    Block offsets are first taken from the trailing palette-directory table
    (8-byte records with resource id 0x00000008 whose rel value is a byte
    offset from palette_offset).  If that does not match the header's block
    count, fall back to 0x7C1F markers between palette_offset and sprite_base
    (0x7C1F can legitimately appear inside a block, so only trust an exact
    marker-count match).  Returns None when neither method matches n_blocks."""
    offsets = []
    p = dir_offset
    while p + 8 <= dir_end and p + 8 <= len(data):
        rid = u32(data, p)
        if rid in (0x00000008, 0x00000004):
            offsets.append(u32(data, p + 4))
        p += 8
    if len(offsets) == n_blocks:
        abs_offs = [palette_offset + o for o in offsets]
        blocks = []
        for i, a in enumerate(abs_offs):
            end = abs_offs[i + 1] if i + 1 < len(abs_offs) else sprite_base
            blocks.append((a, (end - a) // 2))
        return blocks

    markers = [palette_offset]
    off = palette_offset + 2
    while off + 2 <= sprite_base:
        if struct.unpack_from("<H", data, off)[0] == 0x7C1F:
            markers.append(off)
        off += 2
    if len(markers) == n_blocks:
        blocks = []
        for i, m in enumerate(markers):
            end = markers[i + 1] if i + 1 < len(markers) else sprite_base
            blocks.append((m, (end - m) // 2))
        return blocks
    return None


def read_palette(data: bytes, offset: int, ncolors: int) -> list[tuple[int, int, int]]:
    return [
        ds_color_to_rgb(struct.unpack_from("<H", data, offset + i * 2)[0])
        for i in range(ncolors)
    ]


def parse_spp(data: bytes) -> tuple[list[dict], list[tuple[int, int, int]], dict]:
    """
    Parse an SPP file.

    Returns:
        sprites: list of dicts, each with keys
            - id: resource ID (int)
            - offset: offset from sprite data base
            - absolute_offset: absolute file offset
            - chunk: raw sprite data bytes
            - width: decoded width in pixels
            - height: decoded height in pixels
        palette: list of (r,g,b) tuples, 256 entries
        info: dict with version, palette_offset, sprite_base, frame_count
    """
    if len(data) < HEADER_SIZE:
        raise SPPError(
            f"File is only {len(data)} bytes; too small for SPP header (need {HEADER_SIZE})."
        )

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

    # Directory end: value at 0x20, or conservative fallback.
    dir_end = u32(data, DIR_END_OFFSET) if len(data) >= DIR_END_OFFSET + 4 else 0
    if dir_end <= dir_offset or dir_end > len(data):
        dir_end = min(len(data), dir_offset + 64)

    # Read directory records (8 bytes each). Exclude the trailing
    # palette-directory records (rid 0x00000008/0x00000004 and their size
    # companions 0x00000200/0x00000020): their rel values are palette-relative
    # offsets, not sprite-data offsets, and would corrupt chunk boundaries.
    # Every other record stays in this list EVEN IF it is not a sprite
    # (rids below 0x01000000 are palette/metadata rows, e.g. game_common's
    # 0x00000021-0x00000029): their rel values still mark where the next
    # sprite's data ends.
    PALETTE_TABLE_RIDS = {0x00000008, 0x00000004, 0x00000200, 0x00000020}
    records = []
    p = dir_offset
    while p + 8 <= dir_end and p + 8 <= len(data):
        rid = u32(data, p)
        rel_off = u32(data, p + 4)
        if rid not in PALETTE_TABLE_RIDS:
            records.append((rid, rel_off))
        p += 8

    # ALL record types hold image data (verified 2026-08-20): the type byte
    # (0x10/0x11/0x12/0x21/0x22/0x23/0x32/0x33) varies per file and is
    # NOT a graphics-vs-metadata marker. Rids below 0x01000000 are palette/
    # metadata rows (never sprites) but remain in `records` for boundaries
    # AND for the group->block ordering (their groups occupy block slots,
    # e.g. game_common's 0x00000021-0x00000029 rows).
    image_records = [(rid, off) for rid, off in records if rid >= 0x01000000]

    if not image_records:
        raise SPPError("No sprite graphics records were found.")

    image_records.sort(key=lambda x: x[1])

    # Chunk length of every record (end = next record of ANY type).
    rec_chunks: list[tuple[int, int]] = []
    for i, (rid, rel) in enumerate(records):
        nxt = records[i + 1][1] if i + 1 < len(records) else len(data) - sprite_base
        rec_chunks.append((rid, nxt - rel))

    # Read fallback palette (256 colors from palette_offset).
    palette = read_palette(data, palette_offset, PALETTE_ENTRIES)

    # Detect palette blocks. Only trusted when the 0x7C1F marker count matches
    # the header's block count (0x10); otherwise fall back to `palette` above.
    n_blocks = u32(data, 0x10)
    blocks = find_palette_blocks(
        data, dir_offset, dir_end, palette_offset, sprite_base, n_blocks
    )
    if blocks is None:
        blocks = group_palette_needs(rec_chunks, sprite_base, palette_offset)

    # Palette-block rule (refined 2026-08-21): a record's block index is the
    # FIRST-APPEARANCE ORDER of its group byte (rid bits 16-23) among ALL
    # directory records (metadata groups occupy block slots too), not
    # literally grp - 0x10. Verified exact fits: menu_game_retry_swg 9/9,
    # menu_lobby_cap_swg 8/8, menu_item_release_swg 6/6, demo_game_over_swg
    # 5/5, game_common 16/16, menu_result_common 10/10.
    grp_order: dict[int, int] = {}
    for rid, _ in records:
        g = (rid >> 16) & 0xFF
        if g not in grp_order:
            grp_order[g] = len(grp_order)

    def select_palette(rid: int) -> list[tuple[int, int, int]]:
        if blocks is None:
            return palette
        idx = grp_order.get((rid >> 16) & 0xFF, 0)
        idx = min(idx, len(blocks) - 1)
        off, ncolors = blocks[idx]
        return read_palette(data, off, ncolors)

    # Extract each sprite chunk; determine dimensions from chunk size.
    # Note: for 0x200 (512-byte) chunks in some files (e.g. demo_planet_street003),
    # each chunk contains TWO 16x16 sprites (halves of 256 bytes each).
    sprites: list[dict] = []
    for i, (rid, rel_start) in enumerate(image_records):
        # The chunk ends at the next record of ANY type (images are sometimes
        # interleaved with 0x23/0x33 map/animation records), so the end must
        # come from the full record list, not the next image record.
        following_offsets = [off for rrid, off in records if off > rel_start]
        rel_end = min(following_offsets) if following_offsets else None

        abs_start = sprite_base + rel_start
        abs_end = (sprite_base + rel_end) if rel_end is not None else None

        if abs_start >= len(data):
            raise SPPError(f"Resource 0x{rid:08X} starts past the end of the file.")

        if abs_end is None:
            abs_end = len(data)

        chunk = data[abs_start:abs_end]
        chunk_size = len(chunk)

        # Dimensions from the record type byte; bpp from the chunk size.
        # Type 0x10 with a 0x80 chunk is ambiguous (16x16 4bpp or 8x16
        # 8bpp): the group's palette block decides (16 colors -> 4bpp).
        # Records matching neither w*h (8bpp) nor w*h/2 (4bpp) are skipped
        # (padded/oversized) instead of failing the whole file.
        t = rid >> 24
        fw, fh = sprite_dims_for_type(rid)
        if t == 0x10 and chunk_size == 0x80:
            blk = None
            if blocks is not None:
                bidx = min(
                    grp_order.get((rid >> 16) & 0xFF, 0), len(blocks) - 1
                )
                blk = blocks[bidx]
            if blk is not None and blk[1] <= 16:
                width, height, is_4bpp = 16, 16, True
            else:
                width, height, is_4bpp = 8, 16, False
        elif chunk_size * 2 == fw * fh:
            width, height, is_4bpp = fw, fh, True
        elif chunk_size == fw * fh:
            width, height, is_4bpp = fw, fh, False
        else:
            print(
                f"WARN  resource 0x{rid:08X}: chunk 0x{chunk_size:X} bytes "
                f"does not match {fw}x{fh} ({fw * fh} px); skipped.",
                file=sys.stderr,
            )
            continue
        sprite_palette = select_palette(rid)

        # Type 0x10 stores its tiles column-major (vertical 8px strips):
        # a 16x16 frame is [left 8x16 strip][right 8x16 strip]. Row-major
        # placement shows the top half on the left and the bottom half on
        # the right (user-verified 2026-08-21). Harmless for 8x16 frames
        # (single tile column).
        column_major = t == 0x10

        # 0x200 (512-byte) chunks are 32x32 sprites at 4bpp (same as 0x400 but half the size)
        # No splitting needed - each chunk is one 32x32 sprite
        sprites.append(
            {
                "id": rid,
                "offset": rel_start,
                "absolute_offset": abs_start,
                "chunk": chunk,
                "width": width,
                "height": height,
                "palette": sprite_palette,
                "column_major": column_major,
            }
        )

    info = {
        "version": version,
        "palette_offset": palette_offset,
        "sprite_base": sprite_base,
        "frame_count": len(sprites),
    }
    return sprites, palette, info


def decode_frame(sprite: dict, palette: list[tuple[int, int, int]]) -> Image.Image:
    """Decode a single sprite frame from raw tile data."""
    palette = sprite.get("palette", palette)
    raw = sprite["chunk"]
    width = sprite["width"]
    height = sprite["height"]
    chunk_size = len(raw)

    # 8bpp: chunk_size == width * height
    # 4bpp: chunk_size == (width * height) // 2
    is_4bpp = chunk_size * 2 == width * height

    total_pixels = width * height
    pixels = bytearray(total_pixels)

    tiles_x = width // TILE_SIZE
    tiles_y = height // TILE_SIZE
    total_tiles = tiles_x * tiles_y

    if is_4bpp:
        # 4bpp tiled (Nintendo DS style): each 8x8 tile is 32 bytes,
        # 8 rows x 4 bytes, 2 pixels per byte (low nibble = even column).
        tile_bytes = (TILE_SIZE * TILE_SIZE) // 2
        for tile_index in range(total_tiles):
            if sprite.get("column_major"):
                tcol = tile_index // tiles_y
                trow = tile_index % tiles_y
            else:
                tcol = tile_index % tiles_x
                trow = tile_index // tiles_x
            tbase = tile_index * tile_bytes
            for row in range(TILE_SIZE):
                for col in range(TILE_SIZE):
                    byte_val = raw[tbase + row * (TILE_SIZE // 2) + col // 2]
                    if col % 2 == 0:
                        palette_index = byte_val & 0xF
                    else:
                        palette_index = (byte_val >> 4) & 0xF
                    dst = (trow * TILE_SIZE + row) * width + tcol * TILE_SIZE + col
                    pixels[dst] = palette_index
    else:
        # 8bpp tiled: each 8x8 tile is 64 bytes, row-major tile grid.
        for tile_index in range(total_tiles):
            if sprite.get("column_major"):
                tcol = tile_index // tiles_y
                trow = tile_index % tiles_y
            else:
                tcol = tile_index % tiles_x
                trow = tile_index // tiles_x
            tbase = tile_index * TILE_SIZE * TILE_SIZE
            for row in range(TILE_SIZE):
                src = tbase + row * TILE_SIZE
                dst = (trow * TILE_SIZE + row) * width + tcol * TILE_SIZE
                pixels[dst : dst + TILE_SIZE] = raw[src : src + TILE_SIZE]

    rgb = bytearray(total_pixels * 4)
    npal = len(palette)
    for i, palette_index in enumerate(pixels):
        if palette_index >= npal:
            palette_index %= npal
        r, g, b = palette[palette_index]
        a = 0 if palette_index == 0 else 255
        j = i * 4
        rgb[j : j + 4] = bytes((r, g, b, a))

    return Image.frombytes("RGBA", (width, height), bytes(rgb))


def find_input_files(path: Path, recursive: bool) -> list[Path]:
    """Return list of .spp files found at path (file or directory)."""
    if path.is_file():
        if path.suffix.lower() == ".spp":
            return [path]
        else:
            return []
    if not path.is_dir():
        raise FileNotFoundError(path)

    patterns = ["**/*.spp"] if recursive else ["*.spp"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(path.glob(pattern))
    return sorted(set(found))


def save_frame(frame: Image.Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame.save(dst, format="PNG", optimize=False)


SHEET_MAX_WIDTH = 1024


def save_sheet(frames: list[Image.Image], dst: Path, scale: int = 1) -> None:
    if not frames:
        return

    scale = min(max(scale, 1), 4)
    gap = 4

    # Files can mix frame sizes (e.g. 64x32 banners + 8x16 icons); use the
    # largest frame as the cell size and center smaller frames in their cell.
    cell_w = max(f.width for f in frames)
    cell_h = max(f.height for f in frames)
    sw, sh = cell_w * scale, cell_h * scale

    cols = max(1, (SHEET_MAX_WIDTH + gap) // (sw + gap))
    cols = min(cols, len(frames))
    rows = (len(frames) + cols - 1) // cols

    sheet = Image.new(
        "RGBA",
        (
            cols * sw + (cols - 1) * gap,
            rows * sh + (rows - 1) * gap,
        ),
        (0, 0, 0, 0),
    )

    for i, frame in enumerate(frames):
        cx, cy = i % cols, i // cols
        fw, fh = frame.width * scale, frame.height * scale
        x = cx * (sw + gap) + (sw - fw) // 2
        y = cy * (sh + gap) + (sh - fh) // 2
        sheet.alpha_composite(
            frame.resize((fw, fh), Image.Resampling.NEAREST),
            (x, y),
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, format="PNG")


def choose_output_dir(src: Path, output_root: Path | None, script_dir: Path | None = None) -> Path:
    """Return a base output directory.
    If no root given, use a folder named 'spp_output' alongside the script."""
    if output_root is None:
        if script_dir is None:
            script_dir = Path(__file__).resolve().parent
        return script_dir / "spp_output"
    return output_root


def extract_one(src: Path, output_dir: Path, make_sheet: bool, overwrite: bool, scale: int = 1) -> tuple[int, list[Path]]:
    """Extract a single .spp file into output_dir/<src.stem>/.

    Returns (frames_written, png_paths).
    """
    data = src.read_bytes()
    sprites, palette, info = parse_spp(data)

    # Create per-file subfolder: output_dir/<src.stem>/
    out_subdir = output_dir / src.stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    png_paths: list[Path] = []
    written = 0

    for i, sprite in enumerate(sprites, start=1):
        frame = decode_frame(sprite, palette)
        frames.append(frame)

        dst = out_subdir / f"{src.stem}_frame{i:02d}.png"
        if dst.exists() and not overwrite:
            print(f"SKIP  {src.name} -> {dst.name} (already exists)")
            continue

        save_frame(frame, dst)
        print(f"OK    {src.name} -> {dst.name}  {sprite['width']}x{sprite['height']} px")
        written += 1
        png_paths.append(dst)

    if make_sheet:
        sheet_path = out_subdir / f"{src.stem}_sheet.png"
        if sheet_path.exists() and not overwrite:
            print(f"SKIP  {src.name} -> {sheet_path.name} (already exists)")
        else:
            save_sheet(frames, sheet_path, scale)
            print(f"OK    {src.name} -> {sheet_path.name}  {len(frames)} frames")
            png_paths.append(sheet_path)

    return written, png_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract KMSP .spp sprite resources to PNG (variable sizes)."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="One or more .spp files or folders containing .spp files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output folder. If not given, uses <script_dir>/spp_output/.",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Only rebuild PNGs; do not write uncompressed metadata (no-op for SPP).",
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
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Upscale sheet frames by 1-4x (default 1).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for .spp files recursively within folders.",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent

    # Collect all .spp file paths.
    spp_files: list[Path] = []
    for inp in args.inputs:
        try:
            files = find_input_files(inp, args.recursive)
            spp_files.extend(files)
        except FileNotFoundError:
            print(f"ERROR: input does not exist: {inp}", file=sys.stderr)
            return 1

    # Deduplicate and sort.
    spp_files = sorted(set(spp_files))

    if not spp_files:
        print(f"No .spp files found.")
        return 0

    output_root = args.output.resolve() if args.output else None

    ok = 0
    failed = 0
    all_png_paths: list[Path] = []

    for src in spp_files:
        try:
            out_dir = choose_output_dir(src, output_root, script_dir)
            written, png_paths = extract_one(
                src, out_dir, make_sheet=args.sheet, overwrite=args.overwrite, scale=args.scale
            )
            ok += 1
            all_png_paths.extend(png_paths)
        except Exception as exc:
            print(f"FAIL  {src}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. Converted: {ok}, failed: {failed}")

    # If no output folder was specified, print the default output folder.
    if not args.output and ok > 0:
        print(f"Output written to: {script_dir / 'spp_output'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
