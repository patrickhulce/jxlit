# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=2.0",
#   "OpenImageIO>=3.0",
# ]
# ///
"""Convert a float EXR reference to RGB PFM for in-memory test comparison.

PFM is never committed; test suites invoke this at runtime to avoid parsing EXR
in Rust/TypeScript.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio


def read_exr_rgb_f32(path: Path) -> tuple[int, int, np.ndarray]:
    buf = oiio.ImageBuf(str(path))
    if buf.has_error:
        raise RuntimeError(f"failed to read EXR: {buf.geterror()}")
    spec = buf.spec()
    width = spec.width
    height = spec.height
    channels = spec.nchannels
    if channels < 3:
        raise RuntimeError(f"expected >=3 channels, got {channels}")

    pixels = buf.get_pixels(oiio.FLOAT)
    if pixels is None:
        raise RuntimeError("get_pixels returned None")
    arr = np.asarray(pixels, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(height, width, channels)
    rgb = arr[..., :3].copy()
    return height, width, rgb


def write_pfm_le(path: Path | None, rgb: np.ndarray) -> bytes:
    height, width, _ = rgb.shape
    header = f"PF\n{width} {height}\n-1.0\n".encode("ascii")
    body = rgb.astype(np.float32, copy=False).tobytes()
    data = header + body
    if path is not None:
        path.write_bytes(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exr", type=Path, help="input EXR path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output PFM path (default: stdout with --stdout)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="write PFM bytes to stdout",
    )
    args = parser.parse_args()

    _, _, rgb = read_exr_rgb_f32(args.exr)
    data = write_pfm_le(args.output, rgb)
    if args.stdout or args.output is None:
        sys.stdout.buffer.write(data)


if __name__ == "__main__":
    main()
