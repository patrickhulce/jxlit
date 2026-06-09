"""Generate a perceptually-uniform color test card PNG.

512x512 image, 64x64 grid of 8x8 solid-color blocks (4096 colors).
- Middle row: a perceptually-uniform rainbow at maximum saturation, hue 0..360.
- Going up: lighter and progressively desaturated (toward white).
- Going down: darker and progressively desaturated (toward black).

Colors are built in OkHSL (Bjorn Ottosson's perceptual HSL based on OkLab).
The key property: saturation 1.0 means "as colorful as this hue can get in
sRGB" -- the max chroma is normalized PER HUE against the gamut boundary. That
avoids the brightness/chroma cliff you get when you clamp a single absolute
chroma (blue's sRGB gamut is far smaller than yellow's). Hue and lightness
steps are perceptually even because OkLab is perceptually uniform.

Also encodes a lossy JXL alongside the PNG, tuned for the fastest possible
decode (lowest effort, --faster_decoding=4) while staying close to lossless
(distance 0.5).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from coloraide import Color as _Color
from coloraide.spaces.okhsl import Okhsl


class Color(_Color):
    """coloraide Color with the OkHSL space registered (it's a plugin)."""


Color.register(Okhsl())

GRID = 64            # blocks per side
BLOCK = 8            # pixels per block
SIZE = GRID * BLOCK  # 512

# Vertical axis is OkHSL lightness: top row pure white, bottom row pure black.
L_TOP, L_BOT = 1.0, 0.0
# Saturation peaks at the middle rainbow row and eases toward a floor at the
# edges (not all the way to gray) so color is retained top-to-bottom.
S_MID, S_EDGE = 1.0, 0.45
S_EASE = 0.7  # <1 holds saturation high longer before falling off

# Output next to the repo root, regardless of where the script is run from.
OUT_DIR = Path(__file__).resolve().parent.parent
PNG_PATH = OUT_DIR / "test_card.png"
JXL_PATH = OUT_DIR / "test_card.jxl"

# Lossy JXL tuned for fastest decode while staying close to lossless.
JXL_DISTANCE = 0.5        # Butteraugli distance; 0 = lossless, 0.5 = near-lossless
JXL_EFFORT = 1            # lowest encoder effort (fastest encode)
JXL_FASTER_DECODING = 4   # 0..4, higher = faster decode (drops loop filters etc.)


def encode_jxl():
    cjxl = shutil.which("cjxl")
    if cjxl is None:
        print("cjxl not found on PATH; skipping JXL encode")
        return
    cmd = [
        cjxl,
        str(PNG_PATH),
        str(JXL_PATH),
        "-d", str(JXL_DISTANCE),
        "-e", str(JXL_EFFORT),
        f"--faster_decoding={JXL_FASTER_DECODING}",
        "--quiet",
    ]
    subprocess.run(cmd, check=True)
    kib = JXL_PATH.stat().st_size / 1024
    print(f"wrote {JXL_PATH.name} (d={JXL_DISTANCE}, e={JXL_EFFORT}, "
          f"faster_decoding={JXL_FASTER_DECODING}, {kib:.1f} KiB)")


def main():
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    mid = (GRID - 1) / 2.0  # middle of 64 rows sits between index 31 and 32

    for r in range(GRID):
        frac = r / (GRID - 1)                 # 0 top .. 1 bottom
        L = L_TOP + (L_BOT - L_TOP) * frac
        dist = abs(r - mid) / mid             # 0 at middle .. 1 at top/bottom edge
        S = S_EDGE + (S_MID - S_EDGE) * (1.0 - dist) ** S_EASE
        for c in range(GRID):
            h = (c / GRID) * 360.0            # hue sweep 0..360
            srgb = Color("okhsl", [h, S, L]).convert("srgb")
            srgb.fit("srgb")                  # nudge into gamut if needed
            rgb = tuple(round(min(max(v, 0.0), 1.0) * 255) for v in srgb[:-1])
            y, x = r * BLOCK, c * BLOCK
            img[y:y + BLOCK, x:x + BLOCK] = rgb

    Image.fromarray(img, "RGB").save(PNG_PATH)
    print(f"wrote {PNG_PATH.name} ({SIZE}x{SIZE}, {GRID*GRID} blocks)")
    encode_jxl()


if __name__ == "__main__":
    main()
