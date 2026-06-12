# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "coloraide>=4.0",
#   "numpy>=2.0",
#   "pillow>=11.0",
#   "OpenImageIO>=3.0",
#   "opencolorio>=2.4",
# ]
# ///
"""Generate colorspace test assets for jxlit.

512x512 OkHSL color grid (ACEScg gamut fit) -> ACES2065-1 master EXR, then OCIO
variants encoded to JXL with enum color_space hints.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenImageIO as oiio
import PyOpenColorIO as ocio
from coloraide import Color as _Color
from coloraide.spaces.okhsl import Okhsl
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

GRID = 64
BLOCK = 8
SIZE = GRID * BLOCK

L_TOP, L_BOT = 1.0, 0.0
S_MID, S_EDGE = 1.0, 0.45
S_EASE = 0.7

JXL_DISTANCE = 0.5
JXL_EFFORT = 1
JXL_FASTER_DECODING = 4
JXL_SUFFIX = "e1_d0p5_fd4"

EXR_COMPRESSION = "zip"
# Super-highlights only: the brightest HDR_RAMP_BLOCKS rows (8px blocks) ramp up by a
# fixed exposure step, topping out at this scene-linear multiplier; the rest is unchanged.
ACES_HDR_WHITE_PEAK = 10.0
HDR_RAMP_BLOCKS = 8


class Color(_Color):
    """coloraide Color with OkHSL registered."""


Color.register(Okhsl())


@dataclass(frozen=True)
class FixtureSpec:
    slug: str
    ocio_space: str
    encode: dict[str, str]
    mae_tolerance: float
    # HDR fixtures keep scene-linear values >1.0. cjxl only preserves extended-range float
    # for sRGB/709 primaries with a linear transfer; wide-gamut (Rec.2020/P3) or display
    # encodings get clamped to the [0,1] display range, so those fixtures are clamped here.
    hdr: bool = False


FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        "colors-ACES-RGB_D65_SRG_Rel_Lin",
        "Linear Rec.709 (sRGB)",
        {"color_space": "RGB_D65_SRG_Rel_Lin"},
        0.03,
        hdr=True,
    ),
    # ACEScct log code values stored in a (cheap) linear Rec.709 enum container instead of
    # an embedded ACEScct ICC profile, which is ~100x larger than the codestream itself.
    FixtureSpec(
        "colors-ACEScct-RGB_D65_SRG_Rel_Lin",
        "ACEScct",
        {"color_space": "RGB_D65_SRG_Rel_Lin"},
        0.03,
    ),
    FixtureSpec(
        "colors-Rec2020-Rec2100PQ",
        "Rec.2100-PQ - Display",
        {"color_space": "Rec2100PQ"},
        0.04,
    ),
    # P3 primaries + gamma 2.6 (DCI transfer), D65 white. studio-config-latest has no
    # DCI-white P3 display, so this uses its D65-white P3 space (the only difference).
    FixtureSpec(
        "colors-P3-RGB_D65_DCI_Rel_DCI",
        "P3-D65 - Display",
        {"color_space": "RGB_D65_DCI_Rel_DCI"},
        0.03,
    ),
)


def build_okhsl_srgb_grid() -> tuple[np.ndarray, np.ndarray]:
    """Vibrant OkHSL grid as sRGB-display floats plus the per-pixel OkHSL lightness."""
    grid = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    lightness_map = np.zeros((SIZE, SIZE), dtype=np.float32)
    mid = (GRID - 1) / 2.0

    for r in range(GRID):
        frac = r / (GRID - 1)
        lightness = L_TOP + (L_BOT - L_TOP) * frac
        dist = abs(r - mid) / mid
        saturation = S_EDGE + (S_MID - S_EDGE) * (1.0 - dist) ** S_EASE
        for c in range(GRID):
            hue = (c / GRID) * 360.0
            block = Color("okhsl", [hue, saturation, lightness]).convert("srgb")
            block.fit("srgb")
            rgb = [min(max(float(v), 0.0), 1.0) for v in block[:-1]]
            y, x = r * BLOCK, c * BLOCK
            grid[y : y + BLOCK, x : x + BLOCK] = rgb
            lightness_map[y : y + BLOCK, x : x + BLOCK] = lightness

    return grid, lightness_map


def srgb_to_aces2065(srgb: np.ndarray) -> np.ndarray:
    cfg = ocio.Config.CreateFromBuiltinConfig("studio-config-latest")
    processor = cfg.getProcessor(
        "sRGB Encoded Rec.709 (sRGB)", "ACES2065-1"
    ).getDefaultCPUProcessor()
    flat = srgb.reshape(-1, 3).astype(np.float32).copy()
    processor.applyRGB(flat)
    return flat.reshape(srgb.shape)


def escalate_highlights(master: np.ndarray, lightness_map: np.ndarray) -> np.ndarray:
    """Leave the lower lightness range untouched; step the brightest rows up into HDR.

    The ramp is keyed to OkHSL lightness (the grid's vertical axis), so the boundary is a
    clean horizontal gradient. The top HDR_RAMP_BLOCKS rows ramp up by a fixed exposure
    step, topping out at ACES_HDR_WHITE_PEAK at the top edge.
    """
    peak = float(master.max())
    if peak <= 0.0:
        return master.astype(np.float32)

    target_scale = ACES_HDR_WHITE_PEAK / peak
    stops_per_block = float(np.log2(target_scale)) / HDR_RAMP_BLOCKS
    # OkHSL lightness is linear in grid row, so this counts 8px blocks down from the top.
    blocks_from_top = (L_TOP - lightness_map) * (GRID - 1)
    stops = np.clip(HDR_RAMP_BLOCKS - blocks_from_top, 0.0, None) * stops_per_block
    scale = np.exp2(stops)
    return (master * scale[..., np.newaxis]).astype(np.float32)


def build_master_aces2065(
    srgb_grid: np.ndarray, lightness_map: np.ndarray
) -> np.ndarray:
    """Vibrant OkHSL grid in ACES2065-1 with a smooth HDR super-highlight band."""
    master = srgb_to_aces2065(srgb_grid)
    return escalate_highlights(master, lightness_map)


def write_exr_half(path: Path, rgb: np.ndarray, colorspace: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = oiio.ImageSpec(SIZE, SIZE, 3, oiio.HALF)
    spec.attribute("compression", EXR_COMPRESSION)
    spec.attribute("ocio:colorspace", colorspace)
    buf = oiio.ImageBuf(spec)
    half = rgb.astype(np.float16)
    buf.set_pixels(oiio.ROI(0, SIZE, 0, SIZE, 0, 1, 0, 3), half)
    if not buf.write(str(path), oiio.HALF):
        raise RuntimeError(f"failed to write EXR {path}: {buf.geterror()}")


def write_pfm(path: Path, rgb: np.ndarray) -> None:
    """Write a spec-correct PFM (scanlines bottom-to-top, little-endian) for cjxl input.

    cjxl only honours the `-x color_space=`/`icc_pathname=` color hints for raw formats
    like PFM (it reads color metadata directly from EXR/PNG and ignores the hints), so
    fixtures must be encoded from PFM to embed their target color profile.
    """
    header = f"PF\n{SIZE} {SIZE}\n-1.0\n".encode("ascii")
    body = rgb[::-1].astype(np.float32, copy=False).tobytes()
    path.write_bytes(header + body)


def ocio_transform(master: np.ndarray, dst_space: str) -> np.ndarray:
    if dst_space == "ACES2065-1":
        return master.copy()
    cfg = ocio.Config.CreateFromBuiltinConfig("studio-config-latest")
    processor = cfg.getProcessor("ACES2065-1", dst_space)
    cpu = processor.getDefaultCPUProcessor()
    flat = master.reshape(-1, 3).copy()
    cpu.applyRGB(flat)
    return flat.reshape(SIZE, SIZE, 3).astype(np.float32)


def encode_jxl(input_path: Path, jxl_path: Path, encode: dict[str, str]) -> None:
    cjxl = shutil.which("cjxl")
    if cjxl is None:
        raise RuntimeError("cjxl not found on PATH")

    cmd = [
        cjxl,
        str(input_path),
        str(jxl_path),
        "-d",
        str(JXL_DISTANCE),
        "-e",
        str(JXL_EFFORT),
        f"--faster_decoding={JXL_FASTER_DECODING}",
        "--quiet",
    ]
    for key, value in encode.items():
        cmd.extend(["-x", f"{key}={value}"])

    subprocess.run(cmd, check=True)


def write_srgb_baseline(srgb_grid: np.ndarray) -> None:
    """sRGB PNG/JXL baseline: the same vibrant OkHSL grid used to build the ACES master."""
    png = np.clip(srgb_grid * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    png_path = ASSETS / "colors.png"
    Image.fromarray(png, "RGB").save(png_path)

    cjxl = shutil.which("cjxl")
    if cjxl is None:
        raise RuntimeError("cjxl not found on PATH")
    subprocess.run(
        [
            cjxl,
            str(png_path),
            str(ASSETS / f"colors_{JXL_SUFFIX}.jxl"),
            "-d",
            str(JXL_DISTANCE),
            "-e",
            str(JXL_EFFORT),
            f"--faster_decoding={JXL_FASTER_DECODING}",
            "--quiet",
            "-x",
            "color_space=sRGB",
        ],
        check=True,
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    srgb_grid, lightness_map = build_okhsl_srgb_grid()
    master = build_master_aces2065(srgb_grid, lightness_map)
    write_exr_half(ASSETS / "master_aces2065.exr", master, "ACES2065-1")
    print(f"wrote assets/master_aces2065.exr ({SIZE}x{SIZE}, half, {EXR_COMPRESSION})")

    manifest_fixtures: list[dict[str, object]] = []

    tmp_dir = tempfile.mkdtemp(prefix="jxlit-fixtures-")
    for spec in FIXTURES:
        pixels = ocio_transform(master, spec.ocio_space)

        # Display/log-encoded fixtures live in [0,1]; cjxl clamps any value >1.0 unless the
        # encoding is sRGB/709-linear (the HDR fixture), so clamp the rest to match the codec.
        if not spec.hdr:
            pixels = np.clip(pixels, 0.0, 1.0)

        ref_exr = ASSETS / f"{spec.slug}_ref.exr"
        jxl_path = ASSETS / f"{spec.slug}_{JXL_SUFFIX}.jxl"
        write_exr_half(ref_exr, pixels, spec.ocio_space)

        # Encode from a half-quantized PFM (a raw format) so the JXL matches the committed
        # reference EXR and cjxl honours the color_space / icc_pathname color hints.
        pfm_path = Path(tmp_dir) / f"{spec.slug}.pfm"
        write_pfm(pfm_path, pixels.astype(np.float16).astype(np.float32))
        encode_jxl(pfm_path, jxl_path, spec.encode)

        manifest_fixtures.append(
            {
                "slug": spec.slug,
                "jxl": str(jxl_path.relative_to(ROOT)),
                "reference_exr": str(ref_exr.relative_to(ROOT)),
                "ocio_space": spec.ocio_space,
                "encode": spec.encode,
                "mae_tolerance": spec.mae_tolerance,
            }
        )
        print(f"wrote {jxl_path.name} + {ref_exr.name}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    write_srgb_baseline(srgb_grid)
    print(f"wrote colors.png + colors_{JXL_SUFFIX}.jxl")

    manifest = {"fixtures": manifest_fixtures}
    manifest_path = ASSETS / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
