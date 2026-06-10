from pathlib import Path

import numpy as np
from PIL import Image

from jxlit import DecodeOptions, decode

ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
JXL_FIXTURE = ASSETS_DIR / "colors_e1_d0p5_fd4.jxl"
PNG_FIXTURE = ASSETS_DIR / "colors.png"
MAE_TOLERANCE = 0.02


def load_png_rgb_f32(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def test_decode_colors_fixture_is_close_to_png() -> None:
    decoded = decode(JXL_FIXTURE.read_bytes())
    expected = load_png_rgb_f32(PNG_FIXTURE)

    assert decoded.height == expected.shape[0]
    assert decoded.width == expected.shape[1]
    assert decoded.channels == expected.shape[2]
    assert decoded.pixels.shape == expected.shape

    mae = float(np.abs(decoded.pixels - expected).mean())
    assert mae < MAE_TOLERANCE, f"mean absolute error {mae} exceeds {MAE_TOLERANCE}"


def test_decode_metadata_includes_version() -> None:
    decoded = decode(JXL_FIXTURE.read_bytes())
    assert decoded.metadata["_jxlit"].version
    assert decoded.metadata["_jxlit"].telemetry is None


def test_decode_telemetry_collects_measures() -> None:
    decoded = decode(
        JXL_FIXTURE.read_bytes(),
        options=DecodeOptions(telemetry=True),
    )
    telemetry = decoded.metadata["_jxlit"].telemetry
    assert telemetry is not None
    assert telemetry.timebase > 0
    assert telemetry.total_ns > 0
    assert telemetry.measures
    names = {measure.name for measure in telemetry.measures}
    assert "python_decode" in names
    assert "decode" in names
    assert "parse" in names
    assert "render" in names

    outer = next(m for m in telemetry.measures if m.name == "python_decode")
    assert outer.start_ns == 0
    assert outer.duration_ns == telemetry.total_ns

    inner_decode = next(m for m in telemetry.measures if m.name == "decode")
    assert inner_decode.start_ns > 0


def test_decode_rejects_non_bytes_like() -> None:
    try:
        decode("not-a-jxl")  # type: ignore[arg-type]
    except TypeError as exc:
        assert "bytes-like" in str(exc)
    else:
        raise AssertionError("expected TypeError")
