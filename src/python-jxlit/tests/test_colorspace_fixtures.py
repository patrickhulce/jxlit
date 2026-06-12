from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jxlit import decode

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "assets" / "manifest.json"


def load_manifest_fixtures() -> list[dict[str, Any]]:
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
    fixtures: list[dict[str, Any]] = manifest["fixtures"]
    return fixtures


def parse_pfm_rgb_f32(data: bytes) -> np.ndarray:
    offset = 0
    lines: list[bytes] = []
    while len(lines) < 3:
        end = data.index(b"\n", offset)
        lines.append(data[offset:end])
        offset = end + 1

    assert lines[0] == b"PF"
    width, height = map(int, lines[1].split())
    assert lines[2].startswith(b"-")
    rgb = np.frombuffer(data[offset:], dtype="<f4").reshape(height, width, 3)
    return rgb


def load_reference_rgb_f32(reference_exr: str) -> np.ndarray:
    result = subprocess.run(
        [
            "uv",
            "run",
            "scripts/exr_to_pfm.py",
            str(ROOT / reference_exr),
            "--stdout",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return parse_pfm_rgb_f32(result.stdout)


@pytest.mark.parametrize(
    "fixture",
    load_manifest_fixtures(),
    ids=lambda fixture: fixture["slug"],
)
def test_decode_colorspace_fixture_matches_reference(fixture: dict[str, Any]) -> None:
    decoded = decode((ROOT / fixture["jxl"]).read_bytes())
    expected = load_reference_rgb_f32(fixture["reference_exr"])

    assert decoded.height == expected.shape[0]
    assert decoded.width == expected.shape[1]
    assert decoded.channels == expected.shape[2]
    assert decoded.pixels.shape == expected.shape

    mae = float(np.abs(decoded.pixels - expected).mean())
    tolerance = float(fixture["mae_tolerance"])
    assert mae < tolerance, (
        f"{fixture['slug']}: mean absolute error {mae} exceeds {tolerance}"
    )
