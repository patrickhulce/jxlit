"""Idiomatic Python API for jxlit."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jxlit._jxlit import DecodeOptions as _DecodeOptionsNative
from jxlit._jxlit import DecodedImage as _DecodedImageNative
from jxlit._jxlit import decode as _decode_native


@dataclass(frozen=True)
class DecodeOptions:
    """Options controlling decode behavior."""

    threads: int | None = None
    telemetry: bool = False


@dataclass(frozen=True)
class Measure:
    name: str
    start_ms: float
    duration_ms: float


@dataclass(frozen=True)
class DecodeTelemetry:
    timebase: float
    total_ms: float
    measures: list[Measure]


@dataclass(frozen=True)
class JxlitMeta:
    version: str
    telemetry: DecodeTelemetry | None


@dataclass(frozen=True)
class DecodedImage:
    """Decoded JPEG XL image."""

    height: int
    width: int
    channels: int
    pixels: NDArray[np.float32]
    metadata: dict[str, JxlitMeta]


def _rebase_telemetry(
    native: object,
    *,
    timebase: float,
    wall_ms: float,
    outer_name: str,
) -> DecodeTelemetry:
    delta = float(native.rust_timebase) - timebase  # type: ignore[attr-defined]
    shifted = [
        Measure(
            name=str(measure.name),
            start_ms=float(measure.start_ms) + delta,
            duration_ms=float(measure.duration_ms),
        )
        for measure in native.measures  # type: ignore[attr-defined]
    ]
    return DecodeTelemetry(
        timebase=timebase,
        total_ms=wall_ms,
        measures=[Measure(outer_name, 0.0, wall_ms), *shifted],
    )


def _telemetry_from_native(
    telemetry: object | None,
    *,
    timebase: float | None = None,
    wall_ms: float | None = None,
    outer_name: str = "python_decode",
) -> DecodeTelemetry | None:
    if telemetry is None:
        return None
    if timebase is not None and wall_ms is not None:
        return _rebase_telemetry(
            telemetry,
            timebase=timebase,
            wall_ms=wall_ms,
            outer_name=outer_name,
        )
    return DecodeTelemetry(
        timebase=float(telemetry.rust_timebase),  # type: ignore[attr-defined]
        total_ms=float(telemetry.total_ms),  # type: ignore[attr-defined]
        measures=[
            Measure(
                name=str(measure.name),
                start_ms=float(measure.start_ms),
                duration_ms=float(measure.duration_ms),
            )
            for measure in telemetry.measures  # type: ignore[attr-defined]
        ],
    )


def _decoded_image_from_native(
    decoded: _DecodedImageNative,
    *,
    timebase: float | None = None,
    wall_ms: float | None = None,
) -> DecodedImage:
    jxlit_meta = decoded.metadata._jxlit
    return DecodedImage(
        height=decoded.height,
        width=decoded.width,
        channels=decoded.channels,
        pixels=decoded.pixels,
        metadata={
            "_jxlit": JxlitMeta(
                version=jxlit_meta.version,
                telemetry=_telemetry_from_native(
                    jxlit_meta.telemetry,
                    timebase=timebase,
                    wall_ms=wall_ms,
                ),
            )
        },
    )


def decode(data: bytes, *, options: DecodeOptions | None = None) -> DecodedImage:
    """Decode JPEG XL bytes into an f32 HWC pixel array.

    Args:
        data: Raw JPEG XL file bytes.
        options: Optional decode settings. When omitted, available CPU cores are used.

    Returns:
        Decoded image with shape metadata, pixels, and library metadata.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    telemetry = options is not None and options.telemetry
    native_options = None
    if options is not None:
        native_options = _DecodeOptionsNative(
            threads=options.threads,
            telemetry=options.telemetry,
        )

    if telemetry:
        timebase = time.time_ns() / 1_000_000.0
        start = time.perf_counter_ns()
        decoded = _decode_native(bytes(data), options=native_options)
        wall_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        return _decoded_image_from_native(decoded, timebase=timebase, wall_ms=wall_ms)

    decoded = _decode_native(bytes(data), options=native_options)
    return _decoded_image_from_native(decoded)


def telemetry_to_dict(telemetry: DecodeTelemetry) -> dict[str, object]:
    """Serialize rebased telemetry for benchmark JSON output."""
    return {
        "timebase": telemetry.timebase,
        "total_ms": telemetry.total_ms,
        "measures": [
            {
                "name": measure.name,
                "start_ms": measure.start_ms,
                "duration_ms": measure.duration_ms,
            }
            for measure in telemetry.measures
        ],
    }


def print_phase_summary(telemetry: DecodeTelemetry, *, lang: str, top_n: int = 10) -> None:
    """Print the largest phases to stderr."""
    import sys

    outer = next((m for m in telemetry.measures if m.name.endswith("_decode")), None)
    total_ms = outer.duration_ms if outer is not None else telemetry.total_ms
    ranked = sorted(telemetry.measures, key=lambda m: m.duration_ms, reverse=True)[:top_n]
    print(f"\n== phase breakdown ({lang}) ==", file=sys.stderr)
    for measure in ranked:
        pct = 100.0 * measure.duration_ms / total_ms if total_ms else 0.0
        print(f"{measure.name:<16} {measure.duration_ms:8.2f}ms {pct:6.1f}%", file=sys.stderr)
