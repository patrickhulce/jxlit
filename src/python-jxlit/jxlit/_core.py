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
    start_ns: int
    duration_ns: int


@dataclass(frozen=True)
class DecodeTelemetry:
    timebase: int
    total_ns: int
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
    timebase: int,
    wall_ns: int,
    outer_name: str,
) -> DecodeTelemetry:
    delta = int(native.rust_timebase) - timebase  # type: ignore[attr-defined]
    shifted = [
        Measure(
            name=str(measure.name),  # type: ignore[attr-defined]
            start_ns=int(measure.start_ns) + delta,  # type: ignore[attr-defined]
            duration_ns=int(measure.duration_ns),  # type: ignore[attr-defined]
        )
        for measure in native.measures  # type: ignore[attr-defined]
    ]
    return DecodeTelemetry(
        timebase=timebase,
        total_ns=wall_ns,
        measures=[Measure(outer_name, 0, wall_ns), *shifted],
    )


def _telemetry_from_native(
    telemetry: object | None,
    *,
    timebase: int | None = None,
    wall_ns: int | None = None,
    outer_name: str = "python_decode",
) -> DecodeTelemetry | None:
    if telemetry is None:
        return None
    if timebase is not None and wall_ns is not None:
        return _rebase_telemetry(
            telemetry,
            timebase=timebase,
            wall_ns=wall_ns,
            outer_name=outer_name,
        )
    return DecodeTelemetry(
        timebase=int(telemetry.rust_timebase),  # type: ignore[attr-defined]
        total_ns=int(telemetry.total_ns),  # type: ignore[attr-defined]
        measures=[
            Measure(
                name=str(measure.name),  # type: ignore[attr-defined]
                start_ns=int(measure.start_ns),  # type: ignore[attr-defined]
                duration_ns=int(measure.duration_ns),  # type: ignore[attr-defined]
            )
            for measure in telemetry.measures  # type: ignore[attr-defined]
        ],
    )


def _decoded_image_from_native(
    decoded: _DecodedImageNative,
    *,
    timebase: int | None = None,
    wall_ns: int | None = None,
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
                    wall_ns=wall_ns,
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
        timebase = time.time_ns()
        start = time.perf_counter_ns()
        decoded = _decode_native(bytes(data), options=native_options)
        wall_ns = time.perf_counter_ns() - start
        return _decoded_image_from_native(decoded, timebase=timebase, wall_ns=wall_ns)

    decoded = _decode_native(bytes(data), options=native_options)
    return _decoded_image_from_native(decoded)


def telemetry_to_dict(telemetry: DecodeTelemetry) -> dict[str, object]:
    """Serialize rebased telemetry for benchmark JSON output."""
    return {
        "timebase": telemetry.timebase,
        "total_ns": telemetry.total_ns,
        "measures": [
            {
                "name": measure.name,
                "start_ns": measure.start_ns,
                "duration_ns": measure.duration_ns,
            }
            for measure in telemetry.measures
        ],
    }


def print_phase_summary(telemetry: DecodeTelemetry, *, lang: str, top_n: int = 10) -> None:
    """Print the largest phases to stderr."""
    import sys

    outer = next((m for m in telemetry.measures if m.name.endswith("_decode")), None)
    total_ns = outer.duration_ns if outer is not None else telemetry.total_ns
    ranked = sorted(telemetry.measures, key=lambda m: m.duration_ns, reverse=True)[:top_n]
    print(f"\n== phase breakdown ({lang}) ==", file=sys.stderr)
    for measure in ranked:
        ms = measure.duration_ns / 1_000_000.0
        pct = 100.0 * measure.duration_ns / total_ns if total_ns else 0.0
        print(f"{measure.name:<16} {ms:8.2f}ms {pct:6.1f}%", file=sys.stderr)
