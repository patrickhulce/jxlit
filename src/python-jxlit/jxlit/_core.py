"""Idiomatic Python API for jxlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jxlit._jxlit import DecodeOptions as _DecodeOptionsNative
from jxlit._jxlit import decode as _decode_native


@dataclass(frozen=True)
class DecodeOptions:
    """Options controlling decode behavior."""

    threads: int | None = None


@dataclass(frozen=True)
class DecodedImage:
    """Decoded JPEG XL image."""

    height: int
    width: int
    channels: int
    pixels: NDArray[np.float32]


def decode(data: bytes, *, options: DecodeOptions | None = None) -> DecodedImage:
    """Decode JPEG XL bytes into an f32 HWC pixel array.

    Args:
        data: Raw JPEG XL file bytes.
        options: Optional decode settings. When omitted, available CPU cores are used.

    Returns:
        Decoded image with shape metadata and an f32 HWC ndarray in [0, 1].
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    native_options = None
    if options is not None:
        native_options = _DecodeOptionsNative(threads=options.threads)

    pixels = _decode_native(bytes(data), options=native_options)
    height, width, channels = pixels.shape
    return DecodedImage(
        height=height,
        width=width,
        channels=channels,
        pixels=pixels,
    )
