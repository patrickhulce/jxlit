"""Idiomatic Python API for jxlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jxlit._jxlit import decode as _decode_native


@dataclass(frozen=True)
class DecodedImage:
    """Decoded JPEG XL image."""

    height: int
    width: int
    channels: int
    pixels: NDArray[np.float32]


def decode(data: bytes) -> DecodedImage:
    """Decode JPEG XL bytes into an f32 HWC pixel array.

    Args:
        data: Raw JPEG XL file bytes.

    Returns:
        Decoded image with shape metadata and an f32 HWC ndarray in [0, 1].
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    pixels = _decode_native(bytes(data))
    height, width, channels = pixels.shape
    return DecodedImage(
        height=height,
        width=width,
        channels=channels,
        pixels=pixels,
    )
