"""Idiomatic Python API for jxlit."""

from __future__ import annotations

from jxlit._jxlit import decode as _decode_native


def decode(data: bytes) -> bytes:
    """Decode JPEG XL bytes into a pixel buffer.

    Args:
        data: Raw JPEG XL file bytes.

    Returns:
        Decoded pixel buffer. Currently a stub that returns empty bytes.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    return _decode_native(bytes(data))
