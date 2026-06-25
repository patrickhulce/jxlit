from enum import Enum

import numpy as np
from numpy.typing import NDArray

class PixelLayout(Enum):
    Interleaved = 0
    Planar = 1

class DecodeOptions:
    threads: int | None
    telemetry: bool
    layout: PixelLayout

    def __init__(
        self,
        threads: int | None = ...,
        telemetry: bool = ...,
        layout: PixelLayout = ...,
    ) -> None: ...

class Measure:
    name: str
    start_ms: float
    duration_ms: float

class DecodeTelemetry:
    rust_timebase: float
    total_ms: float
    measures: list[Measure]

class JxlitMeta:
    version: str
    telemetry: DecodeTelemetry | None

class DecodeMetadata:
    _jxlit: JxlitMeta

class DecodedImage:
    height: int
    width: int
    channels: int
    metadata: DecodeMetadata

    @property
    def pixels(self) -> NDArray[np.float32]: ...

def decode(
    data: bytes,
    *,
    options: DecodeOptions | None = ...,
) -> DecodedImage: ...
