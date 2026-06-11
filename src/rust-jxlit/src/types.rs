//! Core public types for the decoder.

use std::fmt;

use jxl_threadpool::JxlThreadPool;

use crate::pipeline::gpu::GpuPixelBuffer;

/// Hardware backend for decode compute.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Hardware {
    /// CPU decode (default).
    #[default]
    Cpu,
    /// GPU decode (VarDCT frames only; placeholders until kernels land).
    Gpu,
}

/// Where decoded pixel samples should reside after decode completes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Destination {
    /// Return pixels as a CPU `Vec<f32>` (default).
    #[default]
    Cpu,
    /// Return pixels as a GPU buffer handle.
    Gpu,
}

/// Output pixel buffer layout for decoded samples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PixelLayout {
    /// Interleaved height-width-channel (HWC) order.
    #[default]
    Interleaved,
    /// Planar channel-height-width (CHW) order.
    Planar,
}

/// Options controlling decode behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Thread count for the decode pool. `None` uses available CPU cores.
    pub threads: Option<usize>,
    /// When true, collect per-phase timing measures in decode metadata.
    pub telemetry: bool,
    /// Flat `pixels` layout: interleaved HWC (default) or planar CHW.
    pub layout: PixelLayout,
    /// Compute backend for decode. Defaults to CPU.
    pub hardware: Hardware,
    /// Where decoded pixel samples should reside. Defaults to CPU.
    pub destination: Destination,
}

/// A single flat phase timing measure.
#[derive(Debug, Clone, PartialEq)]
pub struct Measure {
    pub name: &'static str,
    pub start_ms: f64,
    pub duration_ms: f64,
}

/// Collected decode phase timings from the Rust core (monotonic timeline).
///
/// Bindings rebase measures against an outer `<lang>_decode` wall-clock origin
/// and expose consumer-facing `{ timebase, total_ms, measures }`.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeTelemetry {
    /// Unix-epoch milliseconds at Rust decode-session start (for rebase delta).
    pub rust_timebase: f64,
    pub total_ms: f64,
    pub measures: Vec<Measure>,
}

/// Library-specific metadata returned on every decode.
#[derive(Debug, Clone, PartialEq)]
pub struct JxlitMeta {
    pub version: &'static str,
    pub telemetry: Option<DecodeTelemetry>,
}

/// Decode result metadata. Exposed as `_jxlit` in language bindings.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeMetadata {
    pub jxlit: JxlitMeta,
}

impl DecodeMetadata {
    pub(crate) fn with_version(version: &'static str) -> Self {
        Self {
            jxlit: JxlitMeta {
                version,
                telemetry: None,
            },
        }
    }

    pub(crate) fn with_telemetry(
        version: &'static str,
        telemetry: Option<DecodeTelemetry>,
    ) -> Self {
        Self {
            jxlit: JxlitMeta { version, telemetry },
        }
    }
}

/// Builds a thread pool from decode options.
pub(crate) fn pool_for_options(options: &DecodeOptions) -> JxlThreadPool {
    #[cfg(feature = "rayon")]
    {
        match options.threads {
            None | Some(0) => JxlThreadPool::rayon(None),
            Some(n) => JxlThreadPool::rayon(Some(n)),
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        let _ = options;
        JxlThreadPool::none()
    }
}

/// Decoded pixel samples on CPU or GPU.
#[derive(Debug)]
pub enum DecodedPixels {
    /// CPU-resident flat `f32` buffer (HWC when interleaved, CHW when planar).
    Cpu(Vec<f32>),
    /// GPU-resident buffer handle.
    Gpu(GpuPixelBuffer),
}

impl DecodedPixels {
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(v) => v.len(),
            Self::Gpu(g) => g.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn cpu(self) -> Option<Vec<f32>> {
        match self {
            Self::Cpu(v) => Some(v),
            Self::Gpu(_) => None,
        }
    }

    pub fn as_cpu(&self) -> Option<&[f32]> {
        match self {
            Self::Cpu(v) => Some(v),
            Self::Gpu(_) => None,
        }
    }
}

/// A decoded image as a flat `f32` buffer (HWC when interleaved, CHW when planar).
#[derive(Debug)]
pub struct DecodedImage {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub pixels: DecodedPixels,
    pub metadata: DecodeMetadata,
}

impl DecodedImage {
    pub(crate) fn attach_metadata(mut self, metadata: DecodeMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Error type returned by [`crate::decode`].
#[derive(Debug)]
pub struct DecodeError(String);

impl DecodeError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        DecodeError(message.into())
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DecodeError {}
