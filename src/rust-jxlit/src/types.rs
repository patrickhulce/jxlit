//! Core public types for the decoder.

use std::fmt;

use jxl_threadpool::JxlThreadPool;

/// Options controlling decode behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Thread count for the decode pool. `None` uses available CPU cores.
    pub threads: Option<usize>,
    /// When true, collect per-phase timing measures in decode metadata.
    pub telemetry: bool,
}

/// A single flat phase timing measure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Measure {
    pub name: &'static str,
    pub start_ns: u64,
    pub duration_ns: u64,
}

/// Collected decode phase timings from the Rust core (monotonic timeline).
///
/// Bindings rebase measures against an outer `<lang>_decode` wall-clock origin
/// and expose consumer-facing `{ timebase, total_ns, measures }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeTelemetry {
    /// Unix-epoch nanoseconds at Rust decode-session start (for rebase delta).
    pub rust_timebase: u64,
    pub total_ns: u64,
    pub measures: Vec<Measure>,
}

/// Library-specific metadata returned on every decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxlitMeta {
    pub version: &'static str,
    pub telemetry: Option<DecodeTelemetry>,
}

/// Decode result metadata. Exposed as `_jxlit` in language bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
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
            jxlit: JxlitMeta {
                version,
                telemetry,
            },
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

/// A decoded image in interleaved (HWC) `f32` form.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedImage {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub pixels: Vec<f32>,
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
