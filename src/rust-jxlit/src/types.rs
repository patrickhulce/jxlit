//! Core public types for the decoder.

use std::fmt;

use jxl_threadpool::JxlThreadPool;

/// Options controlling decode behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Thread count for the decode pool. `None` uses available CPU cores.
    pub threads: Option<usize>,
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
