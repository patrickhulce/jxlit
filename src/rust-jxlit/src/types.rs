//! Core public types for the decoder.

use std::fmt;

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
