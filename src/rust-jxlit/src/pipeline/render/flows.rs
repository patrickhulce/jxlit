//! Per-encoding flow implementations.
//!
//! Each flow turns a frame's coefficients into a pixel-domain XYB color buffer;
//! `render::frame::render_frame` decides which flow to run and handles the
//! shared post-decode, blend, color transform and interleave around it.

pub mod modular;
pub mod vardct;
