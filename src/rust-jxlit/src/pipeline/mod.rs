//! Staged JPEG XL decode pipeline built on the narrow jxl-oxide sub-crates.
//!
//! The decode is split into explicit stages so individual pieces can later be
//! swapped for GPU-accelerated implementations without disturbing the rest:
//!
//! ```text
//! read_codestream -> read_header -> build_context -> read_frames
//!   -> build_frame_plan -> render_keyframe -> interleave
//! ```
//!
//! [`parse::build_frame_plan`] exposes the per-tile (pass-group) structure of a
//! frame, which is the intended parse/render boundary for future threaded,
//! per-tile rendering.

pub mod interleave;
pub mod modular;
pub mod parse;
pub mod render;
pub mod vardct;

use crate::{DecodeError, DecodedImage};

/// Decodes a JPEG XL image into an interleaved (HWC) f32 buffer.
pub fn decode(input: &[u8]) -> Result<DecodedImage, DecodeError> {
    let codestream = parse::read_codestream(input)?;
    let header = parse::read_header(&codestream)?;
    let mut ctx = parse::build_context(header.image_header.clone(), header.embedded_icc)?;
    parse::read_frames(&mut ctx, &codestream[header.offset..])?;

    let keyframe_index = 0;
    let plan = parse::build_frame_plan(&ctx, keyframe_index)?;
    let rendered = render::render_keyframe(&ctx, &header.image_header, keyframe_index, &plan)?;
    interleave::interleave(&rendered)
}
