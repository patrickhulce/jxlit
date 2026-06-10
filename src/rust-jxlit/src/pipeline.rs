//! Staged JPEG XL decode pipeline built on the narrow jxl-oxide sub-crates.
//!
//! The decode is organized as scope orchestrators (`render_region` /
//! `render_tile` / `render_frame`) over per-encoding flow implementations
//! (`render::flows::{vardct,modular}`), plus scoped component folders:
//!
//! ```text
//! parse/    read bytes from the stream (container, frames, frame-global, tiles)
//! decode/   per-tile (pass-group) numeric slices (entropy, dequant, idct)
//! render/   scope orchestrators (region, tile, frame) + flow implementations
//! process/  post-decode steps on XYB / RGB data (filters, upsample, features,
//!           xyb2rgb, interleave)
//! structs/  Declaration (cheap description) vs Ctx (rich materialized state)
//! ```
//!
//! `pipeline::decode` is thin: it runs parse, then drives the keyframe loop
//! through `render::frame::render_frame`, which decides the VarDCT vs Modular
//! flow. Numeric work delegates to the vendored jxl-oxide building blocks, so
//! output is bit-identical to the upstream renderer.

pub mod decode;
pub mod parse;
pub mod process;
pub mod render;
pub mod structs;

use crate::types::pool_for_options;
use crate::{DecodeError, DecodeOptions, DecodedImage};

/// Decodes a JPEG XL image into an interleaved (HWC) f32 buffer.
pub fn decode(input: &[u8], options: &DecodeOptions) -> Result<DecodedImage, DecodeError> {
    let _parse = crate::phase_guard!("parse");
    let pool = pool_for_options(options);
    let codestream = parse::container::read_codestream(input)?;
    let declaration = parse::container::read_header(&codestream, pool.clone())?;
    let frame_offset = declaration.offset;
    let mut container = parse::container::build_container_ctx(declaration, pool)?;
    parse::frames::read_frames(&mut container.render_context, &codestream[frame_offset..])?;

    let keyframe_index = 0;
    render::frame::render_frame(&container, keyframe_index)
}
