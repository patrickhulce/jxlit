//! Render stage: turns a loaded [`RenderContext`] into a planar [`RenderedFrame`].
//!
//! Today this delegates the entire frame decode to
//! `jxl_render::RenderContext::render_keyframe`, which performs entropy decode,
//! VarDCT coefficient decode, inverse DCT, loop filters/upsampling, and the
//! XYB->RGB color transform internally (with its own threading). The child
//! modules below are scaffolding placeholders marking where those stages will
//! live once the renderer is forked for per-tile / GPU execution; see
//! [`FramePlan`] for the parse/render tile boundary.

mod entropy;
mod idct;
mod postprocess;
mod vardct;
mod xyb2rgb;

use std::sync::Arc;

use jxl_image::{BitDepth, ExtraChannelType, ImageHeader};
use jxl_render::{ImageWithRegion, Region, RenderContext};

use super::parse::FramePlan;
use crate::DecodeError;

/// A rendered keyframe in planar form, plus the metadata needed to interleave
/// (or, in the future, emit as CHW planar data).
pub struct RenderedFrame {
    pub image: Arc<ImageWithRegion>,
    pub target_frame_region: Region,
    pub orientation: u32,
    pub color_bit_depth: BitDepth,
    pub extra_channels: Vec<(ExtraChannelType, BitDepth)>,
    pub is_cmyk: bool,
    pub render_spot_color: bool,
}

/// Renders the given keyframe.
///
/// The `plan` describes the per-tile structure (the parse/render boundary). It
/// is accepted here to lock in the seam, but for now rendering is delegated to
/// `jxl-render`'s monolithic `render_keyframe`; per-tile, threaded consumption
/// of the plan is deferred to the renderer fork.
pub fn render_keyframe(
    ctx: &RenderContext,
    image_header: &ImageHeader,
    keyframe_index: usize,
    _plan: &FramePlan,
) -> Result<RenderedFrame, DecodeError> {
    let image = ctx
        .render_keyframe(keyframe_index)
        .map_err(|e| DecodeError::new(e.to_string()))?;

    let frame = ctx
        .keyframe(keyframe_index)
        .ok_or_else(|| DecodeError::new("keyframe not loaded"))?;
    let frame_header = frame.header();

    let image_region = ctx.image_region().apply_orientation(image_header);
    let target_frame_region = image_region.translate(-frame_header.x0, -frame_header.y0);

    let metadata = &image_header.metadata;
    let extra_channels = metadata
        .ec_info
        .iter()
        .map(|ec_info| (ec_info.ty, ec_info.bit_depth))
        .collect();

    Ok(RenderedFrame {
        image,
        target_frame_region,
        orientation: metadata.orientation,
        color_bit_depth: metadata.bit_depth,
        extra_channels,
        is_cmyk: ctx.requested_color_encoding().is_cmyk(),
        render_spot_color: !metadata.grayscale(),
    })
}
