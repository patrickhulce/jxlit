//! Render stage: turns a loaded [`RenderContext`] into a planar [`RenderedFrame`].
//!
//! This forks `jxl_render::RenderContext::render_keyframe` into our own staged
//! flow. [`render_keyframe`] dispatches the per-frame decode to the VarDCT or
//! Modular path, runs the shared post-decode stage, blends, and applies the
//! color transform:
//!
//! ```text
//! decode_frame -> {vardct|modular} -> postprocess -> blend -> xyb2rgb
//! ```
//!
//! The VarDCT path is further split into entropy / dequant / inverse-DCT stages
//! (see the [`entropy`]/[`dequant`]/[`idct`] modules). Numeric work is delegated
//! to the vendored jxl-oxide building blocks so the output is bit-identical to
//! the upstream renderer.

pub(crate) mod dequant;
pub(crate) mod entropy;
pub(crate) mod idct;
pub(crate) mod postprocess;
pub(crate) mod xyb2rgb;

use std::sync::Arc;

use jxl_image::{BitDepth, ExtraChannelType, ImageHeader};
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::header::Encoding;
use crate::vendor::jxl_render::{
    Error as VendorError, ImageWithRegion, ReferenceFrames, Region, RenderCache, RenderContext,
    util,
};

use super::parse::FramePlan;
use super::{modular, vardct};
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

/// Renders the given keyframe through the staged decode flow.
///
/// The `plan` describes the per-tile structure (the parse/render boundary). It
/// is accepted here to lock in the seam; per-tile threaded consumption of the
/// plan is deferred to a later change.
pub fn render_keyframe(
    ctx: &RenderContext,
    image_header: &ImageHeader,
    keyframe_index: usize,
    _plan: &FramePlan,
) -> Result<RenderedFrame, DecodeError> {
    let idx = ctx
        .keyframe_frame_index(keyframe_index)
        .ok_or_else(|| DecodeError::new("keyframe not loaded"))?;

    let image = render_keyframe_image(ctx, idx).map_err(|e| DecodeError::new(e.to_string()))?;

    let frame = &ctx.frames()[idx];
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

/// Reproduces `render_by_index` + `postprocess_keyframe` through the staged
/// decode: decode the frame, inject the result into the vendored blend
/// machinery, then run the color transform.
fn render_keyframe_image(
    ctx: &RenderContext,
    idx: usize,
) -> Result<Arc<ImageWithRegion>, VendorError> {
    let frame = &ctx.frames()[idx];
    let image_region = ctx.requested_image_region();
    let frame_visibility = ctx.get_previous_frames_visibility(frame);
    let pool = ctx.pool();

    let grid = if ctx.narrow_modular() {
        let refs = ctx.reference_frames_narrow(idx);
        let mut cache = RenderCache::<i16>::new(frame);
        decode_frame::<i16>(
            frame,
            refs,
            &mut cache,
            image_region,
            pool,
            frame_visibility,
        )?
    } else {
        let refs = ctx.reference_frames_wide(idx);
        let mut cache = RenderCache::<i32>::new(frame);
        decode_frame::<i32>(
            frame,
            refs,
            &mut cache,
            image_region,
            pool,
            frame_visibility,
        )?
    };

    let blended = ctx.blend_staged(idx, grid)?;
    xyb2rgb::run(ctx, frame, blended)
}

/// Fork of `jxl_render::render::render_frame`: dispatches the coefficient decode
/// to the VarDCT or Modular path, then runs the shared post-decode stage.
fn decode_frame<S: Sample>(
    frame: &crate::vendor::jxl_render::IndexedFrame,
    reference_frames: ReferenceFrames<S>,
    cache: &mut RenderCache<S>,
    image_region: Region,
    pool: &JxlThreadPool,
    frame_visibility: (usize, usize),
) -> Result<ImageWithRegion, VendorError> {
    let frame_region = util::image_region_to_frame(frame, image_region, false);

    let image_header = frame.image_header();
    let frame_header = frame.header();
    let frame_region = util::pad_lf_region(frame_header, frame_region);

    let upsampled_full_frame_region =
        Region::with_size(frame_header.sample_width(1), frame_header.sample_height(1));
    let upsampling_valid_region = util::pad_upsampling(image_header, frame_header, frame_region)
        .intersection(upsampled_full_frame_region);

    let full_frame_region = Region::with_size(
        frame_header.color_sample_width(),
        frame_header.color_sample_height(),
    );
    let color_padded_region = util::pad_color_region(image_header, frame_header, frame_region)
        .intersection(full_frame_region);

    let mut fb = match frame_header.encoding {
        Encoding::Modular => modular::decode(frame, cache, color_padded_region, pool)?,
        Encoding::VarDct => {
            let result = vardct::decode(
                frame,
                reference_frames.lf.as_ref(),
                cache,
                color_padded_region,
                pool,
            );
            match (result, reference_frames.lf) {
                (Ok(grid), _) => grid,
                (Err(e), Some(lf))
                    if matches!(e, VendorError::IncompleteFrame) || e.unexpected_eof() =>
                {
                    let render = lf.image.run_with_image()?;
                    let render = render.blend(None, pool)?;
                    let mut render = render.upsample_lf(1)?;
                    render.fill_opaque_alpha(&image_header.metadata.ec_info);
                    render
                }
                (Err(e), _) => return Err(e),
            }
        }
    };

    if frame_header.do_ycbcr {
        fb.upsample_jpeg(color_padded_region, image_header.metadata.bit_depth)?;
    }

    postprocess::run(
        frame,
        &mut fb,
        color_padded_region,
        upsampling_valid_region,
        reference_frames.refs.clone(),
        cache,
        frame_visibility,
        pool,
    )?;

    Ok(fb)
}
