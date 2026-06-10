//! Frame-scope orchestration.
//!
//! `render_frame` is the per-keyframe entry: it decodes the frame (deciding the
//! VarDCT vs Modular flow), runs the shared post-decode (loop filters, features,
//! upsampling, color-for-record), blends, applies the color transform, and
//! interleaves into the final [`DecodedImage`]. This file also holds the
//! frame-scope buffer/LF builders and the blend wrapper used by the flows.

use std::collections::HashMap;
use std::sync::Arc;

use jxl_grid::{AlignedGrid, AllocTracker};
use jxl_image::{BitDepth, ExtraChannelType};
use jxl_modular::{ChannelShift, Sample, image::TransformedModularSubimage};
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{LfGlobal, LfGlobalVarDct, LfGroup};
use crate::vendor::jxl_frame::header::Encoding;
use crate::vendor::jxl_render::{
    Error, ImageBuffer, ImageWithRegion, IndexedFrame, Reference, ReferenceFrames, Region,
    RenderCache, RenderContext, Result,
};

use crate::pipeline::structs::container::ContainerCtx;
use crate::pipeline::{decode, parse, process, render};
use crate::{DecodeError, DecodedImage};

/// A rendered keyframe in planar form, plus the metadata needed to interleave.
/// Held here (rather than in `structs/`) since it is the direct output of the
/// render stage; `process::interleave` consumes it.
pub struct RenderedFrame {
    pub image: Arc<ImageWithRegion>,
    pub target_frame_region: Region,
    pub orientation: u32,
    pub color_bit_depth: BitDepth,
    pub extra_channels: Vec<(ExtraChannelType, BitDepth)>,
    pub is_cmyk: bool,
    pub render_spot_color: bool,
}

/// Renders the given keyframe through the staged decode flow and interleaves it
/// into an HWC `f32` [`DecodedImage`].
pub fn render_frame(
    container: &ContainerCtx,
    keyframe_index: usize,
) -> std::result::Result<DecodedImage, DecodeError> {
    let _render = crate::phase_guard!("render");
    let ctx = &container.render_context;
    let image_header = container.declaration.image_header.as_ref();

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

    let rendered = RenderedFrame {
        image,
        target_frame_region,
        orientation: metadata.orientation,
        color_bit_depth: metadata.bit_depth,
        extra_channels,
        is_cmyk: ctx.requested_color_encoding().is_cmyk(),
        render_spot_color: !metadata.grayscale(),
    };

    process::interleave::build_decoded_image(&rendered)
}

/// Decodes the keyframe to a blended, color-transformed planar image. Forks
/// `render_by_index` + `postprocess_keyframe`: decode the color buffer, inject
/// it into the vendored blend machinery, then run the color transform.
fn render_keyframe_image(ctx: &RenderContext, idx: usize) -> Result<Arc<ImageWithRegion>> {
    let _render_keyframe = crate::phase_guard!("render_keyframe");
    let frame = &ctx.frames()[idx];
    let image_region = ctx.requested_image_region();
    let frame_visibility = ctx.get_previous_frames_visibility(frame);
    let pool = ctx.pool();

    let grid = if ctx.narrow_modular() {
        let refs = ctx.reference_frames_narrow(idx);
        let mut cache = RenderCache::<i16>::new(frame);
        decode_color_buffer::<i16>(
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
        decode_color_buffer::<i32>(
            frame,
            refs,
            &mut cache,
            image_region,
            pool,
            frame_visibility,
        )?
    };

    let blended = {
        let _blend = crate::phase_guard!("blend");
        run_blend(ctx, idx, grid)?
    };
    let _xyb2rgb = crate::phase_guard!("xyb2rgb");
    process::xyb2rgb::run_xyb2rgb(ctx, frame, blended)
}

/// Decodes a single frame's color buffer: dispatch to the VarDCT/Modular flow,
/// then run the shared post-decode stage (filters, features, upsampling and
/// color-for-record).
fn decode_color_buffer<S: Sample>(
    frame: &IndexedFrame,
    reference_frames: ReferenceFrames<S>,
    cache: &mut RenderCache<S>,
    image_region: Region,
    pool: &JxlThreadPool,
    frame_visibility: (usize, usize),
) -> Result<ImageWithRegion> {
    let _decode_color_buffer = crate::phase_guard!("decode_color_buffer");
    let image_header = frame.image_header();
    let frame_header = frame.header();

    let regions = render::region::render_region(frame, image_region);
    let color_padded_region = regions.color_padded_region;
    let upsampling_valid_region = regions.upsampling_valid_region;

    let mut fb = match frame_header.encoding {
        Encoding::Modular => {
            render::flows::modular::run_modular_flow(frame, cache, color_padded_region, pool)?
        }
        Encoding::VarDct => {
            let result = render::flows::vardct::run_vardct_flow(
                frame,
                reference_frames.lf.as_ref(),
                cache,
                color_padded_region,
                pool,
            );
            match (result, reference_frames.lf) {
                (Ok(grid), _) => grid,
                (Err(e), Some(lf)) if matches!(e, Error::IncompleteFrame) || e.unexpected_eof() => {
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
        let _jpeg_upsample = crate::phase_guard!("jpeg_upsample");
        process::upsample::run_jpeg_upsample(
            &mut fb,
            color_padded_region,
            image_header.metadata.bit_depth,
        )?;
    }

    let color_channels = fb.color_channels();
    {
        let _loop_filters = crate::phase_guard!("loop_filters");
        process::filters::run_loop_filters(
            frame,
            &mut fb,
            color_padded_region,
            &cache.lf_groups,
            pool,
        )?;
    }
    fb.remove_color_channels(color_channels);
    fb.prepare_color_upsampling(frame_header);
    {
        let _features = crate::phase_guard!("features");
        process::features::run_features(
            frame,
            &mut fb,
            upsampling_valid_region,
            reference_frames.refs.clone(),
            cache.lf_global.as_ref(),
            frame_visibility.0,
            frame_visibility.1,
            pool,
        )?;
    }
    {
        let _nonsep_upsample = crate::phase_guard!("nonsep_upsample");
        process::upsample::run_nonseparable_upsample(
            &mut fb,
            image_header,
            frame_header,
            upsampling_valid_region,
        )?;
    }
    if !frame_header.save_before_ct && !frame_header.is_last {
        let _color_for_record = crate::phase_guard!("color_for_record");
        process::xyb2rgb::run_color_for_record(image_header, frame_header.do_ycbcr, &mut fb, pool)?;
    }

    Ok(fb)
}

/// Allocates the (empty) XYB high-frequency coefficient buffer that pass-group
/// decode writes into, shaped to the modular region and channel shifts.
pub fn build_coefficient_buffer(
    frame_header: &FrameHeader,
    modular_region: Region,
    tracker: Option<&AllocTracker>,
) -> Result<ImageWithRegion> {
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let Region { width, height, .. } = modular_region;

    let mut color_buffer = ImageWithRegion::new(3, tracker);
    for shift in shifts_cbycr {
        let (w8, h8) = shift.shift_size((width.div_ceil(8), height.div_ceil(8)));
        let width = w8 * 8;
        let height = h8 * 8;
        let buffer = AlignedGrid::with_alloc_tracker(width as usize, height as usize, tracker)?;
        color_buffer.append_channel_shifted(ImageBuffer::F32(buffer), modular_region, shift);
    }
    Ok(color_buffer)
}

/// Reads the LF groups and produces the dequantized low-frequency image, either
/// from this frame's LF data (with chroma-from-luma + adaptive smoothing) or
/// from a referenced LF frame.
#[allow(clippy::too_many_arguments)]
pub fn build_low_frequency_image<S: Sample>(
    frame: &IndexedFrame,
    low_frequency_global: &LfGlobal<S>,
    low_frequency_groups: &mut HashMap<u32, LfGroup<S>>,
    modular_lf_groups: Vec<TransformedModularSubimage<S>>,
    modular_lf_region: Region,
    lf_frame: Option<&Reference<S>>,
    low_frequency_global_vardct: &LfGlobalVarDct,
    subsampled: bool,
    pool: &JxlThreadPool,
) -> Result<ImageWithRegion> {
    let low_frequency_image = parse::frames::read_low_frequency_groups(
        frame,
        low_frequency_global,
        low_frequency_groups,
        modular_lf_groups,
        modular_lf_region,
        pool,
    )?;

    if let Some(lf) = lf_frame {
        let lf_frame = Arc::clone(&lf.image).run_with_image()?;
        Ok(lf_frame.blend(None, pool)?.try_clone()?)
    } else {
        let mut low_frequency_image = low_frequency_image.unwrap();
        decode::dequant::run_low_frequency_dequant(
            &mut low_frequency_image,
            &low_frequency_global.lf_dequant,
            &low_frequency_global_vardct.quantizer,
            &low_frequency_global_vardct.lf_chan_corr,
            subsampled,
            frame.header().flags.skip_adaptive_lf_smoothing(),
        )?;
        Ok(low_frequency_image)
    }
}

/// Blends the decoded frame onto the canvas via the vendored staged blender.
pub fn run_blend(
    ctx: &RenderContext,
    idx: usize,
    grid: ImageWithRegion,
) -> Result<Arc<ImageWithRegion>> {
    ctx.blend_staged(idx, grid)
}
