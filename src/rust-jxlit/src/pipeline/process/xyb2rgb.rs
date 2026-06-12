//! Color-transform post-decode step: XYB -> requested output encoding.
//!
//! `run_xyb2rgb` drives the vendored keyframe color transform
//! (`RenderContext::postprocess_keyframe`), which builds a `jxl_color`
//! `ColorTransform` from the image metadata + requested encoding + embedded ICC
//! and runs it (handling YCbCr->RGB and CMYK black inversion).
//! `run_color_for_record` is the intermediate conversion applied to non-final
//! frames before blending.

use std::sync::Arc;

use jxl_image::ImageHeader;
use jxl_threadpool::JxlThreadPool;

use crate::pipeline::gpu::{
    DeviceImage, GpuEnvironment, availability, from_cpu_arc, into_cpu_arc, kernels,
};
use crate::types::DecodeOptions;
use crate::vendor::jxl_render::{IndexedFrame, RenderContext, Result, util};

/// Runs the final keyframe color transform (XYB -> requested encoding).
pub fn run_xyb2rgb(
    ctx: &RenderContext,
    frame: &IndexedFrame,
    grid: Arc<DeviceImage>,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<Arc<DeviceImage>> {
    if availability::run_xyb2rgb_available(ctx, frame, grid.as_ref(), options, env) {
        let _gpu = crate::phase_guard!("xyb2rgb_gpu");
        return kernels::run_xyb2rgb_on_gpu(ctx, frame, grid);
    }

    let _cpu = crate::phase_guard!("xyb2rgb_cpu");
    let cpu = {
        let _download = crate::phase_guard!("xyb2rgb_cpu_download");
        into_cpu_arc(grid)?
    };
    let out = {
        let _transform = crate::phase_guard!("xyb2rgb_cpu_transform");
        ctx.postprocess_keyframe(frame, cpu)?
    };
    Ok(from_cpu_arc(out))
}

/// Converts a non-final frame's color buffer "for record" prior to blending.
pub fn run_color_for_record(
    image_header: &ImageHeader,
    do_ycbcr: bool,
    fb: &mut DeviceImage,
    pool: &JxlThreadPool,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<()> {
    if availability::run_color_for_record_available(image_header, do_ycbcr, fb, options, env) {
        return kernels::run_color_for_record_on_gpu(image_header, do_ycbcr, fb, pool);
    }

    let image = fb
        .ensure_cpu()
        .expect("image must be CPU-resident when color-for-record GPU kernel is unavailable");
    util::convert_color_for_record(image_header, do_ycbcr, image, pool)
}
