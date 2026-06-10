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

use crate::vendor::jxl_render::{ImageWithRegion, IndexedFrame, RenderContext, Result, util};

/// Runs the final keyframe color transform (XYB -> requested encoding).
pub fn run_xyb2rgb(
    ctx: &RenderContext,
    frame: &IndexedFrame,
    grid: Arc<ImageWithRegion>,
) -> Result<Arc<ImageWithRegion>> {
    ctx.postprocess_keyframe(frame, grid)
}

/// Converts a non-final frame's color buffer "for record" prior to blending.
pub fn run_color_for_record(
    image_header: &ImageHeader,
    do_ycbcr: bool,
    fb: &mut ImageWithRegion,
    pool: &JxlThreadPool,
) -> Result<()> {
    util::convert_color_for_record(image_header, do_ycbcr, fb, pool)
}
