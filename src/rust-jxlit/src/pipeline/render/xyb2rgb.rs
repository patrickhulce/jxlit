//! Color-transform stage (XYB -> requested output encoding).
//!
//! Reproduces the keyframe color transform by driving the vendored
//! `RenderContext::postprocess_keyframe`, which builds a `jxl_color`
//! `ColorTransform` from the image metadata + requested encoding + embedded ICC
//! and runs it (handling YCbCr->RGB and CMYK black inversion). This is the final
//! stage of `render_keyframe` in our flow.

use std::sync::Arc;

use crate::vendor::jxl_render::{ImageWithRegion, IndexedFrame, RenderContext, Result};

pub(crate) fn run(
    ctx: &RenderContext,
    frame: &IndexedFrame,
    grid: Arc<ImageWithRegion>,
) -> Result<Arc<ImageWithRegion>> {
    ctx.postprocess_keyframe(frame, grid)
}
