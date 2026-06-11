//! Modular frame-decode flow.
//!
//! The Modular encoding has no separate dequant/IDCT stages, so (unlike the
//! VarDCT flow) it is not decomposed further: this forwards to the vendored
//! `jxl_render::modular::render_modular`, which performs the modular entropy
//! decode and inverse transforms, returning the pixel-domain color buffer.

use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::pipeline::gpu::{DeviceImage, from_cpu};
use crate::vendor::jxl_render::{IndexedFrame, Region, RenderCache, Result, modular};

/// Decodes a Modular frame into its pixel-domain color buffer.
pub fn run_modular_flow<S: Sample>(
    frame: &IndexedFrame,
    cache: &mut RenderCache<S>,
    region: Region,
    pool: &JxlThreadPool,
) -> Result<DeviceImage> {
    let _modular_flow = crate::phase_guard!("modular_flow");
    modular::render_modular(frame, cache, region, pool).map(from_cpu)
}
