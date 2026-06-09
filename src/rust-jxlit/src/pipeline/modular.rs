//! Modular frame-decode path.
//!
//! The Modular encoding has no separate dequant/IDCT stages, so (unlike the
//! VarDCT path) it is not decomposed further: this forwards to the vendored
//! `jxl_render::modular::render_modular`, which performs the modular entropy
//! decode and inverse transforms.

use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_render::{
    ImageWithRegion, IndexedFrame, Region, RenderCache, Result, modular,
};

pub(crate) fn decode<S: Sample>(
    frame: &IndexedFrame,
    cache: &mut RenderCache<S>,
    region: Region,
    pool: &JxlThreadPool,
) -> Result<ImageWithRegion> {
    modular::render_modular(frame, cache, region, pool)
}
