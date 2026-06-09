// Vendored from jxl-oxide (jxl-render 0.12.4), (c) Wonwoo Choi, licensed MIT OR Apache-2.0.
// Source: https://github.com/tirr-c/jxl-oxide/blob/f8ae722ef2d6b782941c89517d19cfbf605c4a9d/crates/jxl-render/src/vardct/wasm32/mod.rs
// Copied as-is; only crate-path references changed.

use jxl_grid::AllocTracker;

use super::generic;

mod dct;
mod transform;
pub use transform::transform_varblocks;

pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::vendor::jxl_render::Result<()> {
    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale, tracker)
}
