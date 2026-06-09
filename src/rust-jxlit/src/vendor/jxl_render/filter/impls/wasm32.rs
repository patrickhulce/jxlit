// Vendored from jxl-oxide (jxl-render 0.12.4), (c) Wonwoo Choi, licensed MIT OR Apache-2.0.
// Source: https://github.com/tirr-c/jxl-oxide/blob/f8ae722ef2d6b782941c89517d19cfbf605c4a9d/crates/jxl-render/src/filter/impls/wasm32.rs
// Copied as-is; only crate-path references changed.

use crate::vendor::jxl_frame::{FrameHeader, filter::EpfParams};
use jxl_grid::{AlignedGrid, MutableSubgrid};
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_render::Region;
use crate::vendor::jxl_render::filter::epf::run_epf_rows;

mod epf;

pub fn epf<const STEP: usize>(
    input: &mut [MutableSubgrid<f32>; 3],
    output: &mut [MutableSubgrid<f32>; 3],
    color_padded_region: Region,
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&AlignedGrid<f32>>],
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        run_epf_rows(
            input,
            output,
            color_padded_region,
            frame_header,
            sigma_grid_map,
            epf_params,
            pool,
            Some(epf::epf_row_wasm32_simd128::<STEP>),
            super::generic::epf::epf_row::<STEP>,
        )
    }
}

pub use super::generic::apply_gabor_like;
