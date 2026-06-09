// Vendored from jxl-oxide (jxl-render 0.12.4), (c) Wonwoo Choi, licensed MIT OR Apache-2.0.
// Source: https://github.com/tirr-c/jxl-oxide/blob/f8ae722ef2d6b782941c89517d19cfbf605c4a9d/crates/jxl-render/src/filter/impls/x86_64/gabor_avx2.rs
// Copied as-is; only crate-path references changed.

use crate::vendor::jxl_render::filter::gabor::GaborRow;

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub(super) unsafe fn run_gabor_row_x86_64_avx2(row: GaborRow) {
    super::super::generic::gabor::run_gabor_row_generic(row)
}
