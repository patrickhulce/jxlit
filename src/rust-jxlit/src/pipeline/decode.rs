//! Per-tile (JXL pass-group) decode slices: the future GPU offload target.
//!
//! `entropy` reads the per-tile ANS coefficients; `dequant` and `idct` run the
//! numeric dequantization and inverse transform. All numeric work delegates to
//! the vendored jxl-oxide VarDCT routines.

pub mod dequant;
pub mod entropy;
pub mod idct;
