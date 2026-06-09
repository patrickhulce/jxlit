// Vendored from jxl-oxide (jxl-vardct 0.11.1), (c) Wonwoo Choi, licensed MIT OR Apache-2.0.
// Source: https://github.com/tirr-c/jxl-oxide/blob/f8ae722ef2d6b782941c89517d19cfbf605c4a9d/crates/jxl-vardct/src/lib.rs
// Copied as-is; only crate-path references changed.

//! This crate provides types related to representation of VarDCT frames, such as
//! [varblock transform types][TransformType], [LF images][LfCoeff],
//! [dequantization matrices][DequantMatrixSet] and [HF coefficients][write_hf_coeff].
//!
//! Actual decoding (dequantization and rendering) of such frames is not done in this crate.
mod dct_select;
mod dequant;
mod error;
mod hf_coeff;
mod hf_metadata;
mod hf_pass;
mod lf;

pub use dct_select::TransformType;
pub use dequant::*;
pub use error::{Error, Result};
pub use hf_coeff::*;
pub use hf_metadata::*;
pub use hf_pass::*;
pub use lf::*;
