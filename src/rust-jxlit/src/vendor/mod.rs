//! Vendored jxl-oxide sub-crates (jxl-render 0.12.4, jxl-frame 0.13.3,
//! jxl-vardct 0.11.1, jxl-color 0.11.0), copied verbatim from
//! <https://github.com/tirr-c/jxl-oxide/tree/f8ae722ef2d6b782941c89517d19cfbf605c4a9d>
//! with only crate-path references rewritten to point at the vendored copies.
//! jxl-color lives in `src/vendor/jxl-color/` as a path dependency (GPU `gpu_ops` patch).
//!
//! These are kept in-tree so the staged decode pipeline can call their building
//! blocks directly (entropy decode, dequant, inverse DCT, color transform) while
//! the renderer is forked stage-by-stage.

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(mismatched_lifetime_syntaxes)]
#![allow(clippy::all)]

pub(crate) mod jxl_frame;
pub(crate) mod jxl_render;
pub(crate) mod jxl_vardct;
