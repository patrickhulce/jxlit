// Vendored from jxl-oxide (jxl-render 0.12.4), (c) Wonwoo Choi, licensed MIT OR Apache-2.0.
// Source: https://github.com/tirr-c/jxl-oxide/blob/f8ae722ef2d6b782941c89517d19cfbf605c4a9d/crates/jxl-render/src/features/mod.rs
// Copied as-is; only crate-path references changed.

mod noise;
mod spline;
mod spot_colors;
mod upsampling;

pub use noise::{render_noise, synthesize_noise};
pub use spline::render_spline;
pub use spot_colors::render_spot_color;
pub use upsampling::upsample;
