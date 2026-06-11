//! Post-decode processing steps on XYB / RGB data.
//!
//! `filters` (loop filters), `upsample`, `features` and `xyb2rgb` (color
//! transform) run after the per-frame decode; `interleave` packs the planar
//! result into the final HWC `DecodedImage`.

pub mod export;
pub mod features;
pub mod filters;
pub mod interleave;
pub mod upsample;
pub mod xyb2rgb;
