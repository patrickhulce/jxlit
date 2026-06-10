//! Inverse-DCT stage of the VarDCT path.
//!
//! Consumes the dequantized coefficients from [`super::dequant`] and applies the
//! per-varblock inverse transform (adding back the LF coefficients), writing
//! pixel-domain XYB samples in place. Delegates to the vendored `jxl-render`
//! `transform_with_lf_grouped`.

use std::collections::HashMap;

use jxl_grid::MutableSubgrid;
use jxl_modular::Sample;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::LfGroup;
use crate::vendor::jxl_render::{ImageWithRegion, vardct};

/// Applies the inverse DCT to a single tile, combining the dequantized HF
/// coefficients with the LF image.
pub fn run_inverse_dct<S: Sample>(
    low_frequency_image: &ImageWithRegion,
    xyb_coefficients: &mut [MutableSubgrid<'_, f32>; 3],
    group_index: u32,
    frame_header: &FrameHeader,
    low_frequency_groups: &HashMap<u32, LfGroup<S>>,
) {
    vardct::transform_with_lf_grouped(
        low_frequency_image,
        xyb_coefficients,
        group_index,
        frame_header,
        low_frequency_groups,
    );
}
