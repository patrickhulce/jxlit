//! Inverse-DCT stage of the VarDCT path.
//!
//! Consumes the dequantized coefficients from [`super::dequant`] and applies the
//! per-varblock inverse transform (adding back the LF coefficients), writing
//! pixel-domain XYB samples in place. Delegates to the vendored `jxl-render`
//! `transform_with_lf_grouped` (which dispatches to the SIMD/generic
//! `transform_varblocks` implementations).

use std::collections::HashMap;

use jxl_grid::MutableSubgrid;
use jxl_modular::Sample;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::LfGroup;
use crate::vendor::jxl_render::{ImageWithRegion, vardct};

/// Applies the inverse DCT to a single group, combining the dequantized HF
/// coefficients with the LF image.
pub(crate) fn transform_group<S: Sample>(
    lf_xyb: &ImageWithRegion,
    grid_xyb: &mut [MutableSubgrid<'_, f32>; 3],
    group_idx: u32,
    frame_header: &FrameHeader,
    lf_groups: &HashMap<u32, LfGroup<S>>,
) {
    vardct::transform_with_lf_grouped(lf_xyb, grid_xyb, group_idx, frame_header, lf_groups);
}
