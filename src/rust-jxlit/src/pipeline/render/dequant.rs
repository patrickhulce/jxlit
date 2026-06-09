//! Dequantization stage of the VarDCT path (formerly `vardct.rs`).
//!
//! Turns the quantized coefficients produced by [`super::entropy`] into
//! dequantized frequency-domain coefficients, ready for the inverse DCT in
//! [`super::idct`]. All numeric work is delegated to the vendored `jxl-render`
//! VarDCT routines (which in turn use the vendored `jxl-vardct` dequant
//! matrices / quantizer).

use std::collections::HashMap;

use jxl_grid::{MutableSubgrid, SharedSubgrid};
use jxl_image::ImageHeader;
use jxl_modular::Sample;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use crate::vendor::jxl_render::{ImageWithRegion, Result, vardct};
use crate::vendor::jxl_vardct::{LfChannelCorrelation, LfChannelDequantization, Quantizer};

/// LF-side preparation: chroma-from-luma on the LF image and adaptive LF
/// smoothing (mirrors the LF branch of `render_vardct`).
pub(crate) fn prepare_lf(
    lf_xyb: &mut ImageWithRegion,
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    lf_chan_corr: &LfChannelCorrelation,
    subsampled: bool,
    skip_adaptive_lf_smoothing: bool,
) -> Result<()> {
    if !subsampled {
        vardct::chroma_from_luma_lf(lf_xyb.as_color_floats_mut(), lf_chan_corr);
    }
    if !skip_adaptive_lf_smoothing {
        vardct::adaptive_lf_smoothing(lf_xyb.as_color_floats_mut(), lf_dequant, quantizer)?;
    }
    Ok(())
}

/// Per-group HF dequantization of a varblock coefficient grid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_group<S: Sample>(
    grid_xyb: &mut [MutableSubgrid<'_, f32>; 3],
    group_idx: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal<S>,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    hf_global: &HfGlobal,
) {
    vardct::dequant_hf_varblock_grouped(
        grid_xyb,
        group_idx,
        image_header,
        frame_header,
        lf_global,
        lf_groups,
        hf_global,
    );
}

/// Per-group HF chroma-from-luma correction.
pub(crate) fn chroma_from_luma_hf(
    grid_xyb: &mut [MutableSubgrid<'_, f32>; 3],
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
) {
    vardct::chroma_from_luma_hf_grouped(grid_xyb, x_from_y, b_from_y, lf_chan_corr);
}
