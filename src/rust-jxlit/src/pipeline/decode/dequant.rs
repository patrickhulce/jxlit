//! Dequantization stage of the VarDCT path.
//!
//! Turns quantized coefficients into dequantized frequency-domain coefficients,
//! ready for the inverse DCT in [`super::idct`]. Includes both the frame-global
//! low-frequency preparation and the per-tile high-frequency dequant; all
//! numeric work delegates to the vendored `jxl-render` VarDCT routines.

use std::collections::HashMap;

use jxl_grid::{MutableSubgrid, SharedSubgrid};
use jxl_image::ImageHeader;
use jxl_modular::Sample;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use crate::vendor::jxl_render::{ImageWithRegion, Result, vardct};
use crate::vendor::jxl_vardct::{LfChannelCorrelation, LfChannelDequantization, Quantizer};

/// Frame-global low-frequency preparation: chroma-from-luma on the LF image and
/// adaptive LF smoothing (mirrors the LF branch of `render_vardct`).
pub fn run_low_frequency_dequant(
    low_frequency_image: &mut ImageWithRegion,
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    lf_chan_corr: &LfChannelCorrelation,
    subsampled: bool,
    skip_adaptive_lf_smoothing: bool,
) -> Result<()> {
    if !subsampled {
        vardct::chroma_from_luma_lf(low_frequency_image.as_color_floats_mut(), lf_chan_corr);
    }
    if !skip_adaptive_lf_smoothing {
        vardct::adaptive_lf_smoothing(
            low_frequency_image.as_color_floats_mut(),
            lf_dequant,
            quantizer,
        )?;
    }
    Ok(())
}

/// Per-tile high-frequency dequantization of a varblock coefficient grid.
pub fn run_high_frequency_dequant<S: Sample>(
    xyb_coefficients: &mut [MutableSubgrid<'_, f32>; 3],
    group_index: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    low_frequency_global: &LfGlobal<S>,
    low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    high_frequency_global: &HfGlobal,
) {
    vardct::dequant_hf_varblock_grouped(
        xyb_coefficients,
        group_index,
        image_header,
        frame_header,
        low_frequency_global,
        low_frequency_groups,
        high_frequency_global,
    );
}

/// Per-tile high-frequency chroma-from-luma correction.
pub fn run_chroma_from_luma_high_frequency(
    xyb_coefficients: &mut [MutableSubgrid<'_, f32>; 3],
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
) {
    vardct::chroma_from_luma_hf_grouped(xyb_coefficients, x_from_y, b_from_y, lf_chan_corr);
}
