//! Dequantization stage of the VarDCT path.
//!
//! Turns quantized coefficients into dequantized frequency-domain coefficients,
//! ready for the inverse DCT in [`super::idct`]. Includes both the frame-global
//! low-frequency preparation and the per-tile high-frequency dequant; all
//! numeric work delegates to the vendored `jxl-render` VarDCT routines.

use std::collections::HashMap;

use jxl_grid::SharedSubgrid;
use jxl_image::ImageHeader;
use jxl_modular::Sample;

use crate::pipeline::gpu::{DeviceCoefficients, DeviceImage, kernels};
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use crate::vendor::jxl_render::{ImageWithRegion, Result, vardct};
use crate::vendor::jxl_vardct::{LfChannelCorrelation, LfChannelDequantization, Quantizer};

/// Frame-global low-frequency preparation on a device-resident LF image.
#[allow(dead_code)]
pub fn run_low_frequency_dequant(
    low_frequency_image: &mut DeviceImage,
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    lf_chan_corr: &LfChannelCorrelation,
    subsampled: bool,
    skip_adaptive_lf_smoothing: bool,
) -> Result<()> {
    match low_frequency_image {
        DeviceImage::Cpu(image) => {
            if !subsampled {
                vardct::chroma_from_luma_lf(image.as_color_floats_mut(), lf_chan_corr);
            }
            if !skip_adaptive_lf_smoothing {
                vardct::adaptive_lf_smoothing(image.as_color_floats_mut(), lf_dequant, quantizer)?;
            }
            Ok(())
        }
        DeviceImage::Gpu(_) => {
            unimplemented!("GPU path not implemented: run_low_frequency_dequant")
        }
    }
}

/// Per-tile high-frequency dequantization of a varblock coefficient grid.
pub fn run_high_frequency_dequant<S: Sample>(
    xyb_coefficients: &mut DeviceCoefficients<'_>,
    group_index: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    low_frequency_global: &LfGlobal<S>,
    low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    high_frequency_global: &HfGlobal,
) {
    match xyb_coefficients {
        DeviceCoefficients::Cpu(coeffs) => {
            vardct::dequant_hf_varblock_grouped(
                coeffs,
                group_index,
                image_header,
                frame_header,
                low_frequency_global,
                low_frequency_groups,
                high_frequency_global,
            );
        }
        DeviceCoefficients::Gpu(_) => {
            kernels::run_high_frequency_dequant_on_gpu(
                xyb_coefficients,
                group_index,
                image_header,
                frame_header,
                low_frequency_global,
                low_frequency_groups,
                high_frequency_global,
            );
        }
    }
}

/// Per-tile high-frequency chroma-from-luma correction.
pub fn run_chroma_from_luma_high_frequency(
    xyb_coefficients: &mut DeviceCoefficients<'_>,
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
) {
    match xyb_coefficients {
        DeviceCoefficients::Cpu(coeffs) => {
            vardct::chroma_from_luma_hf_grouped(coeffs, x_from_y, b_from_y, lf_chan_corr);
        }
        DeviceCoefficients::Gpu(_) => {
            kernels::run_chroma_from_luma_high_frequency_on_gpu(
                xyb_coefficients,
                x_from_y,
                b_from_y,
                lf_chan_corr,
            );
        }
    }
}

/// CPU-only LF dequant for the parse stage before device wrapping.
pub fn run_low_frequency_dequant_cpu(
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
