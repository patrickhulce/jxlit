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

use crate::pipeline::gpu::{
    DeviceCoefficients, DeviceImage, GpuEnvironment, availability, kernels,
};
use crate::types::DecodeOptions;
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use crate::vendor::jxl_render::{ImageWithRegion, Result, vardct};
use crate::vendor::jxl_vardct::{LfChannelCorrelation, LfChannelDequantization, Quantizer};

/// Frame-global low-frequency preparation on a device-resident LF image.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn run_low_frequency_dequant(
    low_frequency_image: &mut DeviceImage,
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    lf_chan_corr: &LfChannelCorrelation,
    subsampled: bool,
    skip_adaptive_lf_smoothing: bool,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<()> {
    if availability::run_low_frequency_dequant_available(
        low_frequency_image,
        subsampled,
        skip_adaptive_lf_smoothing,
        options,
        env,
    ) {
        unimplemented!("GPU path not implemented: run_low_frequency_dequant");
    }

    let image = low_frequency_image
        .ensure_cpu()
        .expect("LF image must be CPU-resident when LF dequant GPU kernel is unavailable");
    if !subsampled {
        vardct::chroma_from_luma_lf(image.as_color_floats_mut(), lf_chan_corr);
    }
    if !skip_adaptive_lf_smoothing {
        vardct::adaptive_lf_smoothing(image.as_color_floats_mut(), lf_dequant, quantizer)?;
    }
    Ok(())
}

/// Per-tile high-frequency dequantization of a varblock coefficient grid.
#[allow(clippy::too_many_arguments)]
pub fn run_high_frequency_dequant<S: Sample>(
    xyb_coefficients: &mut DeviceCoefficients<'_>,
    group_index: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    low_frequency_global: &LfGlobal<S>,
    low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    high_frequency_global: &HfGlobal,
    options: &DecodeOptions,
    env: GpuEnvironment,
) {
    if availability::run_high_frequency_dequant_available(
        xyb_coefficients,
        group_index,
        image_header,
        frame_header,
        low_frequency_global,
        low_frequency_groups,
        high_frequency_global,
        options,
        env,
    ) {
        kernels::run_high_frequency_dequant_on_gpu(
            xyb_coefficients,
            group_index,
            image_header,
            frame_header,
            low_frequency_global,
            low_frequency_groups,
            high_frequency_global,
        );
        return;
    }

    let coeffs = xyb_coefficients
        .ensure_cpu_mut()
        .expect("coefficients must be CPU-resident when HF dequant GPU kernel is unavailable");
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

/// Per-tile high-frequency chroma-from-luma correction.
pub fn run_chroma_from_luma_high_frequency(
    xyb_coefficients: &mut DeviceCoefficients<'_>,
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
    options: &DecodeOptions,
    env: GpuEnvironment,
) {
    if availability::run_chroma_from_luma_high_frequency_available(
        xyb_coefficients,
        x_from_y,
        b_from_y,
        lf_chan_corr,
        options,
        env,
    ) {
        kernels::run_chroma_from_luma_high_frequency_on_gpu(
            xyb_coefficients,
            x_from_y,
            b_from_y,
            lf_chan_corr,
        );
        return;
    }

    let coeffs = xyb_coefficients.ensure_cpu_mut().expect(
        "coefficients must be CPU-resident when chroma-from-luma HF GPU kernel is unavailable",
    );
    vardct::chroma_from_luma_hf_grouped(coeffs, x_from_y, b_from_y, lf_chan_corr);
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
