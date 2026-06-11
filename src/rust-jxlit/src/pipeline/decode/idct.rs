//! Inverse-DCT stage of the VarDCT path.
//!
//! Consumes the dequantized coefficients from [`super::dequant`] and applies the
//! per-varblock inverse transform (adding back the LF coefficients), writing
//! pixel-domain XYB samples in place. Delegates to the vendored `jxl-render`
//! `transform_with_lf_grouped`.

use std::collections::HashMap;

use jxl_modular::Sample;

use crate::pipeline::gpu::{DeviceCoefficients, DeviceImage, kernels};
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::LfGroup;
use crate::vendor::jxl_render::{ImageWithRegion, vardct};

/// Applies the inverse DCT to a single tile, combining the dequantized HF
/// coefficients with the LF image.
pub fn run_inverse_dct<S: Sample>(
    low_frequency_image: &DeviceImage,
    xyb_coefficients: &mut DeviceCoefficients<'_>,
    group_index: u32,
    frame_header: &FrameHeader,
    low_frequency_groups: &HashMap<u32, LfGroup<S>>,
) {
    match (low_frequency_image, xyb_coefficients) {
        (DeviceImage::Cpu(lf_image), DeviceCoefficients::Cpu(coeffs)) => {
            vardct::transform_with_lf_grouped(
                lf_image,
                coeffs,
                group_index,
                frame_header,
                low_frequency_groups,
            );
        }
        (lf_image, coeffs) => {
            kernels::run_inverse_dct_on_gpu(
                lf_image,
                coeffs,
                group_index,
                frame_header,
                low_frequency_groups,
            );
        }
    }
}

/// CPU-only inverse DCT used when LF image is still a plain [`ImageWithRegion`].
#[allow(dead_code)]
pub fn run_inverse_dct_cpu<S: Sample>(
    low_frequency_image: &ImageWithRegion,
    xyb_coefficients: &mut [jxl_grid::MutableSubgrid<'_, f32>; 3],
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
