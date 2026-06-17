//! Upsampling post-decode steps: JPEG (chroma) upsampling and non-separable
//! color upsampling. Both delegate to the vendored `ImageWithRegion` methods.

use jxl_image::{BitDepth, ImageHeader};

use crate::pipeline::gpu::{DeviceImage, GpuEnvironment, availability, kernels};
use crate::types::DecodeOptions;
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_render::{Region, Result};

/// Applies in-place JPEG (YCbCr) chroma upsampling to the color buffer.
pub fn run_jpeg_upsample(
    fb: &mut DeviceImage,
    color_padded_region: Region,
    bit_depth: BitDepth,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<()> {
    if availability::run_jpeg_upsample_available(fb, color_padded_region, bit_depth, options, env) {
        return kernels::run_jpeg_upsample_on_gpu(fb, color_padded_region, bit_depth);
    }

    let image = fb
        .ensure_cpu()
        .expect("image must be CPU-resident when JPEG upsample GPU kernel is unavailable");
    image.upsample_jpeg(color_padded_region, bit_depth)
}

/// Applies in-place non-separable color upsampling to the color buffer.
pub fn run_nonseparable_upsample(
    fb: &mut DeviceImage,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    region: Region,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<()> {
    if !availability::nonseparable_upsample_needed(fb, frame_header, false) {
        return Ok(());
    }

    if availability::run_nonseparable_upsample_available(
        fb,
        image_header,
        frame_header,
        region,
        options,
        env,
    ) {
        return kernels::run_nonseparable_upsample_on_gpu(fb, image_header, frame_header, region);
    }

    let image = fb
        .ensure_cpu()
        .expect("image must be CPU-resident when non-separable upsample GPU kernel is unavailable");
    image.upsample_nonseparable(image_header, frame_header, region, false)
}
