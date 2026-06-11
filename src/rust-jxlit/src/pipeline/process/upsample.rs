//! Upsampling post-decode steps: JPEG (chroma) upsampling and non-separable
//! color upsampling. Both delegate to the vendored `ImageWithRegion` methods.

use jxl_image::{BitDepth, ImageHeader};

use crate::pipeline::gpu::{DeviceImage, kernels};
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_render::{Region, Result};

/// Applies in-place JPEG (YCbCr) chroma upsampling to the color buffer.
pub fn run_jpeg_upsample(
    fb: &mut DeviceImage,
    color_padded_region: Region,
    bit_depth: BitDepth,
) -> Result<()> {
    match fb {
        DeviceImage::Cpu(image) => image.upsample_jpeg(color_padded_region, bit_depth),
        DeviceImage::Gpu(_) => {
            kernels::run_jpeg_upsample_on_gpu(fb, color_padded_region, bit_depth)
        }
    }
}

/// Applies in-place non-separable color upsampling to the color buffer.
pub fn run_nonseparable_upsample(
    fb: &mut DeviceImage,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    region: Region,
) -> Result<()> {
    match fb {
        DeviceImage::Cpu(image) => {
            image.upsample_nonseparable(image_header, frame_header, region, false)
        }
        DeviceImage::Gpu(_) => {
            kernels::run_nonseparable_upsample_on_gpu(fb, image_header, frame_header, region)
        }
    }
}
