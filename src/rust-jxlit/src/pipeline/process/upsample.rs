//! Upsampling post-decode steps: JPEG (chroma) upsampling and non-separable
//! color upsampling. Both delegate to the vendored `ImageWithRegion` methods.

use jxl_image::{BitDepth, ImageHeader};

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_render::{ImageWithRegion, Region, Result};

/// Applies in-place JPEG (YCbCr) chroma upsampling to the color buffer.
pub fn run_jpeg_upsample(
    fb: &mut ImageWithRegion,
    color_padded_region: Region,
    bit_depth: BitDepth,
) -> Result<()> {
    fb.upsample_jpeg(color_padded_region, bit_depth)
}

/// Applies in-place non-separable color upsampling to the color buffer.
pub fn run_nonseparable_upsample(
    fb: &mut ImageWithRegion,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    region: Region,
) -> Result<()> {
    fb.upsample_nonseparable(image_header, frame_header, region, false)
}
