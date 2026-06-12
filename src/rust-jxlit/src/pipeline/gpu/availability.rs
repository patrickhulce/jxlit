//! Per-kernel GPU availability predicates.
//!
//! Each predicate decides whether a particular kernel can run on the GPU for
//! the given inputs, decode options, and environment. All default to `false`
//! until individual kernels are implemented.

#![allow(clippy::too_many_arguments)]
#![allow(unused_variables)]

use std::collections::HashMap;

#[cfg(feature = "gpu")]
use crate::types::Hardware;
use crate::types::{DecodeOptions, PixelLayout};
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
#[cfg(feature = "gpu")]
use crate::vendor::jxl_frame::header::BlendMode as FrameBlendMode;
use crate::vendor::jxl_render::{IndexedFrame, Reference, Region, RenderContext};
use crate::vendor::jxl_vardct::LfChannelCorrelation;
use jxl_grid::SharedSubgrid;
use jxl_image::{BitDepth, ExtraChannelType, ImageHeader};
use jxl_modular::Sample;

use super::device::{DeviceCoefficients, DeviceImage};
use super::environment::GpuEnvironment;

pub fn read_pass_group_available(
    _frame_header: &FrameHeader,
    _group_idx: u32,
    _pass_idx: u32,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn run_high_frequency_dequant_available<S: Sample>(
    _xyb_coefficients: &DeviceCoefficients<'_>,
    _group_index: u32,
    _image_header: &ImageHeader,
    _frame_header: &FrameHeader,
    _low_frequency_global: &LfGlobal<S>,
    _low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    _high_frequency_global: &HfGlobal,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn run_chroma_from_luma_high_frequency_available(
    _xyb_coefficients: &DeviceCoefficients<'_>,
    _x_from_y: &SharedSubgrid<i32>,
    _b_from_y: &SharedSubgrid<i32>,
    _lf_chan_corr: &LfChannelCorrelation,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn run_inverse_dct_available<S: Sample>(
    _low_frequency_image: &DeviceImage,
    _xyb_coefficients: &DeviceCoefficients<'_>,
    _group_index: u32,
    _frame_header: &FrameHeader,
    _low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn run_low_frequency_dequant_available(
    _low_frequency_image: &DeviceImage,
    _subsampled: bool,
    _skip_adaptive_lf_smoothing: bool,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn build_low_frequency_image_available(
    _frame_header: &FrameHeader,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn run_loop_filters_available<S: Sample>(
    _frame: &IndexedFrame,
    _fb: &DeviceImage,
    _color_padded_region: Region,
    _low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn run_features_available<S: Sample>(
    _frame: &IndexedFrame,
    _grid: &DeviceImage,
    _upsampling_valid_region: Region,
    _reference_grids: [Option<Reference<S>>; 4],
    _low_frequency_global: Option<&LfGlobal<S>>,
    _visible_frames_num: usize,
    _invisible_frames_num: usize,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn run_jpeg_upsample_available(
    _fb: &DeviceImage,
    _color_padded_region: Region,
    _bit_depth: BitDepth,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}

pub fn run_nonseparable_upsample_available(
    _fb: &DeviceImage,
    _image_header: &ImageHeader,
    frame_header: &FrameHeader,
    _region: Region,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> bool {
    #[cfg(feature = "gpu")]
    {
        gpu_hardware_available(options, env) && frame_header.upsampling != 1
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (frame_header, options, env);
        false
    }
}

pub fn run_color_for_record_available(
    image_header: &ImageHeader,
    do_ycbcr: bool,
    _fb: &DeviceImage,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> bool {
    #[cfg(feature = "gpu")]
    {
        if !gpu_hardware_available(options, env) {
            return false;
        }
        if do_ycbcr {
            return true;
        }
        let metadata = &image_header.metadata;
        if !metadata.xyb_encoded {
            return false;
        }
        super::color_transform::build_record_gpu_plan(image_header).is_ok()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (image_header, do_ycbcr, options, env);
        false
    }
}

pub fn run_blend_available(
    ctx: &RenderContext,
    idx: usize,
    _grid: &DeviceImage,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> bool {
    #[cfg(feature = "gpu")]
    {
        if !gpu_hardware_available(options, env) {
            return false;
        }
        let frame = &ctx.frames()[idx];
        let frame_header = frame.header();
        if !frame_header.frame_type.is_normal_frame() {
            return false;
        }
        if !frame_header.resets_canvas && !frame_header.is_last {
            return false;
        }
        if frame_header.can_reference() {
            return false;
        }
        frame_header.blending_info.mode == FrameBlendMode::Replace
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (ctx, idx, options, env);
        false
    }
}

pub fn run_xyb2rgb_available(
    ctx: &RenderContext,
    frame: &IndexedFrame,
    grid: &DeviceImage,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> bool {
    #[cfg(feature = "gpu")]
    {
        gpu_hardware_available(options, env)
            && !grid.ct_done()
            && frame.header().encoded_color_channels() == 3
            && super::color_transform::build_gpu_plan(ctx).is_ok()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (ctx, frame, grid, options, env);
        false
    }
}

pub fn fuse_spot_colors_available(
    image: &DeviceImage,
    _color_bit_depth: BitDepth,
    extra_channels: &[(ExtraChannelType, BitDepth)],
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> bool {
    #[cfg(feature = "gpu")]
    {
        gpu_hardware_available(options, env)
            && image.color_channels() == 3
            && extra_channels
                .iter()
                .any(|(ec, _)| matches!(ec, ExtraChannelType::SpotColour { .. }))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (image, extra_channels, options, env);
        false
    }
}

pub fn run_interleave_available(
    _image: &DeviceImage,
    _orientation: u32,
    _width: u32,
    _height: u32,
    _channels: usize,
    layout: PixelLayout,
    options: &DecodeOptions,
    env: GpuEnvironment,
    has_spot_colors: bool,
    has_float_sample: bool,
) -> bool {
    gpu_export_base_available(layout, PixelLayout::Interleaved, options, env)
        && !has_spot_colors
        && !has_float_sample
}

pub fn run_export_planar_available(
    _image: &DeviceImage,
    _orientation: u32,
    _width: u32,
    _height: u32,
    _channels: usize,
    layout: PixelLayout,
    options: &DecodeOptions,
    env: GpuEnvironment,
    has_spot_colors: bool,
    has_float_sample: bool,
) -> bool {
    gpu_export_base_available(layout, PixelLayout::Planar, options, env)
        && !has_spot_colors
        && !has_float_sample
}

#[cfg(feature = "gpu")]
fn gpu_hardware_available(options: &DecodeOptions, env: GpuEnvironment) -> bool {
    options.hardware == Hardware::Gpu
        && env.device_available
        && super::context::GpuContext::get().is_some()
}

#[cfg(not(feature = "gpu"))]
fn gpu_hardware_available(_options: &DecodeOptions, _env: GpuEnvironment) -> bool {
    false
}

#[cfg(feature = "gpu")]
fn gpu_export_base_available(
    layout: PixelLayout,
    required: PixelLayout,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> bool {
    gpu_hardware_available(options, env) && layout == required
}

#[cfg(not(feature = "gpu"))]
fn gpu_export_base_available(
    _layout: PixelLayout,
    _required: PixelLayout,
    _options: &DecodeOptions,
    _env: GpuEnvironment,
) -> bool {
    false
}
