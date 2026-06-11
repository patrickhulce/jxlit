//! Panicking GPU kernel placeholders for each forked pipeline step.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;

use jxl_grid::SharedSubgrid;
use jxl_image::{BitDepth, ImageHeader};
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use crate::vendor::jxl_render::{IndexedFrame, Reference, Region, RenderContext, Result};
use crate::vendor::jxl_vardct::LfChannelCorrelation;

use super::device::{DeviceCoefficients, DeviceImage};
use super::image::GpuImageWithRegion;

macro_rules! gpu_unimplemented {
    ($name:literal) => {
        unimplemented!(concat!("GPU path not implemented: ", $name))
    };
}

pub fn read_pass_group_on_gpu(_group_idx: u32, _pass_idx: u32) {
    gpu_unimplemented!("read_pass_group");
}

pub fn run_high_frequency_dequant_on_gpu<S: Sample>(
    _xyb_coefficients: &mut DeviceCoefficients<'_>,
    _group_index: u32,
    _image_header: &ImageHeader,
    _frame_header: &FrameHeader,
    _low_frequency_global: &LfGlobal<S>,
    _low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    _high_frequency_global: &HfGlobal,
) {
    gpu_unimplemented!("run_high_frequency_dequant");
}

pub fn run_chroma_from_luma_high_frequency_on_gpu(
    _xyb_coefficients: &mut DeviceCoefficients<'_>,
    _x_from_y: &SharedSubgrid<i32>,
    _b_from_y: &SharedSubgrid<i32>,
    _lf_chan_corr: &LfChannelCorrelation,
) {
    gpu_unimplemented!("run_chroma_from_luma_high_frequency");
}

pub fn run_inverse_dct_on_gpu<S: Sample>(
    _low_frequency_image: &DeviceImage,
    _xyb_coefficients: &mut DeviceCoefficients<'_>,
    _group_index: u32,
    _frame_header: &FrameHeader,
    _low_frequency_groups: &HashMap<u32, LfGroup<S>>,
) {
    gpu_unimplemented!("run_inverse_dct");
}

pub fn run_loop_filters_on_gpu<S: Sample>(
    _frame: &IndexedFrame,
    _fb: &mut DeviceImage,
    _color_padded_region: Region,
    _low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    _pool: &JxlThreadPool,
) -> Result<()> {
    gpu_unimplemented!("run_loop_filters");
}

pub fn run_features_on_gpu<S: Sample>(
    _frame: &IndexedFrame,
    _grid: &mut DeviceImage,
    _upsampling_valid_region: Region,
    _reference_grids: [Option<Reference<S>>; 4],
    _low_frequency_global: Option<&LfGlobal<S>>,
    _visible_frames_num: usize,
    _invisible_frames_num: usize,
    _pool: &JxlThreadPool,
) -> Result<()> {
    gpu_unimplemented!("run_features");
}

pub fn run_jpeg_upsample_on_gpu(
    _fb: &mut DeviceImage,
    _color_padded_region: Region,
    _bit_depth: BitDepth,
) -> Result<()> {
    gpu_unimplemented!("run_jpeg_upsample");
}

pub fn run_nonseparable_upsample_on_gpu(
    _fb: &mut DeviceImage,
    _image_header: &ImageHeader,
    _frame_header: &FrameHeader,
    _region: Region,
) -> Result<()> {
    gpu_unimplemented!("run_nonseparable_upsample");
}

pub fn run_color_for_record_on_gpu(
    _image_header: &ImageHeader,
    _do_ycbcr: bool,
    _fb: &mut DeviceImage,
    _pool: &JxlThreadPool,
) -> Result<()> {
    gpu_unimplemented!("run_color_for_record");
}

pub fn run_blend_on_gpu(
    _ctx: &RenderContext,
    _idx: usize,
    _grid: DeviceImage,
) -> Result<Arc<DeviceImage>> {
    gpu_unimplemented!("run_blend");
}

pub fn run_xyb2rgb_on_gpu(
    _ctx: &RenderContext,
    _frame: &IndexedFrame,
    _grid: Arc<DeviceImage>,
) -> Result<Arc<DeviceImage>> {
    gpu_unimplemented!("run_xyb2rgb");
}

pub fn run_interleave_on_gpu(
    _pixels: &mut [f32],
    _image: &DeviceImage,
    _bit_depth: &[BitDepth],
    _start_offset_xy: &[(i32, i32)],
    _orientation: u32,
    _width: u32,
    _height: u32,
    _channels: usize,
) -> usize {
    gpu_unimplemented!("run_interleave");
}

pub fn run_export_planar_on_gpu(
    _pixels: &mut [f32],
    _image: &DeviceImage,
    _bit_depth: &[BitDepth],
    _start_offset_xy: &[(i32, i32)],
    _orientation: u32,
    _width: u32,
    _height: u32,
    _channels: usize,
    _plane_size: usize,
) -> usize {
    gpu_unimplemented!("run_export_planar");
}

pub fn build_low_frequency_image_on_gpu(_low_frequency_image: GpuImageWithRegion) -> DeviceImage {
    gpu_unimplemented!("build_low_frequency_image");
}

pub fn fuse_spot_colors_on_gpu(
    _image: Arc<DeviceImage>,
    _color_bit_depth: BitDepth,
    _extra_channels: &[(jxl_image::ExtraChannelType, BitDepth)],
) -> Result<(Arc<DeviceImage>, bool)> {
    gpu_unimplemented!("fuse_spot_colors");
}
