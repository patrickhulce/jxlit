//! GPU compute kernels for pipeline steps.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "gpu")]
use std::sync::OnceLock;

use jxl_grid::SharedSubgrid;
use jxl_image::{BitDepth, ImageHeader};
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::types::PixelLayout;
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use crate::vendor::jxl_render::{IndexedFrame, Reference, Region, RenderContext, Result};
use crate::vendor::jxl_vardct::LfChannelCorrelation;

use super::context::GpuPixelBuffer;
#[cfg(feature = "gpu")]
use super::context::GpuContext;
use super::device::{DeviceCoefficients, DeviceImage};
use super::image::GpuImageWithRegion;
#[cfg(feature = "gpu")]
use super::image::sample_kind_bits;

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

/// GPU-only export: reads GPU-resident channel buffers and writes interleaved HWC output.
pub fn run_interleave_on_gpu(
    image: &GpuImageWithRegion,
    channel_indices: &[usize],
    bit_depth: &[BitDepth],
    start_offset_xy: &[(i32, i32)],
    orientation: u32,
    width: u32,
    height: u32,
    channels: usize,
) -> std::result::Result<GpuPixelBuffer, String> {
    run_export_on_gpu(
        image,
        channel_indices,
        bit_depth,
        start_offset_xy,
        orientation,
        width,
        height,
        channels,
        PixelLayout::Interleaved,
    )
}

/// GPU-only export: reads GPU-resident channel buffers and writes planar CHW output.
pub fn run_export_planar_on_gpu(
    image: &GpuImageWithRegion,
    channel_indices: &[usize],
    bit_depth: &[BitDepth],
    start_offset_xy: &[(i32, i32)],
    orientation: u32,
    width: u32,
    height: u32,
    channels: usize,
    plane_size: usize,
) -> std::result::Result<GpuPixelBuffer, String> {
    let _ = plane_size;
    run_export_on_gpu(
        image,
        channel_indices,
        bit_depth,
        start_offset_xy,
        orientation,
        width,
        height,
        channels,
        PixelLayout::Planar,
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExportParams {
    orientation: u32,
    out_width: u32,
    out_height: u32,
    channels: u32,
    pixel_layout: u32,
    plane_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChannelMeta {
    offset_x: i32,
    offset_y: i32,
    grid_width: u32,
    grid_height: u32,
    sample_kind: u32,
    bits_per_sample: u32,
    base_u32: u32,
}

#[cfg(feature = "gpu")]
struct ExportPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

#[cfg(feature = "gpu")]
fn export_pipeline(ctx: &GpuContext) -> &'static ExportPipeline {
    static PIPELINE: OnceLock<Option<ExportPipeline>> = OnceLock::new();
    PIPELINE
        .get_or_init(|| build_export_pipeline(ctx).ok())
        .as_ref()
        .expect("failed to build export compute pipeline")
}

#[cfg(feature = "gpu")]
fn build_export_pipeline(ctx: &GpuContext) -> std::result::Result<ExportPipeline, String> {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxlit_export"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/export.wgsl").into()),
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jxlit_export_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jxlit_export_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxlit_export"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    Ok(ExportPipeline {
        pipeline,
        bind_group_layout,
    })
}

#[cfg(feature = "gpu")]
fn channel_byte_len(grid: &super::image::GpuImageBuffer) -> u64 {
    let pixels = (grid.width() * grid.height()) as u64;
    match grid.sample_kind() {
        super::image::GpuSampleKind::F32 | super::image::GpuSampleKind::I32 => pixels * 4,
        super::image::GpuSampleKind::I16 => pixels * 2,
    }
}

#[cfg(feature = "gpu")]
fn align_bytes(len: u64) -> u64 {
    (len + 3) & !3
}

fn run_export_on_gpu(
    image: &GpuImageWithRegion,
    channel_indices: &[usize],
    bit_depth: &[BitDepth],
    start_offset_xy: &[(i32, i32)],
    orientation: u32,
    width: u32,
    height: u32,
    channels: usize,
    layout: PixelLayout,
) -> std::result::Result<GpuPixelBuffer, String> {
    #[cfg(feature = "gpu")]
    {
        if channels > 8 {
            return Err(format!("GPU export supports at most 8 channels, got {channels}"));
        }

        let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
        let export = export_pipeline(ctx);
        let plane_size = (width as usize) * (height as usize);
        let out_len = plane_size * channels;
        let layout_flag = match layout {
            PixelLayout::Interleaved => 0,
            PixelLayout::Planar => 1,
        };

        let params = ExportParams {
            orientation,
            out_width: width,
            out_height: height,
            channels: channels as u32,
            pixel_layout: layout_flag,
            plane_size: plane_size as u32,
        };
        let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxlit_export_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mut metas = [ChannelMeta {
            offset_x: 0,
            offset_y: 0,
            grid_width: 1,
            grid_height: 1,
            sample_kind: 0,
            bits_per_sample: 32,
            base_u32: 0,
        }; 8];
        let buffers = image.buffer();
        let mut packed_size = 0u64;
        let mut channel_copies: Vec<(&wgpu::Buffer, u64, u64)> = Vec::with_capacity(channels);
        for slot in 0..channels {
            let idx = channel_indices[slot];
            let grid = &buffers[idx];
            let (sample_kind, bits) = sample_kind_bits(grid.sample_kind(), bit_depth[slot]);
            let (offset_x, offset_y) = start_offset_xy[slot];
            let byte_len = channel_byte_len(grid);
            metas[slot] = ChannelMeta {
                offset_x,
                offset_y,
                grid_width: grid.width() as u32,
                grid_height: grid.height() as u32,
                sample_kind,
                bits_per_sample: bits,
                base_u32: (packed_size / 4) as u32,
            };
            channel_copies.push((grid.wgpu_buffer(), packed_size, byte_len));
            packed_size += align_bytes(byte_len);
        }

        let packed_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxlit_export_packed_channels"),
            size: packed_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let meta_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxlit_export_meta"),
            contents: bytemuck::cast_slice(&metas),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxlit_export_output"),
            size: (out_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxlit_export_bind_group"),
            layout: &export.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: packed_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxlit_export_encoder"),
            });
        for (src, dst_off, len) in channel_copies {
            encoder.copy_buffer_to_buffer(src, 0, &packed_buffer, dst_off, len);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxlit_export_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&export.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = width.div_ceil(8);
            let wg_y = height.div_ceil(8);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        Ok(GpuPixelBuffer::new(
            Arc::new(output_buffer),
            width,
            height,
            channels as u32,
            layout,
            out_len,
        ))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (
            image,
            channel_indices,
            bit_depth,
            start_offset_xy,
            orientation,
            width,
            height,
            channels,
            layout,
        );
        Err("GPU feature not enabled".to_string())
    }
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
