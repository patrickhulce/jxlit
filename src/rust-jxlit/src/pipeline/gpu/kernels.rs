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

#[cfg(feature = "gpu")]
use super::color_transform::{
    build_gpu_plan, build_record_gpu_plan, dispatch_invert_channels, dispatch_ycbcr_to_rgb,
    run_gpu_plan,
};
#[cfg(feature = "gpu")]
use super::context::GpuContext;
use super::context::GpuPixelBuffer;
#[cfg(feature = "gpu")]
use super::crop::dispatch_crop_f32;
#[cfg(feature = "gpu")]
use super::device::materialize_on_gpu;
use super::device::{DeviceCoefficients, DeviceImage};
use super::image::GpuImageWithRegion;
#[cfg(feature = "gpu")]
use super::image::sample_kind_bits;
#[cfg(feature = "gpu")]
use super::upsample::upsample_nonseparable;

macro_rules! gpu_unimplemented {
    ($name:literal) => {
        unimplemented!(concat!("GPU path not implemented: ", $name))
    };
}

#[cfg(feature = "gpu")]
fn gpu_error(msg: String) -> crate::vendor::jxl_render::Error {
    crate::vendor::jxl_render::Error::NotSupported(Box::leak(msg.into_boxed_str()))
}

#[cfg(feature = "gpu")]
fn unwrap_gpu_image(image: Arc<DeviceImage>) -> Result<GpuImageWithRegion> {
    match Arc::try_unwrap(image) {
        Ok(DeviceImage::Gpu(gpu)) => Ok(gpu),
        Ok(DeviceImage::Cpu(_)) => Err(gpu_error("expected GPU image".into())),
        Err(arc) => match arc.as_ref() {
            DeviceImage::Gpu(gpu) => gpu.try_clone().map_err(|e| gpu_error(format!("{e:?}"))),
            DeviceImage::Cpu(_) => Err(gpu_error("expected GPU image".into())),
        },
    }
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
    fb: &mut DeviceImage,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    region: Region,
) -> Result<()> {
    #[cfg(feature = "gpu")]
    {
        match fb {
            DeviceImage::Gpu(gpu) => {
                upsample_nonseparable(gpu, image_header, frame_header, region, false)
            }
            DeviceImage::Cpu(cpu) => {
                let mut gpu =
                    GpuImageWithRegion::from_cpu(cpu).map_err(|e| gpu_error(e.to_string()))?;
                upsample_nonseparable(&mut gpu, image_header, frame_header, region, false)?;
                *fb = DeviceImage::Gpu(gpu);
                Ok(())
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (fb, image_header, frame_header, region);
        gpu_unimplemented!("run_nonseparable_upsample");
    }
}

#[cfg(feature = "gpu")]
fn run_color_for_record_on_gpu_image(
    image_header: &ImageHeader,
    do_ycbcr: bool,
    gpu_image: &mut GpuImageWithRegion,
) -> Result<()> {
    let metadata = &image_header.metadata;
    let bit_depth = metadata.bit_depth;

    if do_ycbcr {
        let _ycbcr = crate::phase_guard!("color_for_record_gpu_ycbcr");
        gpu_image.convert_modular_color(bit_depth)?;
        dispatch_ycbcr_to_rgb(gpu_image).map_err(gpu_error)?;
        if metadata.colour_encoding.colour_space() == jxl_color::ColourSpace::Grey {
            gpu_image.remove_color_channels(1);
        }
        gpu_image.set_ct_done(true);
    } else if metadata.xyb_encoded {
        let plan = build_record_gpu_plan(image_header)
            .map_err(|_| gpu_error("record color transform unsupported on GPU".into()))?;
        gpu_image.convert_modular_color(bit_depth)?;
        let output_channels = {
            let _transform = crate::phase_guard!("color_for_record_gpu_transform");
            run_gpu_plan(&plan, gpu_image).map_err(gpu_error)?
        };
        if output_channels < 3 {
            gpu_image.remove_color_channels(output_channels);
        }
        gpu_image.set_ct_done(true);
    }
    Ok(())
}

pub fn run_color_for_record_on_gpu(
    image_header: &ImageHeader,
    do_ycbcr: bool,
    fb: &mut DeviceImage,
    _pool: &JxlThreadPool,
) -> Result<()> {
    #[cfg(feature = "gpu")]
    {
        match fb {
            DeviceImage::Gpu(gpu) => run_color_for_record_on_gpu_image(image_header, do_ycbcr, gpu),
            DeviceImage::Cpu(cpu) => {
                let mut gpu =
                    GpuImageWithRegion::from_cpu(cpu).map_err(|e| gpu_error(e.to_string()))?;
                run_color_for_record_on_gpu_image(image_header, do_ycbcr, &mut gpu)?;
                *fb = DeviceImage::Gpu(gpu);
                Ok(())
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (image_header, do_ycbcr, fb, _pool);
        gpu_unimplemented!("run_color_for_record");
    }
}

pub fn run_blend_on_gpu(
    ctx: &RenderContext,
    idx: usize,
    grid: DeviceImage,
) -> Result<Arc<DeviceImage>> {
    #[cfg(feature = "gpu")]
    {
        let frame = &ctx.frames()[idx];
        let frame_header = frame.header();
        let image_header = frame.image_header();

        let materialized = materialize_on_gpu(Arc::new(grid)).map_err(gpu_error)?;
        let mut gpu_image = unwrap_gpu_image(materialized)?;

        let oriented_image_region =
            crate::vendor::jxl_render::util::apply_orientation_to_image_region(
                image_header,
                ctx.requested_image_region(),
            );
        let mut frame_region = oriented_image_region
            .translate(-frame_header.x0, -frame_header.y0)
            .downsample(frame_header.lf_level * 3);
        frame_region = crate::vendor::jxl_render::util::pad_lf_region(frame_header, frame_region);
        frame_region = crate::vendor::jxl_render::util::pad_color_region(
            image_header,
            frame_header,
            frame_region,
        );
        frame_region = frame_region.upsample(frame_header.upsampling.ilog2());
        if frame_header.frame_type.is_normal_frame() {
            let full_image_region_in_frame =
                Region::with_size(image_header.size.width, image_header.size.height)
                    .translate(-frame_header.x0, -frame_header.y0);
            frame_region = frame_region.intersection(full_image_region_in_frame);
        }

        let channel_count = gpu_image.buffer().len();
        let mut crops = Vec::new();
        for channel_idx in 0..channel_count {
            let (region, _shift) = gpu_image.regions_and_shifts()[channel_idx];
            let left = frame_region.left.saturating_sub(region.left) as u32;
            let top = frame_region.top.saturating_sub(region.top) as u32;
            if left == 0
                && top == 0
                && frame_region.width == region.width
                && frame_region.height == region.height
            {
                continue;
            }
            let cropped = dispatch_crop_f32(
                &gpu_image.buffer()[channel_idx],
                left,
                top,
                frame_region.width,
                frame_region.height,
            )
            .map_err(gpu_error)?;
            crops.push((channel_idx, cropped));
        }
        for (channel_idx, cropped) in crops {
            gpu_image.buffer_mut()[channel_idx] = cropped;
            gpu_image.regions_mut()[channel_idx].0 = frame_region;
        }

        gpu_image.set_blend_done(true);
        Ok(Arc::new(DeviceImage::Gpu(gpu_image)))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (ctx, idx, grid);
        gpu_unimplemented!("run_blend");
    }
}

pub fn run_xyb2rgb_on_gpu(
    ctx: &RenderContext,
    frame: &IndexedFrame,
    grid: Arc<DeviceImage>,
) -> Result<Arc<DeviceImage>> {
    #[cfg(feature = "gpu")]
    {
        if grid.as_ref().ct_done() {
            let _materialize = crate::phase_guard!("xyb2rgb_gpu_materialize");
            return materialize_on_gpu(grid).map_err(gpu_error);
        }

        let frame_header = frame.header();
        let metadata = ctx.image_metadata();
        let bit_depth = metadata.bit_depth;
        let materialized = {
            let _materialize = crate::phase_guard!("xyb2rgb_gpu_materialize");
            materialize_on_gpu(grid).map_err(gpu_error)?
        };
        let mut working = {
            let _acquire = crate::phase_guard!("xyb2rgb_gpu_acquire");
            unwrap_gpu_image(materialized)?
        };

        if frame_header.do_ycbcr {
            let _ycbcr = crate::phase_guard!("xyb2rgb_gpu_ycbcr");
            working.convert_modular_color(bit_depth)?;
            dispatch_ycbcr_to_rgb(&mut working).map_err(gpu_error)?;
        }

        let transform_noop = ctx.color_transform_is_noop()?;
        if transform_noop && !frame_header.do_ycbcr {
            working.set_ct_done(true);
            return Ok(Arc::new(DeviceImage::Gpu(working)));
        }

        if transform_noop {
            let output_channels = ctx.color_transform_output_channels()?;
            working.remove_color_channels(output_channels);
            working.set_ct_done(true);
            return Ok(Arc::new(DeviceImage::Gpu(working)));
        }

        let encoded_color_channels = frame_header.encoded_color_channels();
        {
            let _prepare = crate::phase_guard!("xyb2rgb_gpu_prepare");
            if encoded_color_channels < 3 {
                working.clone_gray()?;
            }

            working.convert_modular_color(bit_depth)?;

            let mut has_black = false;
            for (ec_idx, ec_info) in metadata.ec_info.iter().enumerate() {
                if ec_info.is_black() {
                    let buf_idx = 3 + ec_idx;
                    working.convert_channel_to_float(buf_idx, ec_info.bit_depth)?;
                    has_black = true;
                    break;
                }
            }

            if has_black {
                dispatch_invert_channels(&mut working).map_err(gpu_error)?;
            }
        }

        let plan = build_gpu_plan(ctx).map_err(|e| match e {
            jxl_color::GpuTransformUnsupported::IccToIcc => {
                gpu_error("ICC transform on GPU".into())
            }
        })?;
        let output_channels = {
            let _transform = crate::phase_guard!("xyb2rgb_gpu_transform");
            run_gpu_plan(&plan, &mut working).map_err(gpu_error)?
        };
        if output_channels < 3 {
            working.remove_color_channels(output_channels);
        }
        working.set_ct_done(true);
        Ok(Arc::new(DeviceImage::Gpu(working)))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (ctx, frame, grid);
        gpu_unimplemented!("run_xyb2rgb");
    }
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
            return Err(format!(
                "GPU export supports at most 8 channels, got {channels}"
            ));
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
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

        let meta_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    image: Arc<DeviceImage>,
    color_bit_depth: BitDepth,
    extra_channels: &[(jxl_image::ExtraChannelType, BitDepth)],
) -> Result<(Arc<DeviceImage>, bool)> {
    #[cfg(feature = "gpu")]
    {
        use jxl_image::ExtraChannelType;

        let color_channels = image.color_channels();
        if color_channels != 3 {
            return Ok((image, false));
        }
        let has_spots = extra_channels
            .iter()
            .any(|(ec, _)| matches!(ec, ExtraChannelType::SpotColour { .. }));
        if !has_spots {
            return Ok((image, false));
        }

        let materialized = materialize_on_gpu(image).map_err(gpu_error)?;
        let mut gpu_image = unwrap_gpu_image(materialized)?;
        gpu_image.convert_modular_color(color_bit_depth)?;

        for (ec_idx, (ec_ty, ec_bit_depth)) in extra_channels.iter().enumerate() {
            let ExtraChannelType::SpotColour {
                red,
                green,
                blue,
                solidity,
            } = ec_ty
            else {
                continue;
            };
            let spot_buf_idx = color_channels + ec_idx;
            gpu_image.convert_channel_to_float(spot_buf_idx, *ec_bit_depth)?;
            dispatch_fuse_spot(
                &mut gpu_image,
                spot_buf_idx,
                (*red, *green, *blue),
                *solidity,
            )
            .map_err(gpu_error)?;
        }

        Ok((Arc::new(DeviceImage::Gpu(gpu_image)), true))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (image, color_bit_depth, extra_channels);
        gpu_unimplemented!("fuse_spot_colors");
    }
}

#[cfg(feature = "gpu")]
fn dispatch_fuse_spot(
    image: &mut GpuImageWithRegion,
    spot_idx: usize,
    rgb: (f32, f32, f32),
    solidity: f32,
) -> std::result::Result<(), String> {
    use std::sync::OnceLock;

    use wgpu::util::DeviceExt;

    use super::pipeline::{
        compute_pipeline, dispatch_2d, storage_read_layout, storage_rw_layout, uniform_layout,
    };

    const FUSE_WGSL: &str = include_str!("shaders/fuse_spot.wgsl");

    static PIPE: OnceLock<Option<super::pipeline::ComputePipeline>> = OnceLock::new();

    let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
    let bufs = image.buffer();
    if bufs.len() <= spot_idx || spot_idx < 3 {
        return Err("invalid spot buffer index".to_string());
    }
    let width = bufs[0].width() as u32;
    let height = bufs[0].height() as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct FuseSpotParams {
        width: u32,
        height: u32,
        spot_r: f32,
        spot_g: f32,
        spot_b: f32,
        solidity: f32,
    }

    let params = FuseSpotParams {
        width,
        height,
        spot_r: rgb.0,
        spot_g: rgb.1,
        spot_b: rgb.2,
        solidity,
    };

    let pipe = compute_pipeline(
        ctx,
        &PIPE,
        "jxlit_fuse_spot",
        FUSE_WGSL,
        "main",
        &[
            uniform_layout(0),
            storage_rw_layout(1),
            storage_rw_layout(2),
            storage_rw_layout(3),
            storage_read_layout(4),
        ],
    );

    let uniform_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fuse_spot_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fuse_spot"),
        layout: &pipe.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: match &bufs[0] {
                    super::image::GpuImageBuffer::F32 { buffer, .. } => buffer.as_entire_binding(),
                    _ => panic!("expected F32"),
                },
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: match &bufs[1] {
                    super::image::GpuImageBuffer::F32 { buffer, .. } => buffer.as_entire_binding(),
                    _ => panic!("expected F32"),
                },
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: match &bufs[2] {
                    super::image::GpuImageBuffer::F32 { buffer, .. } => buffer.as_entire_binding(),
                    _ => panic!("expected F32"),
                },
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: match &bufs[spot_idx] {
                    super::image::GpuImageBuffer::F32 { buffer, .. } => buffer.as_entire_binding(),
                    _ => panic!("expected F32"),
                },
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fuse_spot"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fuse_spot"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        dispatch_2d(ctx, width, height, &mut pass);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
