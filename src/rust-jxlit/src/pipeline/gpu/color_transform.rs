//! GPU color-transform op-chain dispatcher.

#![cfg(feature = "gpu")]

use std::sync::{Arc, OnceLock};

use jxl_color::{ColorTransformGpuOp, GpuTransformUnsupported, TransferFunction};

use crate::vendor::jxl_render::RenderContext;
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::image::{GpuImageBuffer, GpuImageWithRegion, GpuSampleKind};
use super::pipeline::{
    ComputePipeline, compute_pipeline, dispatch_2d, storage_rw_layout, uniform_layout,
};

const COLOR_OPS_WGSL: &str = include_str!("shaders/color_ops.wgsl");

fn channel_bindings() -> [wgpu::BindGroupLayoutEntry; 4] {
    [
        uniform_layout(0),
        storage_rw_layout(1),
        storage_rw_layout(2),
        storage_rw_layout(3),
    ]
}

fn color_ops_pipeline(entry: &'static str) -> &'static ComputePipeline {
    static XYB: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static MATRIX: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static LUMA_XYZ: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static XYZ_LUMA: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static TRANSFER: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static HLG: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static GAMUT: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static CLIP: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static INVERT: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static YCBCR: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static TONE_RGB: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    static TONE_LUMA: OnceLock<Option<ComputePipeline>> = OnceLock::new();

    let ctx = GpuContext::get().expect("GPU context required");
    match entry {
        "xyb_to_lms" => compute_pipeline(
            ctx,
            &XYB,
            "jxlit_xyb_to_lms",
            COLOR_OPS_WGSL,
            "xyb_to_lms",
            &channel_bindings(),
        ),
        "matrix3" => compute_pipeline(
            ctx,
            &MATRIX,
            "jxlit_matrix3",
            COLOR_OPS_WGSL,
            "matrix3",
            &channel_bindings(),
        ),
        "luma_to_xyz" => compute_pipeline(
            ctx,
            &LUMA_XYZ,
            "jxlit_luma_to_xyz",
            COLOR_OPS_WGSL,
            "luma_to_xyz",
            &channel_bindings(),
        ),
        "xyz_to_luma" => compute_pipeline(
            ctx,
            &XYZ_LUMA,
            "jxlit_xyz_to_luma",
            COLOR_OPS_WGSL,
            "xyz_to_luma",
            &channel_bindings(),
        ),
        "transfer_fn" => compute_pipeline(
            ctx,
            &TRANSFER,
            "jxlit_transfer_fn",
            COLOR_OPS_WGSL,
            "transfer_fn",
            &channel_bindings(),
        ),
        "hlg_inverse_ootf" => compute_pipeline(
            ctx,
            &HLG,
            "jxlit_hlg_inverse_ootf",
            COLOR_OPS_WGSL,
            "hlg_inverse_ootf",
            &channel_bindings(),
        ),
        "gamut_map" => compute_pipeline(
            ctx,
            &GAMUT,
            "jxlit_gamut_map",
            COLOR_OPS_WGSL,
            "gamut_map",
            &channel_bindings(),
        ),
        "clip" => compute_pipeline(
            ctx,
            &CLIP,
            "jxlit_clip",
            COLOR_OPS_WGSL,
            "clip",
            &channel_bindings(),
        ),
        "invert_channels" => compute_pipeline(
            ctx,
            &INVERT,
            "jxlit_invert",
            COLOR_OPS_WGSL,
            "invert_channels",
            &channel_bindings(),
        ),
        "ycbcr_to_rgb" => compute_pipeline(
            ctx,
            &YCBCR,
            "jxlit_ycbcr_to_rgb",
            COLOR_OPS_WGSL,
            "ycbcr_to_rgb",
            &channel_bindings(),
        ),
        "tone_map_rgb" => compute_pipeline(
            ctx,
            &TONE_RGB,
            "jxlit_tone_map_rgb",
            COLOR_OPS_WGSL,
            "tone_map_rgb",
            &channel_bindings(),
        ),
        "tone_map_luma" => compute_pipeline(
            ctx,
            &TONE_LUMA,
            "jxlit_tone_map_luma",
            COLOR_OPS_WGSL,
            "tone_map_luma",
            &channel_bindings(),
        ),
        _ => panic!("unknown color ops entry: {entry}"),
    }
}

/// Builds a GPU-dispatchable color transform plan from the render context.
pub fn build_gpu_plan(
    ctx: &RenderContext,
) -> std::result::Result<Vec<ColorTransformGpuOp>, GpuTransformUnsupported> {
    ctx.ensure_color_transform_cached()
        .map_err(|_| GpuTransformUnsupported::IccToIcc)?;
    ctx.with_cached_transform(|t| t.gpu_ops())
        .map_err(|_| GpuTransformUnsupported::IccToIcc)?
}

struct ChannelView {
    width: u32,
    height: u32,
    buffers: [Arc<wgpu::Buffer>; 3],
}

fn channel_view(image: &GpuImageWithRegion, count: usize) -> Result<ChannelView, String> {
    let bufs = image.buffer();
    if count > 3 || bufs.len() < count {
        return Err(format!("expected at least {count} GPU channels"));
    }
    let width = bufs[0].width() as u32;
    let height = bufs[0].height() as u32;
    for b in &bufs[..count] {
        if b.sample_kind() != GpuSampleKind::F32 {
            return Err("color transform requires F32 channels".to_string());
        }
        if b.width() as u32 != width || b.height() as u32 != height {
            return Err("color channel size mismatch".to_string());
        }
    }
    let f32_buf = |b: &GpuImageBuffer| match b {
        GpuImageBuffer::F32 { buffer, .. } => Arc::clone(buffer),
        _ => unreachable!(),
    };
    Ok(ChannelView {
        width,
        height,
        buffers: [f32_buf(&bufs[0]), f32_buf(&bufs[1]), f32_buf(&bufs[2])],
    })
}

fn run_pass(entry: &'static str, uniform: &[u8], view: &ChannelView) {
    let ctx = GpuContext::get().expect("GPU context required");
    let pipe = color_ops_pipeline(entry);
    let uniform_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(entry),
            contents: uniform,
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(entry),
        layout: &pipe.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: view.buffers[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: view.buffers[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: view.buffers[2].as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        dispatch_2d(ctx, view.width, view.height, &mut pass);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

fn tf_kind(tf: TransferFunction) -> (u32, f32) {
    match tf {
        TransferFunction::Linear => (0, 0.0),
        TransferFunction::Srgb => (1, 0.0),
        TransferFunction::Bt709 => (2, 0.0),
        TransferFunction::Gamma { g, .. } => (3, g as f32),
        TransferFunction::Pq => (4, 0.0),
        TransferFunction::Dci => (5, 0.0),
        TransferFunction::Hlg => (6, 0.0),
        TransferFunction::Unknown => (7, 0.0),
    }
}

fn download_f32_channel(buf: &Arc<wgpu::Buffer>, len: usize) -> Vec<f32> {
    let ctx = GpuContext::get().expect("GPU context");
    let byte_len = len * 4;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("download_channel"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_len as u64);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::Maintain::Wait);
    let data = staging.slice(..).get_mapped_range();
    let out = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

fn detect_peak_luminance_cpu(view: &ChannelView, luminances: [f32; 3]) -> f32 {
    let len = (view.width * view.height) as usize;
    let r = download_f32_channel(&view.buffers[0], len);
    let g = download_f32_channel(&view.buffers[1], len);
    let b = download_f32_channel(&view.buffers[2], len);
    let [lr, lg, lb] = luminances;
    let mut peak = 0.0f32;
    for i in 0..len {
        peak = peak.max(r[i] * lr + g[i] * lg + b[i] * lb);
    }
    if peak <= 0.0 { 1.0 } else { peak }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParams {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct XybParams {
    width: u32,
    height: u32,
    opsin_bias_x: f32,
    opsin_bias_y: f32,
    opsin_bias_z: f32,
    intensity_target: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixParams {
    width: u32,
    height: u32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LumaXyzParams {
    width: u32,
    height: u32,
    illuminant_x: f32,
    illuminant_y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TransferParams {
    width: u32,
    height: u32,
    tf_kind: u32,
    inverse: u32,
    gamma: f32,
    intensity_target: f32,
    luminance_r: f32,
    luminance_g: f32,
    luminance_b: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GamutParams {
    width: u32,
    height: u32,
    luminance_r: f32,
    luminance_g: f32,
    luminance_b: f32,
    saturation_factor: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ToneMapParams {
    width: u32,
    height: u32,
    luminance_r: f32,
    luminance_g: f32,
    luminance_b: f32,
    intensity_target: f32,
    min_nits: f32,
    target_display_luminance: f32,
    peak_luminance: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ToneMapLumaParams {
    width: u32,
    height: u32,
    intensity_target: f32,
    min_nits: f32,
    target_display_luminance: f32,
    peak_luminance: f32,
}

/// Runs a serialized GPU color transform plan in-place on the first three channels.
pub fn run_gpu_plan(
    plan: &[ColorTransformGpuOp],
    image: &mut GpuImageWithRegion,
) -> Result<usize, String> {
    let mut channel_count = 3usize;
    let view = channel_view(image, 3)?;
    for op in plan {
        match op {
            ColorTransformGpuOp::XybToMixedLms {
                opsin_bias,
                intensity_target,
            } => {
                let _op = crate::phase_guard!("gpu_ct_xyb_to_lms");
                let p = XybParams {
                    width: view.width,
                    height: view.height,
                    opsin_bias_x: opsin_bias[0],
                    opsin_bias_y: opsin_bias[1],
                    opsin_bias_z: opsin_bias[2],
                    intensity_target: *intensity_target,
                };
                run_pass("xyb_to_lms", bytemuck::bytes_of(&p), &view);
            }
            ColorTransformGpuOp::Matrix(m) => {
                let _op = crate::phase_guard!("gpu_ct_matrix");
                let p = MatrixParams {
                    width: view.width,
                    height: view.height,
                    m00: m[0],
                    m01: m[1],
                    m02: m[2],
                    m10: m[3],
                    m11: m[4],
                    m12: m[5],
                    m20: m[6],
                    m21: m[7],
                    m22: m[8],
                };
                run_pass("matrix3", bytemuck::bytes_of(&p), &view);
            }
            ColorTransformGpuOp::LumaToXyz { illuminant } => {
                let _op = crate::phase_guard!("gpu_ct_luma_to_xyz");
                let p = LumaXyzParams {
                    width: view.width,
                    height: view.height,
                    illuminant_x: illuminant[0],
                    illuminant_y: illuminant[1],
                };
                run_pass("luma_to_xyz", bytemuck::bytes_of(&p), &view);
            }
            ColorTransformGpuOp::XyzToLuma => {
                let _op = crate::phase_guard!("gpu_ct_xyz_to_luma");
                let p = GridParams {
                    width: view.width,
                    height: view.height,
                };
                run_pass("xyz_to_luma", bytemuck::bytes_of(&p), &view);
                channel_count = 1;
            }
            ColorTransformGpuOp::TransferFunction {
                tf,
                luminances,
                intensity_target,
                min_nits: _,
                inverse,
            } => {
                let _op = crate::phase_guard!("gpu_ct_transfer_fn");
                let (tf_kind, gamma) = tf_kind(*tf);
                let p = TransferParams {
                    width: view.width,
                    height: view.height,
                    tf_kind,
                    inverse: u32::from(*inverse),
                    gamma,
                    intensity_target: *intensity_target,
                    luminance_r: luminances[0],
                    luminance_g: luminances[1],
                    luminance_b: luminances[2],
                };
                run_pass("transfer_fn", bytemuck::bytes_of(&p), &view);
            }
            ColorTransformGpuOp::HlgInverseOotf {
                luminances,
                intensity_target,
                min_nits: _,
            } => {
                let _op = crate::phase_guard!("gpu_ct_hlg_ootf");
                let p = TransferParams {
                    width: view.width,
                    height: view.height,
                    tf_kind: 6,
                    inverse: 0,
                    gamma: 0.0,
                    intensity_target: *intensity_target,
                    luminance_r: luminances[0],
                    luminance_g: luminances[1],
                    luminance_b: luminances[2],
                };
                run_pass("hlg_inverse_ootf", bytemuck::bytes_of(&p), &view);
            }
            ColorTransformGpuOp::ToneMapRec2408 {
                luminances,
                intensity_target,
                min_nits,
                target_display_luminance,
                detect_peak,
            } => {
                let _op = crate::phase_guard!("gpu_ct_tone_map_rgb");
                let peak = if *detect_peak {
                    let _peak = crate::phase_guard!("gpu_ct_detect_peak");
                    detect_peak_luminance_cpu(&view, *luminances) * intensity_target
                } else {
                    *intensity_target
                };
                let p = ToneMapParams {
                    width: view.width,
                    height: view.height,
                    luminance_r: luminances[0],
                    luminance_g: luminances[1],
                    luminance_b: luminances[2],
                    intensity_target: *intensity_target,
                    min_nits: *min_nits,
                    target_display_luminance: *target_display_luminance,
                    peak_luminance: peak,
                };
                run_pass("tone_map_rgb", bytemuck::bytes_of(&p), &view);
            }
            ColorTransformGpuOp::ToneMapLumaRec2408 {
                intensity_target,
                min_nits,
                target_display_luminance,
                detect_peak,
            } => {
                let _op = crate::phase_guard!("gpu_ct_tone_map_luma");
                let peak = if *detect_peak {
                    let _peak = crate::phase_guard!("gpu_ct_detect_peak");
                    detect_peak_luminance_cpu(&view, [1.0, 0.0, 0.0]) * intensity_target
                } else {
                    *intensity_target
                };
                let p = ToneMapLumaParams {
                    width: view.width,
                    height: view.height,
                    intensity_target: *intensity_target,
                    min_nits: *min_nits,
                    target_display_luminance: *target_display_luminance,
                    peak_luminance: peak,
                };
                run_pass("tone_map_luma", bytemuck::bytes_of(&p), &view);
                channel_count = 1;
            }
            ColorTransformGpuOp::GamutMap {
                luminances,
                saturation_factor,
            } => {
                let _op = crate::phase_guard!("gpu_ct_gamut_map");
                let p = GamutParams {
                    width: view.width,
                    height: view.height,
                    luminance_r: luminances[0],
                    luminance_g: luminances[1],
                    luminance_b: luminances[2],
                    saturation_factor: *saturation_factor,
                };
                run_pass("gamut_map", bytemuck::bytes_of(&p), &view);
            }
            ColorTransformGpuOp::Clip => {
                let _op = crate::phase_guard!("gpu_ct_clip");
                let p = GridParams {
                    width: view.width,
                    height: view.height,
                };
                run_pass("clip", bytemuck::bytes_of(&p), &view);
            }
        }
    }
    Ok(channel_count)
}

pub fn dispatch_ycbcr_to_rgb(image: &mut GpuImageWithRegion) -> Result<(), String> {
    let view = channel_view(image, 3)?;
    let p = GridParams {
        width: view.width,
        height: view.height,
    };
    run_pass("ycbcr_to_rgb", bytemuck::bytes_of(&p), &view);
    Ok(())
}

pub fn dispatch_invert_channels(image: &mut GpuImageWithRegion) -> Result<(), String> {
    let view = channel_view(image, 3)?;
    let p = GridParams {
        width: view.width,
        height: view.height,
    };
    run_pass("invert_channels", bytemuck::bytes_of(&p), &view);
    Ok(())
}
