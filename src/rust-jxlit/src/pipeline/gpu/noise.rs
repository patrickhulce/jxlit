//! GPU noise synthesis: CPU convolve + GPU apply.
//!
//! RNG and convolution run on the CPU via the vendored [`synthesize_noise`] helper
//! (matching the reference implementation exactly). The final per-pixel XYB mix is
//! applied on the GPU.

#![cfg(feature = "gpu")]

use std::sync::OnceLock;

use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::NoiseParameters;
use crate::vendor::jxl_render::{Region, Result, features::synthesize_noise};

use super::context::GpuContext;
use super::image::GpuImageWithRegion;
use super::pipeline::{
    ComputePipeline, compute_pipeline, dispatch_2d, storage_read_layout, storage_rw_layout,
    uniform_layout,
};
use super::transfer::upload_buffer_init;

const NOISE_APPLY_WGSL: &str = include_str!("shaders/noise_apply.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseApplyParams {
    frame_width: u32,
    frame_height: u32,
    region_left: u32,
    region_top: u32,
    region_width: u32,
    region_height: u32,
    grid_stride: u32,
    grid_off_x: u32,
    grid_off_y: u32,
    corr_x: f32,
    corr_b: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseLutUniform {
    rows: [[f32; 4]; 3],
}

fn apply_pipeline() -> &'static ComputePipeline {
    static PIPE: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    let ctx = GpuContext::get().expect("GPU context required");
    compute_pipeline(
        ctx,
        &PIPE,
        "jxlit_noise_apply",
        NOISE_APPLY_WGSL,
        "main",
        &[
            uniform_layout(0),
            uniform_layout(1),
            storage_read_layout(2),
            storage_rw_layout(3),
            storage_rw_layout(4),
            storage_rw_layout(5),
        ],
    )
}

fn upload_convolved(convolved: &[jxl_grid::AlignedGrid<f32>; 3]) -> wgpu::Buffer {
    let width = convolved[0].width();
    let height = convolved[0].height();
    let plane = width * height;
    let mut bytes = vec![0.0f32; plane * 3];
    for ch in 0..3 {
        bytes[ch * plane..(ch + 1) * plane].copy_from_slice(convolved[ch].buf());
    }
    upload_buffer_init(
        "noise_convolved",
        bytemuck::cast_slice(&bytes),
        wgpu::BufferUsages::STORAGE,
    )
}

/// GPU mirror of vendored [`render_noise`](crate::vendor::jxl_render::features::render_noise).
pub fn dispatch_noise(
    image: &mut GpuImageWithRegion,
    frame_header: &FrameHeader,
    params: &NoiseParameters,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    base_correlations_xb: Option<(f32, f32)>,
) -> Result<()> {
    if image.color_channels() != 3 {
        return Ok(());
    }

    let ctx = GpuContext::get()
        .ok_or_else(|| crate::vendor::jxl_render::Error::NotSupported("GPU device unavailable"))?;

    let convolved = synthesize_noise(
        visible_frames_num,
        invisible_frames_num,
        frame_header,
        &JxlThreadPool::none(),
    )?;

    let frame_width = frame_header.width;
    let frame_height = frame_header.height;

    let (region, shift) = image.regions_and_shifts()[0];
    let full_frame_region = Region::with_size(frame_width, frame_height);
    let actual_region = region
        .intersection(full_frame_region)
        .downsample_with_shift(shift);
    let (corr_x, corr_b) = base_correlations_xb.unwrap_or((0.0, 1.0));

    let convolved_buf = upload_convolved(&convolved);

    let bufs = image.buffer();
    let mut lut_rows = [[0.0f32; 4]; 3];
    for (i, slot) in params.lut.iter().enumerate() {
        lut_rows[i / 4][i % 4] = *slot;
    }
    lut_rows[2][0] = params.lut[7];
    let lut_uniform = NoiseLutUniform { rows: lut_rows };
    let grid_stride = bufs[0].width() as u32;
    let grid_off_x = actual_region.left.saturating_sub(region.left) as u32;
    let grid_off_y = actual_region.top.saturating_sub(region.top) as u32;
    let apply_params = NoiseApplyParams {
        frame_width,
        frame_height,
        region_left: actual_region.left as u32,
        region_top: actual_region.top as u32,
        region_width: actual_region.width,
        region_height: actual_region.height,
        grid_stride,
        grid_off_x,
        grid_off_y,
        corr_x,
        corr_b,
    };
    let apply_params_buf = upload_buffer_init(
        "noise_apply_params",
        bytemuck::bytes_of(&apply_params),
        wgpu::BufferUsages::UNIFORM,
    );
    let apply_lut_buf = upload_buffer_init(
        "noise_apply_lut",
        bytemuck::bytes_of(&lut_uniform),
        wgpu::BufferUsages::UNIFORM,
    );

    let apply_pipe = apply_pipeline();
    let apply_bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("noise_apply"),
        layout: &apply_pipe.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: apply_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: apply_lut_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: convolved_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bufs[0].wgpu_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: bufs[1].wgpu_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: bufs[2].wgpu_buffer().as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("noise_apply"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("noise_apply"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&apply_pipe.pipeline);
        pass.set_bind_group(0, &apply_bind, &[]);
        dispatch_2d(ctx, actual_region.width, actual_region.height, &mut pass);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[cfg(test)]
pub fn dispatch_noise_for_test(
    image: &mut GpuImageWithRegion,
    frame_header: &FrameHeader,
    params: &NoiseParameters,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    base_correlations_xb: Option<(f32, f32)>,
) -> Result<()> {
    dispatch_noise(
        image,
        frame_header,
        params,
        visible_frames_num,
        invisible_frames_num,
        base_correlations_xb,
    )
}
