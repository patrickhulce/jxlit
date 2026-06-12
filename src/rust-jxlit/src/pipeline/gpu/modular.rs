//! Modular integer → F32 GPU conversion.

#![cfg(feature = "gpu")]

use std::sync::OnceLock;

use jxl_image::BitDepth;

use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::image::{GpuImageBuffer, GpuSampleKind, sample_kind_bits};
use super::pipeline::{
    ComputePipeline, compute_pipeline, dispatch_2d, storage_read_layout, storage_rw_layout,
    uniform_layout,
};

const MODULAR_WGSL: &str = include_str!("shaders/modular_to_float.wgsl");

fn modular_pipeline() -> &'static ComputePipeline {
    static PIPE: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    let ctx = GpuContext::get().expect("GPU context required");
    compute_pipeline(
        ctx,
        &PIPE,
        "jxlit_modular_to_float",
        MODULAR_WGSL,
        "main",
        &[
            uniform_layout(0),
            storage_read_layout(1),
            storage_rw_layout(2),
        ],
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModularParams {
    width: u32,
    height: u32,
    sample_kind: u32,
    bits_per_sample: u32,
}

/// Converts one GPU channel from modular integer to F32.
pub fn dispatch_modular_to_float(
    grid: &GpuImageBuffer,
    bit_depth: BitDepth,
) -> Result<GpuImageBuffer, String> {
    if grid.sample_kind() == GpuSampleKind::F32 {
        return grid.try_clone();
    }
    let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
    let width = grid.width();
    let height = grid.height();
    let (sample_kind, bits) = sample_kind_bits(grid.sample_kind(), bit_depth);
    let out = GpuImageBuffer::empty_f32(width, height, ctx);
    let params = ModularParams {
        width: width as u32,
        height: height as u32,
        sample_kind,
        bits_per_sample: bits,
    };
    let pipe = modular_pipeline();
    let uniform_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("modular_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("modular_to_float"),
        layout: &pipe.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grid.wgpu_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.wgpu_buffer().as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("modular_to_float"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("modular_to_float"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        dispatch_2d(ctx, width as u32, height as u32, &mut pass);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(out)
}
