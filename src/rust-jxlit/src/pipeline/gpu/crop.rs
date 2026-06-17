//! GPU channel crop helper.

#![cfg(feature = "gpu")]

use std::sync::OnceLock;

use super::context::GpuContext;
use super::image::GpuImageBuffer;
use super::pipeline::{
    ComputePipeline, compute_pipeline, dispatch_2d, storage_read_layout, storage_rw_layout,
    uniform_layout,
};
use super::transfer::upload_buffer_init;

const CROP_WGSL: &str = include_str!("shaders/crop.wgsl");

fn crop_pipeline() -> &'static ComputePipeline {
    static PIPE: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    let ctx = GpuContext::get().expect("GPU context required");
    compute_pipeline(
        ctx,
        &PIPE,
        "jxlit_crop",
        CROP_WGSL,
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
struct CropParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    off_x: u32,
    off_y: u32,
}

/// Crops an F32 GPU channel buffer to a sub-rectangle.
pub fn dispatch_crop_f32(
    grid: &GpuImageBuffer,
    off_x: u32,
    off_y: u32,
    width: u32,
    height: u32,
) -> Result<GpuImageBuffer, String> {
    let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
    let src_width = grid.width() as u32;
    let src_height = grid.height() as u32;
    if off_x + width > src_width || off_y + height > src_height {
        return Err("crop region out of bounds".to_string());
    }
    if grid.sample_kind() != super::image::GpuSampleKind::F32 {
        return Err("crop requires F32 channel".to_string());
    }

    let out = GpuImageBuffer::empty_f32(width as usize, height as usize, ctx);
    let params = CropParams {
        src_width,
        src_height,
        dst_width: width,
        dst_height: height,
        off_x,
        off_y,
    };
    let pipe = crop_pipeline();
    let uniform_buf = upload_buffer_init(
        "crop_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    );
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("crop"),
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
            label: Some("crop"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("crop"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        dispatch_2d(ctx, width, height, &mut pass);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(out)
}
