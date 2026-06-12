//! Shared wgpu compute pipeline helpers.

#![cfg(feature = "gpu")]

use std::sync::OnceLock;

use super::context::GpuContext;

pub const WORKGROUP_SIZE: u32 = 8;

/// Cached compute pipeline with its bind group layout.
pub struct ComputePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub fn compute_pipeline(
    ctx: &GpuContext,
    cache: &'static OnceLock<Option<ComputePipeline>>,
    label: &str,
    wgsl: &str,
    entry_point: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> &'static ComputePipeline {
    cache
        .get_or_init(|| build_compute_pipeline(ctx, label, wgsl, entry_point, entries).ok())
        .as_ref()
        .unwrap_or_else(|| panic!("failed to build compute pipeline: {label}"))
}

fn build_compute_pipeline(
    ctx: &GpuContext,
    label: &str,
    wgsl: &str,
    entry_point: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Result<ComputePipeline, String> {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

    Ok(ComputePipeline {
        pipeline,
        bind_group_layout,
    })
}

pub fn dispatch_2d(ctx: &GpuContext, width: u32, height: u32, pass: &mut wgpu::ComputePass<'_>) {
    let wg_x = width.div_ceil(WORKGROUP_SIZE);
    let wg_y = height.div_ceil(WORKGROUP_SIZE);
    pass.dispatch_workgroups(wg_x, wg_y, 1);
    let _ = ctx;
}

pub fn copy_buffer(ctx: &GpuContext, src: &wgpu::Buffer, dst: &wgpu::Buffer, len: u64) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxlit_buffer_copy"),
        });
    encoder.copy_buffer_to_buffer(src, 0, dst, 0, len);
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

pub fn uniform_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn storage_read_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn storage_rw_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
