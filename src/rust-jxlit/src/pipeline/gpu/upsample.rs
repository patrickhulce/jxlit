//! Non-separable upsampling GPU kernels.

#![cfg(feature = "gpu")]

use std::sync::OnceLock;

use jxl_image::ImageHeader;
use jxl_modular::ChannelShift;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_render::Region;

use super::context::GpuContext;
use super::crop::dispatch_crop_f32;
use super::image::{GpuImageBuffer, GpuImageWithRegion, GpuSampleKind};
use super::pipeline::{
    ComputePipeline, compute_pipeline, dispatch_2d, storage_read_layout, storage_rw_layout,
    uniform_layout,
};
use super::transfer::upload_buffer_init;

const UPSAMPLE_WGSL: &str = include_str!("shaders/nonseparable_upsample.wgsl");

fn upsample_pipeline() -> &'static ComputePipeline {
    static PIPE: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    let ctx = GpuContext::get().expect("GPU context required");
    compute_pipeline(
        ctx,
        &PIPE,
        "jxlit_nonseparable_upsample",
        UPSAMPLE_WGSL,
        "main",
        &[
            uniform_layout(0),
            storage_read_layout(1),
            storage_read_layout(2),
            storage_rw_layout(3),
        ],
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UpsampleParams {
    in_width: u32,
    in_height: u32,
    k: u32,
    mat_n: u32,
    out_width: u32,
    out_height: u32,
}

fn build_weights_quarter<const K: usize, const NW: usize>(weights: &[f32; NW]) -> Vec<f32> {
    assert!((K == 2 && NW == 15) || (K == 4 && NW == 55) || (K == 8 && NW == 210));
    let mat_n = K / 2;
    let num_kernels = K * K / 4;
    let mut weights_quarter = vec![0.0f32; num_kernels * 25];
    let mut weight_idx = 0usize;
    for y in 0..5 * mat_n {
        let mat_y = y / 5;
        let ky = y % 5;
        for x in y..5 * mat_n {
            let mat_x = x / 5;
            let kx = x % 5;
            let w = weights[weight_idx];
            weight_idx += 1;
            weights_quarter[(mat_y * mat_n + mat_x) * 25 + ky * 5 + kx] = w;
            weights_quarter[(mat_x * mat_n + mat_y) * 25 + kx * 5 + ky] = w;
        }
    }
    weights_quarter
}

fn dispatch_upsample_pass(
    grid: &GpuImageBuffer,
    weights: &[f32],
    k: u32,
) -> std::result::Result<GpuImageBuffer, String> {
    if grid.sample_kind() != GpuSampleKind::F32 {
        return Err("nonseparable upsample requires F32 channel".to_string());
    }
    let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
    let in_width = grid.width() as u32;
    let in_height = grid.height() as u32;
    let mat_n = k / 2;
    let out_width = in_width * k;
    let out_height = in_height * k;
    let out = GpuImageBuffer::empty_f32(out_width as usize, out_height as usize, ctx);

    let params = UpsampleParams {
        in_width,
        in_height,
        k,
        mat_n,
        out_width,
        out_height,
    };
    let weights_buf = upload_buffer_init(
        "upsample_weights",
        bytemuck::cast_slice(weights),
        wgpu::BufferUsages::STORAGE,
    );
    let uniform_buf = upload_buffer_init(
        "upsample_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    );
    let pipe = upsample_pipeline();
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nonseparable_upsample"),
        layout: &pipe.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grid.wgpu_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.wgpu_buffer().as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nonseparable_upsample"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nonseparable_upsample"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        dispatch_2d(ctx, out_width, out_height, &mut pass);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(out)
}

#[cfg(test)]
pub fn gpu_upsample_channel_for_test(
    grid: GpuImageBuffer,
    image_header: &ImageHeader,
    factor: u32,
) -> std::result::Result<GpuImageBuffer, String> {
    gpu_upsample_channel(grid, image_header, factor)
}

fn gpu_upsample_channel(
    grid: GpuImageBuffer,
    image_header: &ImageHeader,
    factor: u32,
) -> std::result::Result<GpuImageBuffer, String> {
    let metadata = &image_header.metadata;
    let mut current = grid;
    let up8 = factor / 3;
    let last_up = factor % 3;

    for _ in 0..up8 {
        let weights = build_weights_quarter::<8, 210>(&metadata.up8_weight);
        current = dispatch_upsample_pass(&current, &weights, 8)?;
    }
    current = match last_up {
        1 => {
            let weights = build_weights_quarter::<2, 15>(&metadata.up2_weight);
            dispatch_upsample_pass(&current, &weights, 2)?
        }
        2 => {
            let weights = build_weights_quarter::<4, 55>(&metadata.up4_weight);
            dispatch_upsample_pass(&current, &weights, 4)?
        }
        _ => current,
    };
    Ok(current)
}

/// GPU mirror of [`ImageWithRegion::upsample_nonseparable`].
pub fn upsample_nonseparable(
    image: &mut GpuImageWithRegion,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    upsampled_valid_region: Region,
    ec_to_color_only: bool,
) -> crate::vendor::jxl_render::Result<()> {
    let color_channels = image.color_channels();
    let color_shift = frame_header.upsampling.trailing_zeros();
    let channel_count = image.buffer().len();

    for idx in 0..channel_count {
        let bit_depth = if let Some(ec_idx) = idx.checked_sub(color_channels) {
            image_header.metadata.ec_info[ec_idx].bit_depth
        } else {
            image_header.metadata.bit_depth
        };

        let (region, shift) = image.regions_and_shifts()[idx];
        let ChannelShift::Shifts(upsampling_factor) = shift else {
            return Err(crate::vendor::jxl_render::Error::NotSupported(
                "invalid channel shift for upsampling",
            ));
        };

        let target_factor = if ec_to_color_only { color_shift } else { 0 };
        if upsampling_factor == target_factor {
            continue;
        }

        if image.buffer()[idx].sample_kind() != GpuSampleKind::F32 {
            image.convert_channel_to_float(idx, bit_depth)?;
        }

        let downsampled_image_region = region.downsample(upsampling_factor);
        let downsampled_valid_region = upsampled_valid_region.downsample(upsampling_factor);
        let left = downsampled_valid_region
            .left
            .abs_diff(downsampled_image_region.left);
        let top = downsampled_valid_region
            .top
            .abs_diff(downsampled_image_region.top);
        let width = downsampled_valid_region.width;
        let height = downsampled_valid_region.height;

        let cropped = {
            let _crop = crate::phase_guard!("nonsep_upsample_gpu_crop");
            dispatch_crop_f32(&image.buffer()[idx], left, top, width, height).map_err(|e| {
                crate::vendor::jxl_render::Error::NotSupported(Box::leak(e.into_boxed_str()))
            })?
        };

        let upsample_factor = upsampling_factor - target_factor;
        let upsampled = {
            let _upsample = crate::phase_guard!("nonsep_upsample_gpu_pass");
            gpu_upsample_channel(cropped, image_header, upsample_factor).map_err(|e| {
                crate::vendor::jxl_render::Error::NotSupported(Box::leak(e.into_boxed_str()))
            })?
        };

        image.buffer_mut()[idx] = upsampled;
        image.regions_mut()[idx] = (
            downsampled_valid_region.upsample(upsample_factor),
            ChannelShift::from_shift(target_factor),
        );
    }

    Ok(())
}
