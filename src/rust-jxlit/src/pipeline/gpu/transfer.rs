//! GPU <-> CPU buffer transfers (all timed via [`crate::phase_guard`]).
//!
//! Host-to-device uploads use `gpu_htod_*` measures; device-to-host downloads use
//! `gpu_dtoh_*`. Call sites should route through these helpers rather than calling
//! `write_buffer`, `create_buffer_init`, or staging readbacks directly.

use crate::types::PixelLayout;

use super::context::GpuPixelBuffer;

#[cfg(feature = "gpu")]
use super::context::GpuContext;

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

#[cfg(feature = "gpu")]
fn upload_buffer_init_impl(
    label: &'static str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let ctx = GpuContext::get().expect("GPU context required");
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
}

#[cfg(feature = "gpu")]
fn download_buffer_bytes_impl(buf: &wgpu::Buffer, byte_len: usize) -> Result<Vec<u8>, String> {
    let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxlit_download_staging"),
        size: byte_len.max(4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxlit_download_buffer"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_len as u64);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| "GPU buffer map channel closed".to_string())?
        .map_err(|e| format!("GPU buffer map failed: {e:?}"))?;

    let data = buffer_slice.get_mapped_range();
    let out = data.to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

/// Uploads host bytes into a new GPU buffer (`gpu_htod_buffer`).
#[cfg(feature = "gpu")]
pub(crate) fn upload_buffer_init(
    label: &'static str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let _guard = crate::phase_guard!("gpu_htod_buffer");
    upload_buffer_init_impl(label, contents, usage)
}

/// Downloads a GPU buffer to host bytes (`gpu_dtoh_buffer`).
#[cfg(feature = "gpu")]
pub(crate) fn download_buffer_bytes(
    buf: &wgpu::Buffer,
    byte_len: usize,
) -> Result<Vec<u8>, String> {
    let _guard = crate::phase_guard!("gpu_dtoh_buffer");
    download_buffer_bytes_impl(buf, byte_len)
}

/// Downloads a GPU `f32` buffer (`gpu_dtoh_buffer`).
#[cfg(feature = "gpu")]
pub(crate) fn download_buffer_f32(buf: &wgpu::Buffer, len: usize) -> Result<Vec<f32>, String> {
    let bytes = download_buffer_bytes(buf, len * std::mem::size_of::<f32>())?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

/// Used by [`super::image::GpuImageWithRegion::to_cpu`] (timed as `gpu_dtoh_image`).
#[cfg(feature = "gpu")]
pub(crate) fn download_buffer_bytes_untimed(
    buf: &wgpu::Buffer,
    byte_len: usize,
) -> Result<Vec<u8>, String> {
    download_buffer_bytes_impl(buf, byte_len)
}

/// Uploads a CPU `f32` pixel buffer to the GPU (`gpu_htod_pixels`).
pub fn upload_pixels(
    pixels: &[f32],
    width: u32,
    height: u32,
    channels: u32,
    layout: PixelLayout,
) -> Result<GpuPixelBuffer, String> {
    #[cfg(feature = "gpu")]
    {
        let _guard = crate::phase_guard!("gpu_htod_pixels");
        let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
        let len = pixels.len();
        let byte_len = std::mem::size_of_val(pixels);
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxlit_upload_pixels"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(pixels));
        Ok(GpuPixelBuffer::new(
            std::sync::Arc::new(buffer),
            width,
            height,
            channels,
            layout,
            len,
        ))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (pixels, width, height, channels, layout);
        Err("GPU feature not enabled".to_string())
    }
}

/// Downloads a GPU pixel buffer to CPU (`gpu_dtoh_pixels`).
pub fn download_pixels(gpu: &GpuPixelBuffer) -> Result<Vec<f32>, String> {
    #[cfg(feature = "gpu")]
    {
        let _guard = crate::phase_guard!("gpu_dtoh_pixels");
        let byte_len = gpu.len * std::mem::size_of::<f32>();
        let bytes = download_buffer_bytes_impl(&gpu.buffer, byte_len)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu;
        Err("GPU feature not enabled".to_string())
    }
}
