//! GPU <-> CPU pixel buffer transfers.

use crate::types::PixelLayout;

use super::context::GpuPixelBuffer;

#[cfg(feature = "gpu")]
use super::context::GpuContext;

/// Uploads a CPU `f32` pixel buffer to the GPU (CPU fallback + `destination=Gpu`).
pub fn upload_pixels(
    pixels: &[f32],
    width: u32,
    height: u32,
    channels: u32,
    layout: PixelLayout,
) -> Result<GpuPixelBuffer, String> {
    #[cfg(feature = "gpu")]
    {
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

/// Downloads a GPU pixel buffer to CPU (GPU final step + `destination=Cpu`).
pub fn download_pixels(gpu: &GpuPixelBuffer) -> Result<Vec<f32>, String> {
    #[cfg(feature = "gpu")]
    {
        let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
        let byte_len = gpu.len * std::mem::size_of::<f32>();
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxlit_download_staging"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxlit_download"),
            });
        encoder.copy_buffer_to_buffer(&gpu.buffer, 0, &staging, 0, byte_len as u64);
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
        let pixels: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(pixels)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu;
        Err("GPU feature not enabled".to_string())
    }
}
