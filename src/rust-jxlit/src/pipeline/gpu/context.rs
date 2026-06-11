//! wgpu device/queue singleton and decoded pixel buffer handle.

use crate::types::PixelLayout;

#[cfg(feature = "gpu")]
use std::sync::{Arc, OnceLock};

/// GPU-resident decoded pixel buffer handle returned when `destination=Gpu`.
#[derive(Debug)]
pub struct GpuPixelBuffer {
    #[cfg(feature = "gpu")]
    pub(crate) buffer: Arc<wgpu::Buffer>,
    #[cfg(not(feature = "gpu"))]
    _stub: (),
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub layout: PixelLayout,
    pub len: usize,
}

impl GpuPixelBuffer {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    pub fn layout(&self) -> PixelLayout {
        self.layout
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn new(
        buffer: Arc<wgpu::Buffer>,
        width: u32,
        height: u32,
        channels: u32,
        layout: PixelLayout,
        len: usize,
    ) -> Self {
        Self {
            buffer,
            width,
            height,
            channels,
            layout,
            len,
        }
    }
}

/// Cached wgpu runtime for compute kernels and transfers.
#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    pub fn get() -> Option<&'static Self> {
        static CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();
        CONTEXT.get_or_init(init_context).as_ref()
    }
}

#[cfg(feature = "gpu")]
fn init_context() -> Option<GpuContext> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("jxlit"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;
        Some(GpuContext { device, queue })
    })
}
