//! GPU-side mirrors of the vendored image buffer types.

#[cfg(feature = "gpu")]
use std::sync::Arc;

use jxl_grid::AllocTracker;
use jxl_image::BitDepth;
use jxl_modular::{ChannelShift, Sample};

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::GlobalModular;
#[cfg(feature = "gpu")]
use crate::vendor::jxl_render::ImageBuffer;
use crate::vendor::jxl_render::{ImageWithRegion, Region, Result};

#[cfg(feature = "gpu")]
use super::context::GpuContext;

/// Sample storage kind for a GPU channel buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSampleKind {
    F32,
    I32,
    I16,
}

/// Mirror of [`ImageBuffer`] for GPU-resident channel storage.
#[derive(Debug)]
pub enum GpuImageBuffer {
    F32 {
        width: usize,
        height: usize,
        #[cfg(feature = "gpu")]
        buffer: Arc<wgpu::Buffer>,
    },
    I32 {
        width: usize,
        height: usize,
        #[cfg(feature = "gpu")]
        buffer: Arc<wgpu::Buffer>,
    },
    I16 {
        width: usize,
        height: usize,
        #[cfg(feature = "gpu")]
        buffer: Arc<wgpu::Buffer>,
    },
}

impl GpuImageBuffer {
    pub fn width(&self) -> usize {
        match self {
            Self::F32 { width, .. } | Self::I32 { width, .. } | Self::I16 { width, .. } => *width,
        }
    }

    pub fn height(&self) -> usize {
        match self {
            Self::F32 { height, .. } | Self::I32 { height, .. } | Self::I16 { height, .. } => {
                *height
            }
        }
    }

    pub fn sample_kind(&self) -> GpuSampleKind {
        match self {
            Self::F32 { .. } => GpuSampleKind::F32,
            Self::I32 { .. } => GpuSampleKind::I32,
            Self::I16 { .. } => GpuSampleKind::I16,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn wgpu_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::F32 { buffer, .. } | Self::I32 { buffer, .. } | Self::I16 { buffer, .. } => {
                buffer
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn empty_f32(width: usize, height: usize, ctx: &GpuContext) -> Self {
        let len = width * height;
        let byte_len = len * std::mem::size_of::<f32>();
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxlit_gpu_f32"),
            size: byte_len.max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self::F32 {
            width,
            height,
            buffer: Arc::new(buffer),
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn empty_f32(width: usize, height: usize) -> Self {
        Self::F32 { width, height }
    }

    #[cfg(feature = "gpu")]
    fn from_cpu_grid(buf: &ImageBuffer, ctx: &GpuContext) -> Self {
        match buf {
            ImageBuffer::F32(g) => {
                let width = g.width();
                let height = g.height();
                let bytes = bytemuck::cast_slice(g.buf());
                let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jxlit_gpu_f32"),
                    size: bytes.len().max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                ctx.queue.write_buffer(&buffer, 0, bytes);
                Self::F32 {
                    width,
                    height,
                    buffer: Arc::new(buffer),
                }
            }
            ImageBuffer::I32(g) => {
                let width = g.width();
                let height = g.height();
                let bytes = bytemuck::cast_slice(g.buf());
                let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jxlit_gpu_i32"),
                    size: bytes.len().max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                ctx.queue.write_buffer(&buffer, 0, bytes);
                Self::I32 {
                    width,
                    height,
                    buffer: Arc::new(buffer),
                }
            }
            ImageBuffer::I16(g) => {
                let width = g.width();
                let height = g.height();
                let bytes = bytemuck::cast_slice(g.buf());
                let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jxlit_gpu_i16"),
                    size: bytes.len().max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                ctx.queue.write_buffer(&buffer, 0, bytes);
                Self::I16 {
                    width,
                    height,
                    buffer: Arc::new(buffer),
                }
            }
        }
    }
}

/// Per-tile coefficient sub-grid view into a GPU color buffer.
#[derive(Debug)]
pub struct GpuMutableSubgrid<'a> {
    pub parent: &'a mut GpuImageWithRegion,
    pub channel: usize,
}

/// Mirror of [`ImageWithRegion`] for GPU-resident planar image data.
#[derive(Debug)]
pub struct GpuImageWithRegion {
    buffer: Vec<GpuImageBuffer>,
    regions: Vec<(Region, ChannelShift)>,
    color_channels: usize,
    ct_done: bool,
    blend_done: bool,
    tracker: Option<AllocTracker>,
}

impl GpuImageWithRegion {
    pub fn new(color_channels: usize, tracker: Option<&AllocTracker>) -> Self {
        Self {
            buffer: Vec::new(),
            regions: Vec::new(),
            color_channels,
            ct_done: false,
            blend_done: false,
            tracker: tracker.cloned(),
        }
    }

    pub fn color_channels(&self) -> usize {
        self.color_channels
    }

    pub fn buffer(&self) -> &[GpuImageBuffer] {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [GpuImageBuffer] {
        &mut self.buffer
    }

    pub fn regions_and_shifts(&self) -> &[(Region, ChannelShift)] {
        &self.regions
    }

    pub fn append_channel_shifted(
        &mut self,
        buffer: GpuImageBuffer,
        original_region: Region,
        shift: ChannelShift,
    ) {
        let (width, height) = shift.shift_size((original_region.width, original_region.height));
        assert_eq!(buffer.width(), width as usize);
        assert_eq!(buffer.height(), height as usize);
        self.buffer.push(buffer);
        self.regions.push((original_region, shift));
    }

    pub fn remove_color_channels(&mut self, count: usize) {
        assert!(self.color_channels >= count);
        self.buffer.drain(count..self.color_channels);
        self.regions.drain(count..self.color_channels);
        self.color_channels = count;
    }

    pub fn prepare_color_upsampling(&mut self, frame_header: &FrameHeader) {
        let upsampling_factor = frame_header.upsampling.trailing_zeros();
        for (region, shift) in &mut self.regions {
            match shift {
                ChannelShift::Raw(..=-1, _) | ChannelShift::Raw(_, ..=-1) => continue,
                ChannelShift::Raw(h, v) => {
                    *h = h.wrapping_add_unsigned(upsampling_factor);
                    *v = v.wrapping_add_unsigned(upsampling_factor);
                }
                ChannelShift::Shifts(shift) => {
                    *shift += upsampling_factor;
                }
                ChannelShift::JpegUpsampling {
                    has_h_subsample: false,
                    h_subsample: false,
                    has_v_subsample: false,
                    v_subsample: false,
                } => {
                    *shift = ChannelShift::Shifts(upsampling_factor);
                }
                ChannelShift::JpegUpsampling { .. } => {
                    panic!("unexpected chroma subsampling {shift:?}");
                }
            }
            *region = region.upsample(upsampling_factor);
        }
    }

    pub fn clone_gray(&mut self) -> Result<()> {
        unimplemented!("GPU path not implemented: clone_gray");
    }

    pub fn convert_modular_color(&mut self, _bit_depth: BitDepth) -> Result<()> {
        unimplemented!("GPU path not implemented: convert_modular_color");
    }

    pub fn fill_opaque_alpha(&mut self, _ec_info: &[jxl_image::ExtraChannelInfo]) {
        unimplemented!("GPU path not implemented: fill_opaque_alpha");
    }

    pub fn extend_from_gmodular<S: Sample>(&mut self, _gmodular: GlobalModular<S>) {
        unimplemented!("GPU path not implemented: extend_from_gmodular");
    }

    pub fn ct_done(&self) -> bool {
        self.ct_done
    }

    pub fn set_ct_done(&mut self, ct_done: bool) {
        self.ct_done = ct_done;
    }

    /// Materializes a CPU image on the GPU via raw-byte upload (cold start).
    pub fn from_cpu(cpu: &ImageWithRegion) -> std::result::Result<Self, String> {
        #[cfg(feature = "gpu")]
        {
            let ctx = GpuContext::get().ok_or_else(|| "GPU device unavailable".to_string())?;
            let mut gpu = Self::new(cpu.color_channels(), cpu.alloc_tracker());
            for (buf, (region, shift)) in cpu.buffer().iter().zip(cpu.regions_and_shifts()) {
                let buffer = GpuImageBuffer::from_cpu_grid(buf, ctx);
                gpu.append_channel_shifted(buffer, *region, *shift);
            }
            Ok(gpu)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = cpu;
            Err("GPU feature not enabled".to_string())
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        unimplemented!("GPU path not implemented: try_clone");
    }

    pub fn color_groups_with_group_id(
        &mut self,
        _frame_header: &FrameHeader,
    ) -> Vec<(u32, [GpuMutableSubgrid<'_>; 3])> {
        unimplemented!("GPU path not implemented: color_groups_with_group_id");
    }
}

/// Builds an empty GPU coefficient buffer shaped like the CPU allocator.
pub fn alloc_coefficient_buffer(
    frame_header: &FrameHeader,
    modular_region: Region,
    tracker: Option<&AllocTracker>,
) -> GpuImageWithRegion {
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let Region { width, height, .. } = modular_region;

    let mut color_buffer = GpuImageWithRegion::new(3, tracker);
    for shift in shifts_cbycr {
        let (w8, h8) = shift.shift_size((width.div_ceil(8), height.div_ceil(8)));
        let width = w8 * 8;
        let height = h8 * 8;
        #[cfg(feature = "gpu")]
        let buffer = {
            let ctx = GpuContext::get().expect("GPU context required for coefficient buffer");
            GpuImageBuffer::empty_f32(width as usize, height as usize, ctx)
        };
        #[cfg(not(feature = "gpu"))]
        let buffer = GpuImageBuffer::empty_f32(width as usize, height as usize);
        color_buffer.append_channel_shifted(buffer, modular_region, shift);
    }
    color_buffer
}

/// Sample kind + bit depth encoding for WGSL export metadata.
pub fn sample_kind_bits(sample_kind: GpuSampleKind, bit_depth: BitDepth) -> (u32, u32) {
    let kind = match sample_kind {
        GpuSampleKind::F32 => 0,
        GpuSampleKind::I32 => 1,
        GpuSampleKind::I16 => 2,
    };
    let bits = match bit_depth {
        BitDepth::IntegerSample { bits_per_sample } => bits_per_sample,
        BitDepth::FloatSample {
            bits_per_sample, ..
        } => bits_per_sample,
    };
    (kind, bits)
}
