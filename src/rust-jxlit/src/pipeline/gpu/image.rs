//! GPU-side mirrors of the vendored image buffer types.
//!
//! Fields and method names match [`crate::vendor::jxl_render::ImageWithRegion`] so
//! the pipeline can share structure; bodies are placeholders until real GPU
//! allocation and kernels land.

use jxl_grid::AllocTracker;
use jxl_image::BitDepth;
use jxl_modular::{ChannelShift, Sample};

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::GlobalModular;
use crate::vendor::jxl_render::{ImageBuffer, ImageWithRegion, Region, Result};

/// Mirror of [`ImageBuffer`] for GPU-resident channel storage.
#[derive(Debug)]
pub enum GpuImageBuffer {
    F32 { width: usize, height: usize },
    I32 { width: usize, height: usize },
    I16 { width: usize, height: usize },
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

    /// Builds a GPU mirror with the same channel geometry as a CPU image (placeholder upload).
    pub fn from_cpu_placeholder(cpu: &ImageWithRegion) -> Self {
        let mut gpu = Self::new(cpu.color_channels(), cpu.alloc_tracker());
        for (buf, (region, shift)) in cpu.buffer().iter().zip(cpu.regions_and_shifts()) {
            let buffer = match buf {
                ImageBuffer::F32(g) => GpuImageBuffer::F32 {
                    width: g.width(),
                    height: g.height(),
                },
                ImageBuffer::I32(g) => GpuImageBuffer::I32 {
                    width: g.width(),
                    height: g.height(),
                },
                ImageBuffer::I16(g) => GpuImageBuffer::I16 {
                    width: g.width(),
                    height: g.height(),
                },
            };
            gpu.append_channel_shifted(buffer, *region, *shift);
        }
        gpu
    }

    pub fn try_clone(&self) -> Result<Self> {
        unimplemented!("GPU path not implemented: try_clone");
    }

    /// Carves per-group coefficient sub-grids (mirrors the CPU layout).
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
        let buffer = GpuImageBuffer::F32 {
            width: width as usize,
            height: height as usize,
        };
        color_buffer.append_channel_shifted(buffer, modular_region, shift);
    }
    color_buffer
}

/// Converts a CPU [`ImageBuffer`] reference for export sampling (GPU path only).
pub fn as_cpu_buffer_ref(_buffer: &GpuImageBuffer) -> &ImageBuffer {
    unimplemented!("GPU path not implemented: download buffer for export");
}
