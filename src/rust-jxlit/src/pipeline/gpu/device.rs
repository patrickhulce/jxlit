//! Device placement wrappers: CPU vs GPU buffer enums and selection logic.

use std::sync::Arc;

use jxl_grid::{AllocTracker, MutableSubgrid};
use jxl_image::BitDepth;
use jxl_modular::{ChannelShift, Sample};

use crate::types::{DecodeOptions, Hardware};

use super::availability;
use super::environment::GpuEnvironment;
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::GlobalModular;
use crate::vendor::jxl_frame::header::Encoding;
use crate::vendor::jxl_render::{ImageBuffer, ImageWithRegion, Region, Result};

use super::image::{GpuImageWithRegion, GpuMutableSubgrid, alloc_coefficient_buffer};

/// Where decode buffers for a frame live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl Device {
    /// Picks the device for a frame based on decode options, encoding, and GPU env.
    pub fn select(
        options: &DecodeOptions,
        frame_header: &FrameHeader,
        env: GpuEnvironment,
    ) -> Self {
        if options.hardware == Hardware::Gpu
            && frame_header.encoding == Encoding::VarDct
            && env.device_available
        {
            Self::Gpu
        } else {
            Self::Cpu
        }
    }

    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }
}

/// Planar image buffer that may live on CPU or GPU.
#[derive(Debug)]
pub enum DeviceImage {
    Cpu(ImageWithRegion),
    Gpu(GpuImageWithRegion),
}

impl DeviceImage {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Gpu(_) => Device::Gpu,
        }
    }

    pub fn as_cpu(&self) -> Option<&ImageWithRegion> {
        match self {
            Self::Cpu(image) => Some(image),
            Self::Gpu(_) => None,
        }
    }

    pub fn as_cpu_mut(&mut self) -> Option<&mut ImageWithRegion> {
        match self {
            Self::Cpu(image) => Some(image),
            Self::Gpu(_) => None,
        }
    }

    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuImageWithRegion> {
        match self {
            Self::Cpu(_) => None,
            Self::Gpu(image) => Some(image),
        }
    }

    pub fn into_cpu(self) -> ImageWithRegion {
        match self {
            Self::Cpu(image) => image,
            Self::Gpu(_) => panic!("expected CPU image"),
        }
    }

    pub fn color_channels(&self) -> usize {
        match self {
            Self::Cpu(image) => image.color_channels(),
            Self::Gpu(image) => image.color_channels(),
        }
    }

    pub fn buffer(&self) -> DeviceImageBufferView<'_> {
        match self {
            Self::Cpu(image) => DeviceImageBufferView::Cpu(image.buffer()),
            Self::Gpu(image) => DeviceImageBufferView::Gpu(image.buffer()),
        }
    }

    pub fn regions_and_shifts(&self) -> &[(Region, ChannelShift)] {
        match self {
            Self::Cpu(image) => image.regions_and_shifts(),
            Self::Gpu(image) => image.regions_and_shifts(),
        }
    }

    pub fn remove_color_channels(&mut self, count: usize) {
        match self {
            Self::Cpu(image) => image.remove_color_channels(count),
            Self::Gpu(image) => image.remove_color_channels(count),
        }
    }

    pub fn prepare_color_upsampling(&mut self, frame_header: &FrameHeader) {
        match self {
            Self::Cpu(image) => image.prepare_color_upsampling(frame_header),
            Self::Gpu(image) => image.prepare_color_upsampling(frame_header),
        }
    }

    pub fn extend_from_gmodular<S: Sample>(&mut self, gmodular: GlobalModular<S>) {
        match self {
            Self::Cpu(image) => image.extend_from_gmodular(gmodular),
            Self::Gpu(image) => image.extend_from_gmodular(gmodular),
        }
    }

    pub fn color_groups_with_group_id(
        &mut self,
        frame_header: &FrameHeader,
    ) -> DeviceColorGroups<'_> {
        match self {
            Self::Cpu(image) => {
                DeviceColorGroups::Cpu(image.color_groups_with_group_id(frame_header))
            }
            Self::Gpu(image) => {
                DeviceColorGroups::Gpu(image.color_groups_with_group_id(frame_header))
            }
        }
    }

    pub fn convert_modular_color(&mut self, bit_depth: BitDepth) -> Result<()> {
        match self {
            Self::Cpu(image) => image.convert_modular_color(bit_depth),
            Self::Gpu(image) => image.convert_modular_color(bit_depth),
        }
    }

    pub fn clone_gray(&mut self) -> Result<()> {
        match self {
            Self::Cpu(image) => image.clone_gray(),
            Self::Gpu(image) => image.clone_gray(),
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::Cpu(image) => Ok(Self::Cpu(image.try_clone()?)),
            Self::Gpu(image) => Ok(Self::Gpu(image.try_clone()?)),
        }
    }

    pub fn ct_done(&self) -> bool {
        match self {
            Self::Cpu(image) => image.ct_done(),
            Self::Gpu(image) => image.ct_done(),
        }
    }

    pub fn set_ct_done(&mut self, ct_done: bool) {
        match self {
            Self::Cpu(image) => image.set_ct_done(ct_done),
            Self::Gpu(image) => image.set_ct_done(ct_done),
        }
    }

    /// Ensures the image is CPU-resident, downloading from GPU when needed.
    pub fn ensure_cpu(&mut self) -> Result<&mut ImageWithRegion> {
        match self {
            Self::Cpu(image) => Ok(image),
            Self::Gpu(_) => unimplemented!("GPU path not implemented: download image"),
        }
    }
}

/// Borrowed view of channel buffers for export / interleave.
pub enum DeviceImageBufferView<'a> {
    Cpu(&'a [ImageBuffer]),
    Gpu(&'a [super::image::GpuImageBuffer]),
}

/// Per-group coefficient sub-grids carved from a color buffer.
pub enum DeviceColorGroups<'a> {
    Cpu(Vec<(u32, [MutableSubgrid<'a, f32>; 3])>),
    Gpu(Vec<(u32, [GpuMutableSubgrid<'a>; 3])>),
}

/// Per-tile XYB coefficient sub-grids.
pub enum DeviceCoefficients<'a> {
    Cpu([MutableSubgrid<'a, f32>; 3]),
    Gpu([GpuMutableSubgrid<'a>; 3]),
}

impl<'a> DeviceCoefficients<'a> {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Gpu(_) => Device::Gpu,
        }
    }

    pub fn as_cpu_mut(&mut self) -> Option<&mut [MutableSubgrid<'a, f32>; 3]> {
        match self {
            Self::Cpu(coeffs) => Some(coeffs),
            Self::Gpu(_) => None,
        }
    }

    pub fn as_gpu_mut(&mut self) -> Option<&mut [GpuMutableSubgrid<'a>; 3]> {
        match self {
            Self::Cpu(_) => None,
            Self::Gpu(coeffs) => Some(coeffs),
        }
    }

    /// Ensures coefficients are CPU-resident, downloading from GPU when needed.
    pub fn ensure_cpu_mut(&mut self) -> Result<&mut [MutableSubgrid<'a, f32>; 3]> {
        match self {
            Self::Cpu(coeffs) => Ok(coeffs),
            Self::Gpu(_) => unimplemented!("GPU path not implemented: download coefficients"),
        }
    }
}

/// Allocates an empty coefficient buffer on the selected device.
pub fn build_coefficient_buffer(
    device: Device,
    frame_header: &FrameHeader,
    modular_region: Region,
    tracker: Option<&AllocTracker>,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> Result<DeviceImage> {
    let use_gpu = device.is_gpu()
        && availability::read_pass_group_available(frame_header, 0, 0, options, env);
    match if use_gpu { Device::Gpu } else { Device::Cpu } {
        Device::Cpu => {
            let image = super::super::render::frame::build_coefficient_buffer_cpu(
                frame_header,
                modular_region,
                tracker,
            )?;
            Ok(DeviceImage::Cpu(image))
        }
        Device::Gpu => Ok(DeviceImage::Gpu(alloc_coefficient_buffer(
            frame_header,
            modular_region,
            tracker,
        ))),
    }
}

/// Wraps a CPU image in a device-aware envelope (default path for modular flow).
pub fn from_cpu(image: ImageWithRegion) -> DeviceImage {
    DeviceImage::Cpu(image)
}

/// Wraps a post-blend CPU image for the color-transform stage.
pub fn from_cpu_arc(image: Arc<ImageWithRegion>) -> Arc<DeviceImage> {
    Arc::new(DeviceImage::Cpu(Arc::try_unwrap(image).unwrap_or_else(
        |arc| arc.as_ref().try_clone().expect("clone image"),
    )))
}

/// Converts a device image back to CPU for vendored APIs that still require it.
pub fn into_cpu_arc(image: Arc<DeviceImage>) -> Result<Arc<ImageWithRegion>> {
    match Arc::try_unwrap(image) {
        Ok(DeviceImage::Cpu(image)) => Ok(Arc::new(image)),
        Ok(DeviceImage::Gpu(_)) => unimplemented!("GPU path not implemented: download image"),
        Err(arc) => match arc.as_ref() {
            DeviceImage::Cpu(image) => Ok(Arc::new(image.try_clone()?)),
            DeviceImage::Gpu(_) => unimplemented!("GPU path not implemented: download image"),
        },
    }
}
