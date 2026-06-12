//! CPU/GPU device placement for decode buffers.
//!
//! Mirrors the vendored [`ImageWithRegion`] / coefficient views so pipeline steps
//! can fork between the existing CPU path and future GPU kernels.

#![allow(dead_code)]

pub mod availability;
#[cfg(feature = "gpu")]
pub mod color_transform;
#[cfg(not(feature = "gpu"))]
pub mod color_transform {
    use crate::vendor::jxl_render::RenderContext;
    use jxl_color::{ColorTransformGpuOp, GpuTransformUnsupported};

    pub fn build_gpu_plan(
        _ctx: &RenderContext,
    ) -> Result<Vec<ColorTransformGpuOp>, GpuTransformUnsupported> {
        Err(GpuTransformUnsupported::IccToIcc)
    }
}
pub mod context;
pub mod device;
pub mod environment;
pub mod image;
pub mod kernels;
#[cfg(feature = "gpu")]
pub mod modular;
#[cfg(feature = "gpu")]
pub mod pipeline;
pub mod transfer;

pub use context::GpuPixelBuffer;
pub use device::{
    Device, DeviceCoefficients, DeviceColorGroups, DeviceImage, build_coefficient_buffer, from_cpu,
    from_cpu_arc, into_cpu_arc,
};
pub use environment::GpuEnvironment;
pub use image::GpuImageWithRegion;
pub use transfer::{download_pixels, upload_pixels};
