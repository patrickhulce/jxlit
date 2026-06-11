//! CPU/GPU device placement for decode buffers.
//!
//! Mirrors the vendored [`ImageWithRegion`] / coefficient views so pipeline steps
//! can fork between the existing CPU path and future GPU kernels.

#![allow(dead_code)]

pub mod availability;
pub mod device;
pub mod environment;
pub mod image;
pub mod kernels;

pub use device::{
    Device, DeviceCoefficients, DeviceColorGroups, DeviceImage, build_coefficient_buffer, from_cpu,
    from_cpu_arc, into_cpu_arc,
};
pub use environment::GpuEnvironment;
pub use image::GpuImageWithRegion;
