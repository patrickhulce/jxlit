//! Frame-scope structures.

use std::collections::HashMap;

use jxl_image::ImageHeader;
use jxl_modular::Sample;

use crate::pipeline::gpu::{Device, DeviceImage, GpuEnvironment};
use crate::types::DecodeOptions;
use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGlobalVarDct, LfGroup};
use crate::vendor::jxl_render::Region;

use super::tile::TileDeclaration;

/// Cheap, immutable description of a frame: header refs, geometry, the VarDCT
/// region set, and the per-tile (pass-group) declarations. Built once the
/// modular configuration (and thus the modular region) is known, then consumed
/// to build the per-tile contexts and the [`FrameCtx`].
pub struct FrameDeclaration<'a> {
    pub frame_header: &'a FrameHeader,
    pub image_header: &'a ImageHeader,
    pub group_dim: u32,
    pub subsampled: bool,
    pub aligned_region: Region,
    pub modular_region: Region,
    pub tiles: Vec<TileDeclaration>,
}

/// Rich frame context used by the per-tile transform phase: the decoded global
/// tables (filled as decoding proceeds) plus the dequantized low-frequency
/// image. The HF coefficient color buffer is not held here; it is the mutable
/// target carved per-tile into [`super::tile::TileCtx`].
pub struct FrameCtx<'a, S: Sample> {
    #[allow(dead_code)]
    pub device: Device,
    pub options: DecodeOptions,
    pub env: GpuEnvironment,
    pub frame_header: &'a FrameHeader,
    pub image_header: &'a ImageHeader,
    pub low_frequency_global: &'a LfGlobal<S>,
    pub low_frequency_global_vardct: &'a LfGlobalVarDct,
    pub high_frequency_global: Option<&'a HfGlobal>,
    pub low_frequency_groups: &'a HashMap<u32, LfGroup<S>>,
    pub low_frequency_image: DeviceImage,
    pub aligned_region: Region,
    pub group_dim: u32,
    pub subsampled: bool,
}
