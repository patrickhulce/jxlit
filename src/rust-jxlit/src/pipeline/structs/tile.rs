//! Per-tile (JXL pass-group) structures.
//!
//! In JPEG XL a frame is divided into *groups* (square tiles of `group_dim`,
//! default 256x256 samples); an 8x8 block of groups forms an *LF group*. A
//! *pass-group* is one group decoded within one pass. We call a pass-group a
//! "tile"; `group_index` is the JXL group index and `low_frequency_group_index`
//! is the JXL LF-group index that contains it.

use jxl_grid::MutableSubgrid;
use jxl_modular::Sample;

use crate::vendor::jxl_frame::data::LfGroup;
use crate::vendor::jxl_render::Region;

/// Cheap, immutable identity/geometry of one tile (JXL pass-group).
#[derive(Clone, Copy)]
pub struct TileDeclaration {
    /// JXL group index within the frame.
    pub group_index: u32,
    /// JXL LF-group index containing this group.
    pub low_frequency_group_index: u32,
    /// Group column (in group units).
    pub group_x: u32,
    /// Group row (in group units).
    pub group_y: u32,
    /// The group's full (`group_dim`-sized) pixel region, used to test overlap
    /// with the decode region when deciding whether to run the HF transform.
    pub region: Region,
}

/// Live per-tile decode context: the tile identity, plus the in-place XYB
/// coefficient sub-grids (carved from the frame color buffer) and a reference to
/// the tile's decoded LF group.
pub struct TileCtx<'a, S: Sample> {
    pub declaration: TileDeclaration,
    /// XYB coefficient sub-grids (X, Y, B) into the frame color buffer.
    pub xyb_coefficients: [MutableSubgrid<'a, f32>; 3],
    /// The decoded LF group covering this tile.
    pub low_frequency_group: &'a LfGroup<S>,
}
