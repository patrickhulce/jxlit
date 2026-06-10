//! Tile parse stage: enumerate the per-tile (JXL pass-group) descriptors and
//! build the live per-tile decode contexts.

use std::collections::HashMap;

use jxl_modular::Sample;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::LfGroup;
use crate::vendor::jxl_render::{ImageWithRegion, Region};

use crate::pipeline::structs::tile::{TileCtx, TileDeclaration};

/// Enumerates the per-group tile descriptors of a frame (one per JXL group).
///
/// Passes are an outer iteration axis, not part of a tile's identity, so there
/// is exactly one [`TileDeclaration`] per group.
pub fn build_tiles(frame_header: &FrameHeader) -> Vec<TileDeclaration> {
    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let num_groups = frame_header.num_groups();

    let mut tiles = Vec::with_capacity(num_groups as usize);
    for group_index in 0..num_groups {
        let group_x = group_index % groups_per_row;
        let group_y = group_index / groups_per_row;
        tiles.push(TileDeclaration {
            group_index,
            low_frequency_group_index: frame_header.lf_group_idx_from_group_idx(group_index),
            group_x,
            group_y,
            region: Region {
                left: (group_x * group_dim) as i32,
                top: (group_y * group_dim) as i32,
                width: group_dim,
                height: group_dim,
            },
        });
    }
    tiles
}

/// Builds the live per-tile decode contexts by carving per-group coefficient
/// sub-grids out of the frame color buffer and pairing each with its tile
/// descriptor and decoded LF group. Tiles whose LF group is absent are skipped
/// (matching the upstream renderer).
pub fn build_tile_contexts<'a, S: Sample>(
    tiles: &[TileDeclaration],
    color_buffer: &'a mut ImageWithRegion,
    frame_header: &FrameHeader,
    low_frequency_groups: &'a HashMap<u32, LfGroup<S>>,
) -> Vec<TileCtx<'a, S>> {
    let by_group: HashMap<u32, TileDeclaration> =
        tiles.iter().map(|tile| (tile.group_index, *tile)).collect();

    color_buffer
        .color_groups_with_group_id(frame_header)
        .into_iter()
        .filter_map(|(group_index, xyb_coefficients)| {
            let declaration = by_group.get(&group_index).copied()?;
            let low_frequency_group =
                low_frequency_groups.get(&declaration.low_frequency_group_index)?;
            Some(TileCtx {
                declaration,
                xyb_coefficients,
                low_frequency_group,
            })
        })
        .collect()
}
