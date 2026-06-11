//! Tile-scope orchestration: run one tile's (pass-group's) transform chain.
//!
//! Given the per-tile dequantized/entropy-decoded coefficients, this runs HF
//! dequant -> chroma-from-luma -> inverse DCT (or the LF-only fallback), writing
//! pixel-domain XYB samples into the tile's color sub-grid. The flow drives the
//! parallel loop, calling [`render_tile`] per [`TileCtx`].

use jxl_modular::Sample;

use crate::pipeline::decode::{dequant, idct};
use crate::pipeline::gpu::DeviceCoefficients;
use crate::pipeline::structs::frame::FrameCtx;
use crate::pipeline::structs::tile::TileCtx;

/// Renders a single tile (JXL pass-group) into its XYB color sub-grid.
pub fn render_tile<S: Sample>(frame_ctx: &FrameCtx<'_, S>, mut tile: TileCtx<'_, S>) {
    let group_index = tile.declaration.group_index;
    let group_x = tile.declaration.group_x;
    let group_y = tile.declaration.group_y;

    let transform_high_frequency = !tile
        .declaration
        .region
        .intersection(frame_ctx.aligned_region)
        .is_empty();

    let xyb_coefficients = &mut tile.xyb_coefficients;

    if tile.low_frequency_group.hf_meta.is_none()
        || frame_ctx.high_frequency_global.is_none()
        || !transform_high_frequency
    {
        idct::run_inverse_dct(
            &frame_ctx.low_frequency_image,
            xyb_coefficients,
            group_index,
            frame_ctx.frame_header,
            frame_ctx.low_frequency_groups,
        );
        return;
    }

    let high_frequency_global = frame_ctx.high_frequency_global.unwrap();

    dequant::run_high_frequency_dequant(
        xyb_coefficients,
        group_index,
        frame_ctx.image_header,
        frame_ctx.frame_header,
        frame_ctx.low_frequency_global,
        frame_ctx.low_frequency_groups,
        high_frequency_global,
    );

    if !frame_ctx.subsampled {
        let hf_meta = tile.low_frequency_group.hf_meta.as_ref().unwrap();
        let lf_chan_corr = &frame_ctx.low_frequency_global_vardct.lf_chan_corr;
        let cfl_base_x = ((group_x % 8) * frame_ctx.group_dim / 64) as usize;
        let cfl_base_y = ((group_y % 8) * frame_ctx.group_dim / 64) as usize;

        let (gw, gh) = match xyb_coefficients {
            DeviceCoefficients::Cpu(coeffs) => (
                coeffs[0].width().div_ceil(64),
                coeffs[0].height().div_ceil(64),
            ),
            DeviceCoefficients::Gpu(_) => (0, 0),
        };

        let x_from_y = hf_meta
            .x_from_y
            .as_subgrid()
            .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
        let b_from_y = hf_meta
            .b_from_y
            .as_subgrid()
            .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
        dequant::run_chroma_from_luma_high_frequency(
            xyb_coefficients,
            &x_from_y,
            &b_from_y,
            lf_chan_corr,
        );
    }

    idct::run_inverse_dct(
        &frame_ctx.low_frequency_image,
        xyb_coefficients,
        group_index,
        frame_ctx.frame_header,
        frame_ctx.low_frequency_groups,
    );
}
