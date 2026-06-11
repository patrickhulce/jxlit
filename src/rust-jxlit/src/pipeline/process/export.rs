//! Tiered export from rendered channel grids into the final flat `f32` buffer.

use jxl_grid::AlignedGrid;
use jxl_image::BitDepth;
use jxl_modular::ChannelShift;

use crate::pipeline::process::interleave::{SpotColor, sample_at, to_original_coord};
use crate::vendor::jxl_render::ImageBuffer;

pub(crate) struct PlanarMemcpy<'a> {
    pub channels: Vec<&'a AlignedGrid<f32>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_planar_memcpy<'a>(
    grids: &[&'a ImageBuffer],
    offsets: &[(i32, i32)],
    shifts: &[ChannelShift],
    orientation: u32,
    width: usize,
    height: usize,
    spots_fused: bool,
    has_spot_colors: bool,
) -> Option<PlanarMemcpy<'a>> {
    if !spots_fused && has_spot_colors {
        return None;
    }

    let mut channels = Vec::with_capacity(grids.len());
    for (grid, (&offset, &shift)) in grids.iter().zip(offsets.iter().zip(shifts)) {
        channels.push(can_memcpy_planar_channel(
            grid,
            offset,
            shift,
            orientation,
            width,
            height,
        )?);
    }
    Some(PlanarMemcpy { channels })
}

fn can_memcpy_planar_channel(
    grid: &ImageBuffer,
    offset: (i32, i32),
    shift: ChannelShift,
    orientation: u32,
    width: usize,
    height: usize,
) -> Option<&AlignedGrid<f32>> {
    if orientation != 1 || offset != (0, 0) || shift != ChannelShift::from_shift(0) {
        return None;
    }
    let ImageBuffer::F32(g) = grid else {
        return None;
    };
    if g.width() == width && g.height() == height {
        Some(g)
    } else {
        None
    }
}

pub(crate) fn export_planar_memcpy(
    pixels: &mut [f32],
    channels: &[&AlignedGrid<f32>],
    plane_size: usize,
) {
    let _export_planar_memcpy = crate::phase_guard!("export_planar_memcpy");
    for (c, grid) in channels.iter().enumerate() {
        let dst = &mut pixels[c * plane_size..(c + 1) * plane_size];
        dst.copy_from_slice(grid.buf());
    }
}

/// Per-pixel planar export when memcpy preconditions are not met.
#[allow(clippy::too_many_arguments)]
pub(crate) fn export_planar_sample(
    pixels: &mut [f32],
    grids: &[&ImageBuffer],
    bit_depth: &[BitDepth],
    start_offset_xy: &[(i32, i32)],
    spot_colors: &[SpotColor],
    orientation: u32,
    width: u32,
    height: u32,
) -> usize {
    let _export_planar_sample = crate::phase_guard!("export_planar_sample");
    let width_us = width as usize;
    let channels = grids.len();
    let plane_size = width_us * height as usize;

    let mut count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let (orig_x, orig_y) = to_original_coord(orientation, width, height, x, y);
            for c in 0..channels {
                let idx = c * plane_size + y as usize * width_us + x as usize;
                pixels[idx] = sample_at(
                    grids,
                    bit_depth,
                    start_offset_xy,
                    spot_colors,
                    c,
                    orig_x,
                    orig_y,
                );
                count += 1;
            }
        }
    }
    count
}
