//! Interleave transform: planar channel grids -> interleaved (HWC) `f32` buffer.
//!
//! [`run_interleave`] is the transform: it samples the channel grids chosen by
//! [`build_decoded_image`](crate::pipeline::render::frame::build_decoded_image)
//! (color, then the black channel for CMYK, then alpha), applies spot-color
//! mixing, maps coordinates through the image orientation, and writes samples
//! into an HWC `f32` buffer.
//!
//! Direct port of `jxl-oxide`'s `ImageStream` (`from_render` +
//! `write_to_buffer::<f32>`).

use jxl_image::BitDepth;

use crate::vendor::jxl_render::ImageBuffer;

pub(crate) struct SpotColor<'r> {
    pub grid: &'r ImageBuffer,
    pub start_offset_xy: (i32, i32),
    pub bit_depth: BitDepth,
    pub rgb: (f32, f32, f32),
    pub solidity: f32,
}

/// Writes the interleaved (HWC) samples into `pixels`, returning the number of
/// samples written. The output layout and channel selection are decided by
/// [`build_decoded_image`](crate::pipeline::render::frame::build_decoded_image).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_interleave(
    pixels: &mut [f32],
    grids: &[&ImageBuffer],
    bit_depth: &[BitDepth],
    start_offset_xy: &[(i32, i32)],
    spot_colors: &[SpotColor],
    orientation: u32,
    width: u32,
    height: u32,
) -> usize {
    let _run_interleave = crate::phase_guard!("run_interleave");
    let width_us = width as usize;
    let channels = grids.len();

    let mut count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let (orig_x, orig_y) = to_original_coord(orientation, width, height, x, y);
            for c in 0..channels {
                let idx = c + (x as usize + y as usize * width_us) * channels;
                let (start_x, start_y) = start_offset_xy[c];
                let (Some(px), Some(py)) = (
                    orig_x.checked_add_signed(start_x),
                    orig_y.checked_add_signed(start_y),
                ) else {
                    pixels[idx] = 0.0;
                    count += 1;
                    continue;
                };
                let grid = grids[c];
                let bd = bit_depth[c];

                if c >= 3 || spot_colors.is_empty() {
                    pixels[idx] = sample_from_grid(grid, px as usize, py as usize, bd);
                } else {
                    let mut sample = sample_from_grid(grid, px as usize, py as usize, bd);
                    for spot in spot_colors {
                        let color = [spot.rgb.0, spot.rgb.1, spot.rgb.2][c];
                        let mix = match (
                            orig_x.checked_add_signed(spot.start_offset_xy.0),
                            orig_y.checked_add_signed(spot.start_offset_xy.1),
                        ) {
                            (Some(sx), Some(sy)) => {
                                sample_from_grid(
                                    spot.grid,
                                    sx as usize,
                                    sy as usize,
                                    spot.bit_depth,
                                ) * spot.solidity
                            }
                            _ => 0.0,
                        };
                        sample = color * mix + sample * (1.0 - mix);
                    }
                    pixels[idx] = sample;
                }
                count += 1;
            }
        }
    }
    count
}

#[inline]
fn sample_from_grid(grid: &ImageBuffer, x: usize, y: usize, bit_depth: BitDepth) -> f32 {
    match grid {
        ImageBuffer::F32(g) => g.try_get_ref(x, y).copied().unwrap_or(0.0),
        ImageBuffer::I32(g) => {
            bit_depth.parse_integer_sample(g.try_get_ref(x, y).copied().unwrap_or(0))
        }
        ImageBuffer::I16(g) => {
            bit_depth.parse_integer_sample(g.try_get_ref(x, y).copied().unwrap_or(0) as i32)
        }
    }
}

#[inline]
fn to_original_coord(orientation: u32, width: u32, height: u32, x: u32, y: u32) -> (u32, u32) {
    match orientation {
        1 => (x, y),
        2 => (width - x - 1, y),
        3 => (width - x - 1, height - y - 1),
        4 => (x, height - y - 1),
        5 => (y, x),
        6 => (y, width - x - 1),
        7 => (height - y - 1, width - x - 1),
        8 => (height - y - 1, x),
        _ => unreachable!(),
    }
}
