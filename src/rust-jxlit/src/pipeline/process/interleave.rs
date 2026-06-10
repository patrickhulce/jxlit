//! Interleave post-decode step: planar [`RenderedFrame`] -> interleaved (HWC)
//! `f32` buffer.
//!
//! `run_interleave` is the transform: it selects color channels, then the black
//! channel (for CMYK), then alpha, applies spot-color mixing, and maps
//! coordinates through the image orientation, writing samples into an HWC `f32`
//! buffer. `build_decoded_image` computes the output layout, allocates the
//! buffer, runs the transform and wraps the result into a [`DecodedImage`].
//!
//! Direct port of `jxl-oxide`'s `ImageStream` (`from_render` +
//! `write_to_buffer::<f32>`).

use jxl_image::{BitDepth, ExtraChannelType};

use crate::vendor::jxl_render::{ImageBuffer, Region};

use crate::pipeline::render::frame::RenderedFrame;
use crate::{DecodeError, DecodedImage};

struct SpotColor<'r> {
    grid: &'r ImageBuffer,
    start_offset_xy: (i32, i32),
    bit_depth: BitDepth,
    rgb: (f32, f32, f32),
    solidity: f32,
}

/// Computes the output layout, allocates the HWC buffer, runs [`run_interleave`]
/// and wraps the result into a [`DecodedImage`].
pub fn build_decoded_image(rendered: &RenderedFrame) -> Result<DecodedImage, DecodeError> {
    let orientation = rendered.orientation;
    debug_assert!((1..=8).contains(&orientation));

    let Region {
        left,
        top,
        mut width,
        mut height,
    } = rendered.target_frame_region;
    if orientation >= 5 {
        std::mem::swap(&mut width, &mut height);
    }

    let image = &rendered.image;
    let fb = image.buffer();
    let color_channels = image.color_channels();
    let regions_and_shifts = image.regions_and_shifts();

    let mut grids: Vec<&ImageBuffer> = fb[..color_channels].iter().collect();
    let mut bit_depth = vec![rendered.color_bit_depth; grids.len()];
    let mut start_offset_xy: Vec<(i32, i32)> = Vec::new();
    for (region, _) in &regions_and_shifts[..color_channels] {
        start_offset_xy.push((left - region.left, top - region.top));
    }

    // Black channel (CMYK only).
    if rendered.is_cmyk {
        for (ec_idx, (ec, (region, _))) in rendered
            .extra_channels
            .iter()
            .zip(&regions_and_shifts[color_channels..])
            .enumerate()
        {
            if matches!(ec.0, ExtraChannelType::Black) {
                grids.push(&fb[color_channels + ec_idx]);
                bit_depth.push(ec.1);
                start_offset_xy.push((left - region.left, top - region.top));
                break;
            }
        }
    }

    // Alpha channel (the `stream()` variant includes alpha).
    for (ec_idx, (ec, (region, _))) in rendered
        .extra_channels
        .iter()
        .zip(&regions_and_shifts[color_channels..])
        .enumerate()
    {
        if matches!(ec.0, ExtraChannelType::Alpha { .. }) {
            grids.push(&fb[color_channels + ec_idx]);
            bit_depth.push(ec.1);
            start_offset_xy.push((left - region.left, top - region.top));
            break;
        }
    }

    // Spot colors (only mixed into RGB color channels).
    let mut spot_colors = Vec::new();
    if rendered.render_spot_color && color_channels == 3 {
        for (ec_idx, (ec, (region, _))) in rendered
            .extra_channels
            .iter()
            .zip(&regions_and_shifts[color_channels..])
            .enumerate()
        {
            if let ExtraChannelType::SpotColour {
                red,
                green,
                blue,
                solidity,
            } = ec.0
            {
                spot_colors.push(SpotColor {
                    grid: &fb[color_channels + ec_idx],
                    start_offset_xy: (left - region.left, top - region.top),
                    bit_depth: ec.1,
                    rgb: (red, green, blue),
                    solidity,
                });
            }
        }
    }

    let channels = grids.len();
    let width_us = width as usize;
    let height_us = height as usize;
    let mut pixels = vec![0.0f32; width_us * height_us * channels];

    let count = run_interleave(
        &mut pixels,
        &grids,
        &bit_depth,
        &start_offset_xy,
        &spot_colors,
        orientation,
        width,
        height,
    );

    if count != pixels.len() {
        return Err(DecodeError::new(format!(
            "expected to write {} samples, wrote {count}",
            pixels.len()
        )));
    }

    Ok(DecodedImage {
        height: height_us,
        width: width_us,
        channels,
        pixels,
    })
}

/// Writes the interleaved (HWC) samples into `pixels`, returning the number of
/// samples written. The output layout and channel selection are decided by
/// [`build_decoded_image`].
#[allow(clippy::too_many_arguments)]
fn run_interleave(
    pixels: &mut [f32],
    grids: &[&ImageBuffer],
    bit_depth: &[BitDepth],
    start_offset_xy: &[(i32, i32)],
    spot_colors: &[SpotColor],
    orientation: u32,
    width: u32,
    height: u32,
) -> usize {
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
