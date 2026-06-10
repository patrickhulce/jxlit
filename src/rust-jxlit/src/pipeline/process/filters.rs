//! Loop-filter post-decode step: Gabor-like and edge-preserving filters.
//!
//! Ports the loop-filter portion of the vendored post-decode stage. Each filter
//! calls the vendored building block directly; the scratch buffer is shared
//! between the two filters when both are enabled.

use std::collections::HashMap;

use jxl_grid::AlignedGrid;
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::data::LfGroup;
use crate::vendor::jxl_frame::filter::{EdgePreservingFilter, Gabor};
use crate::vendor::jxl_render::{ImageWithRegion, IndexedFrame, Region, Result, filter};

/// Applies the Gabor-like and edge-preserving loop filters to the decoded color
/// buffer in place.
pub fn run_loop_filters<S: Sample>(
    frame: &IndexedFrame,
    fb: &mut ImageWithRegion,
    color_padded_region: Region,
    low_frequency_groups: &HashMap<u32, LfGroup<S>>,
    pool: &JxlThreadPool,
) -> Result<()> {
    let image_header = frame.image_header();
    let frame_header = frame.header();

    let mut scratch_buffer = None;
    if let Gabor::Enabled(weights) = frame_header.restoration_filter.gab {
        if fb.color_channels() < 3 {
            fb.clone_gray()?;
        }

        fb.convert_modular_color(image_header.metadata.bit_depth)?;
        let mut fb_scratch = {
            let tracker = fb.alloc_tracker();
            let width = color_padded_region.width as usize;
            let height = color_padded_region.height as usize;
            [
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
            ]
        };
        filter::apply_gabor_like(fb, color_padded_region, &mut fb_scratch, weights, pool);
        scratch_buffer = Some(fb_scratch);
    }

    if let EdgePreservingFilter::Enabled(epf_params) = &frame_header.restoration_filter.epf {
        if fb.color_channels() < 3 {
            fb.clone_gray()?;
        }

        fb.convert_modular_color(image_header.metadata.bit_depth)?;
        let fb_scratch = if let Some(buffer) = scratch_buffer {
            buffer
        } else {
            let tracker = fb.alloc_tracker();
            let width = color_padded_region.width as usize;
            let height = color_padded_region.height as usize;
            [
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
            ]
        };
        filter::apply_epf(
            fb,
            fb_scratch,
            color_padded_region,
            low_frequency_groups,
            frame_header,
            epf_params,
            pool,
        );
    }

    Ok(())
}
