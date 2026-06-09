//! Shared post-decode stage (both VarDCT and Modular paths feed into this).
//!
//! Ports the post-decode body of the vendored `jxl_render::render::render_frame`
//! (everything after the coefficient decode): Gabor-like and edge-preserving
//! loop filters, feature rendering (patches/splines/noise), non-separable
//! upsampling, and the "color for record" conversion. Each step calls the
//! vendored building block directly.

use jxl_grid::AlignedGrid;
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::filter::{EdgePreservingFilter, Gabor};
use crate::vendor::jxl_render::{
    Error, ImageWithRegion, IndexedFrame, Reference, Region, RenderCache, Result, blend, features,
    filter, util,
};

/// Runs loop filters, feature rendering, upsampling and color-for-record on the
/// decoded color buffer `fb`, in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<S: Sample>(
    frame: &IndexedFrame,
    fb: &mut ImageWithRegion,
    color_padded_region: Region,
    upsampling_valid_region: Region,
    reference_refs: [Option<Reference<S>>; 4],
    cache: &mut RenderCache<S>,
    frame_visibility: (usize, usize),
    pool: &JxlThreadPool,
) -> Result<()> {
    let image_header = frame.image_header();
    let frame_header = frame.header();

    let color_channels = fb.color_channels();
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
            &cache.lf_groups,
            frame_header,
            epf_params,
            pool,
        );
    }

    // Truncate cloned gray channels.
    fb.remove_color_channels(color_channels);

    fb.prepare_color_upsampling(frame_header);

    render_features(
        frame,
        fb,
        upsampling_valid_region,
        reference_refs,
        cache,
        frame_visibility.0,
        frame_visibility.1,
        pool,
    )?;

    fb.upsample_nonseparable(image_header, frame_header, upsampling_valid_region, false)?;

    if !frame_header.save_before_ct && !frame_header.is_last {
        util::convert_color_for_record(image_header, frame_header.do_ycbcr, fb, pool)?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_features<S: Sample>(
    frame: &IndexedFrame,
    grid: &mut ImageWithRegion,
    upsampling_valid_region: Region,
    reference_grids: [Option<Reference<S>>; 4],
    cache: &mut RenderCache<S>,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    pool: &JxlThreadPool,
) -> Result<()> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let Some(lf_global) = cache.lf_global.as_ref() else {
        return Ok(());
    };
    let base_correlations_xb = lf_global.vardct.as_ref().map(|x| {
        (
            x.lf_chan_corr.base_correlation_x,
            x.lf_chan_corr.base_correlation_b,
        )
    });

    if let Some(patches) = &lf_global.patches {
        grid.upsample_nonseparable(image_header, frame_header, upsampling_valid_region, true)?;

        for patch in &patches.patches {
            let Some(ref_grid) = &reference_grids[patch.ref_idx as usize] else {
                return Err(Error::InvalidReference(patch.ref_idx));
            };
            let ref_header = ref_grid.frame.header();
            let oriented_image_region = Region::with_size(ref_header.width, ref_header.height)
                .translate(ref_header.x0, ref_header.y0);
            let ref_grid_image = std::sync::Arc::clone(&ref_grid.image).run_with_image()?;
            let ref_grid_image = ref_grid_image.blend(Some(oriented_image_region), pool)?;
            blend::patch(image_header, grid, &ref_grid_image, patch)?;
        }
    }

    if let Some(splines) = &lf_global.splines
        && grid.color_channels() == 3
    {
        grid.convert_modular_color(image_header.metadata.bit_depth)?;
        features::render_spline(frame_header, grid, splines, base_correlations_xb)?;
    }

    if let Some(noise) = &lf_global.noise
        && grid.color_channels() == 3
    {
        grid.convert_modular_color(image_header.metadata.bit_depth)?;
        features::render_noise(
            frame.header(),
            visible_frames_num,
            invisible_frames_num,
            base_correlations_xb,
            grid,
            noise,
            pool,
        )?;
    }

    Ok(())
}
