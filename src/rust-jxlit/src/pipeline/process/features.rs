//! Feature-rendering post-decode step: patches, splines and noise.
//!
//! Ports the feature-rendering portion of the vendored post-decode stage. Each
//! feature calls the vendored building block directly.

use std::sync::Arc;

use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::data::LfGlobal;
use crate::vendor::jxl_render::{
    Error, ImageWithRegion, IndexedFrame, Reference, Region, Result, blend, features,
};

/// Renders patches, splines and noise into the decoded color buffer in place.
#[allow(clippy::too_many_arguments)]
pub fn run_features<S: Sample>(
    frame: &IndexedFrame,
    grid: &mut ImageWithRegion,
    upsampling_valid_region: Region,
    reference_grids: [Option<Reference<S>>; 4],
    low_frequency_global: Option<&LfGlobal<S>>,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    pool: &JxlThreadPool,
) -> Result<()> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let Some(low_frequency_global) = low_frequency_global else {
        return Ok(());
    };
    let base_correlations_xb = low_frequency_global.vardct.as_ref().map(|x| {
        (
            x.lf_chan_corr.base_correlation_x,
            x.lf_chan_corr.base_correlation_b,
        )
    });

    if let Some(patches) = &low_frequency_global.patches {
        grid.upsample_nonseparable(image_header, frame_header, upsampling_valid_region, true)?;

        for patch in &patches.patches {
            let Some(ref_grid) = &reference_grids[patch.ref_idx as usize] else {
                return Err(Error::InvalidReference(patch.ref_idx));
            };
            let ref_header = ref_grid.frame.header();
            let oriented_image_region = Region::with_size(ref_header.width, ref_header.height)
                .translate(ref_header.x0, ref_header.y0);
            let ref_grid_image = Arc::clone(&ref_grid.image).run_with_image()?;
            let ref_grid_image = ref_grid_image.blend(Some(oriented_image_region), pool)?;
            blend::patch(image_header, grid, &ref_grid_image, patch)?;
        }
    }

    if let Some(splines) = &low_frequency_global.splines
        && grid.color_channels() == 3
    {
        grid.convert_modular_color(image_header.metadata.bit_depth)?;
        features::render_spline(frame_header, grid, splines, base_correlations_xb)?;
    }

    if let Some(noise) = &low_frequency_global.noise
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
