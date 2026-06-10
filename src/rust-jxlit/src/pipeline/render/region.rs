//! Region-scope orchestration: derive the decode regions for a frame.
//!
//! `render_region` computes the frame-level region set (the part of the frame to
//! decode, padded for filters/upsampling); `build_vardct_regions` derives the
//! VarDCT-internal aligned/modular regions that additionally depend on the
//! frame's modular configuration.

use jxl_modular::Sample;

use crate::vendor::jxl_frame::FrameHeader;
use crate::vendor::jxl_frame::data::GlobalModular;
use crate::vendor::jxl_render::{IndexedFrame, Region, modular, util};

/// Frame-level region set produced by [`render_region`].
pub struct FrameRegions {
    pub color_padded_region: Region,
    pub upsampling_valid_region: Region,
}

/// VarDCT-internal region set produced by [`build_vardct_regions`].
pub struct VarDctRegions {
    pub aligned_region: Region,
    pub modular_region: Region,
    pub modular_lf_region: Region,
}

/// Derives the frame-level regions to decode from the requested image region.
pub fn render_region(frame: &IndexedFrame, image_region: Region) -> FrameRegions {
    let image_header = frame.image_header();
    let frame_header = frame.header();

    let frame_region = util::image_region_to_frame(frame, image_region, false);
    let frame_region = util::pad_lf_region(frame_header, frame_region);

    let upsampled_full_frame_region =
        Region::with_size(frame_header.sample_width(1), frame_header.sample_height(1));
    let upsampling_valid_region = util::pad_upsampling(image_header, frame_header, frame_region)
        .intersection(upsampled_full_frame_region);

    let full_frame_region = Region::with_size(
        frame_header.color_sample_width(),
        frame_header.color_sample_height(),
    );
    let color_padded_region = util::pad_color_region(image_header, frame_header, frame_region)
        .intersection(full_frame_region);

    FrameRegions {
        color_padded_region,
        upsampling_valid_region,
    }
}

/// Derives the VarDCT-internal aligned and modular regions from the (color
/// padded) decode `region` and the frame's modular configuration.
pub fn build_vardct_regions<S: Sample>(
    frame_header: &FrameHeader,
    region: Region,
    width_rounded: usize,
    height_rounded: usize,
    gmodular: &GlobalModular<S>,
) -> VarDctRegions {
    let aligned_region = region.container_aligned(frame_header.group_dim());
    let aligned_lf_region = {
        // group_dim is a multiple of 8
        let aligned_region_div8 = Region {
            left: aligned_region.left / 8,
            top: aligned_region.top / 8,
            width: aligned_region.width / 8,
            height: aligned_region.height / 8,
        };
        if frame_header.flags.skip_adaptive_lf_smoothing() {
            aligned_region_div8
        } else {
            aligned_region_div8.pad(1)
        }
        .container_aligned(frame_header.group_dim())
    };

    let aligned_region = aligned_region.intersection(Region::with_size(
        width_rounded as u32,
        height_rounded as u32,
    ));
    let aligned_lf_region = aligned_lf_region.intersection(Region::with_size(
        width_rounded as u32 / 8,
        height_rounded as u32 / 8,
    ));

    let modular_region =
        modular::compute_modular_region(frame_header, gmodular, aligned_region, false);
    let modular_lf_region =
        modular::compute_modular_region(frame_header, gmodular, aligned_lf_region, true)
            .intersection(Region::with_size(
                width_rounded as u32 / 8,
                height_rounded as u32 / 8,
            ));

    VarDctRegions {
        aligned_region,
        modular_region,
        modular_lf_region,
    }
}
