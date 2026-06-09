//! Parse stages: container demux, image header/ICC, render-context setup, frame
//! loading, and frame-plan (tile/group) extraction.
//!
//! These stages reproduce the glue logic that `jxl-oxide` keeps private
//! (`UninitializedJxlImage::try_init` and `JxlImageInner::feed_bytes_inner`) on
//! top of the narrow sub-crates, so the resulting `RenderContext` is identical
//! to what the umbrella crate would have built.

use std::sync::Arc;

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_frame::{Frame, FrameContext};
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;
use jxl_render::{Region, RenderContext};
use jxl_threadpool::JxlThreadPool;

use crate::DecodeError;

/// Parsed image header along with the decoded embedded ICC profile (if any) and
/// the byte offset of the first real frame within the codestream.
pub struct ParsedHeader {
    pub image_header: Arc<ImageHeader>,
    pub embedded_icc: Option<Vec<u8>>,
    pub offset: usize,
}

/// Tile/group structure of a frame: the parse/render boundary.
///
/// Each [`TileJob`] identifies a pass-group tile by `(pass_idx, group_idx)`. The
/// compressed bytes for a tile are fetched on demand by the renderer via
/// `Frame::pass_group_bitstream`, so this plan holds no borrows and can be
/// dispatched across threads later.
///
/// Fields are not yet consumed by the renderer (which still delegates to
/// `jxl-render`); they exist to lock in the parse/render tile boundary.
#[allow(dead_code)]
pub struct FramePlan {
    pub group_dim: u32,
    pub groups_per_row: u32,
    pub num_groups: u32,
    pub num_passes: u32,
    pub tiles: Vec<TileJob>,
}

/// A single pass-group tile to be rendered.
#[allow(dead_code)]
pub struct TileJob {
    pub pass_idx: u32,
    pub group_idx: u32,
    pub lf_group_idx: u32,
    pub region: Region,
}

/// Demuxes the JPEG XL container and returns the concatenated codestream bytes.
///
/// Auxiliary boxes (Exif/XMP/jbrd) are ignored, matching the current decoder.
pub fn read_codestream(input: &[u8]) -> Result<Vec<u8>, DecodeError> {
    let mut parser = ContainerParser::new();
    let mut codestream = Vec::new();

    let mut offset = 0;
    while offset < input.len() {
        for event in parser.feed_bytes(&input[offset..]) {
            if let ParseEvent::Codestream(buf) =
                event.map_err(|e| DecodeError::new(e.to_string()))?
            {
                codestream.extend_from_slice(buf)
            }
        }
        let consumed = parser.previous_consumed_bytes();
        if consumed == 0 {
            break;
        }
        offset += consumed;
    }

    Ok(codestream)
}

/// Parses the image header and embedded ICC profile, returning the offset of the
/// first real frame (skipping the preview frame if present).
pub fn read_header(codestream: &[u8]) -> Result<ParsedHeader, DecodeError> {
    let mut bitstream = Bitstream::new(codestream);
    let image_header =
        ImageHeader::parse(&mut bitstream, ()).map_err(|e| DecodeError::new(e.to_string()))?;

    let embedded_icc = if image_header.metadata.colour_encoding.want_icc() {
        let icc = jxl_color::icc::read_icc(&mut bitstream)
            .map_err(|e| DecodeError::new(e.to_string()))?;
        let icc = jxl_color::icc::decode_icc(&icc).map_err(|e| DecodeError::new(e.to_string()))?;
        Some(icc)
    } else {
        None
    };
    bitstream
        .zero_pad_to_byte()
        .map_err(|e| DecodeError::new(e.to_string()))?;

    let image_header = Arc::new(image_header);

    let skip_bytes = if image_header.metadata.preview.is_some() {
        let frame = Frame::parse(
            &mut bitstream,
            FrameContext {
                image_header: image_header.clone(),
                tracker: None,
                pool: JxlThreadPool::none(),
            },
        )
        .map_err(|e| DecodeError::new(e.to_string()))?;
        frame.toc().total_byte_size()
    } else {
        0
    };

    let offset = bitstream.num_read_bits() / 8 + skip_bytes;
    Ok(ParsedHeader {
        image_header,
        embedded_icc,
        offset,
    })
}

/// Builds a single-threaded, CMS-less render context (matching the previous
/// `jxl-oxide` `default-features = false` configuration).
pub fn build_context(
    image_header: Arc<ImageHeader>,
    embedded_icc: Option<Vec<u8>>,
) -> Result<RenderContext, DecodeError> {
    let mut builder = RenderContext::builder()
        .pool(JxlThreadPool::none())
        .force_wide_buffers(false);
    if let Some(icc) = embedded_icc {
        builder = builder.embedded_icc(icc);
    }
    builder
        .build(image_header)
        .map_err(|e| DecodeError::new(e.to_string()))
}

/// Loads all frames in the codestream into the render context.
///
/// In-memory port of `JxlImageInner::feed_bytes_inner` for the fully-buffered
/// case.
pub fn read_frames(ctx: &mut RenderContext, frame_bytes: &[u8]) -> Result<(), DecodeError> {
    let mut buf = frame_bytes;
    while !buf.is_empty() {
        let mut bitstream = Bitstream::new(buf);
        let frame = ctx
            .load_frame_header(&mut bitstream)
            .map_err(|e| DecodeError::new(e.to_string()))?;
        let read_bytes = bitstream.num_read_bits() / 8;
        buf = &buf[read_bytes..];

        let remaining = frame
            .feed_bytes(buf)
            .map_err(|e| DecodeError::new(e.to_string()))?;
        let is_last = frame.header().is_last;
        let done = frame.is_loading_done();
        buf = remaining;

        if !done {
            return Err(DecodeError::new("frame did not fully load"));
        }
        ctx.finalize_current_frame();
        if is_last {
            break;
        }
    }
    Ok(())
}

/// Extracts the tile/group structure of the given keyframe.
pub fn build_frame_plan(
    ctx: &RenderContext,
    keyframe_index: usize,
) -> Result<FramePlan, DecodeError> {
    let frame = ctx
        .keyframe(keyframe_index)
        .ok_or_else(|| DecodeError::new("keyframe not loaded"))?;
    let header = frame.header();

    let group_dim = header.group_dim();
    let groups_per_row = header.groups_per_row();
    let num_groups = header.num_groups();
    let num_passes = header.passes.num_passes;

    let mut tiles = Vec::with_capacity(num_groups as usize * num_passes as usize);
    for pass_idx in 0..num_passes {
        for group_idx in 0..num_groups {
            let group_x = group_idx % groups_per_row;
            let group_y = group_idx / groups_per_row;
            let (width, height) = header.group_size_for(group_idx);
            let region = Region {
                left: (group_x * group_dim) as i32,
                top: (group_y * group_dim) as i32,
                width,
                height,
            };
            tiles.push(TileJob {
                pass_idx,
                group_idx,
                lf_group_idx: header.lf_group_idx_from_group_idx(group_idx),
                region,
            });
        }
    }

    Ok(FramePlan {
        group_dim,
        groups_per_row,
        num_groups,
        num_passes,
        tiles,
    })
}
