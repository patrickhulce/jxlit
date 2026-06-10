//! Container parse stage: demux the JPEG XL container, parse the image header /
//! ICC, and build the render context.
//!
//! Reproduces the glue `jxl-oxide` keeps private (`UninitializedJxlImage::
//! try_init`) on top of the narrow sub-crates, so the resulting `RenderContext`
//! is identical to what the umbrella crate would have built.

use std::sync::Arc;

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::{Frame, FrameContext};
use crate::vendor::jxl_render::RenderContext;

use crate::DecodeError;
use crate::pipeline::structs::container::{ContainerCtx, ContainerDeclaration};

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

/// Parses the image header and embedded ICC profile, returning a
/// [`ContainerDeclaration`] with the offset of the first real frame (skipping
/// the preview frame if present).
pub fn read_header(
    codestream: &[u8],
    pool: JxlThreadPool,
) -> Result<ContainerDeclaration, DecodeError> {
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
                pool: pool.clone(),
            },
        )
        .map_err(|e| DecodeError::new(e.to_string()))?;
        frame.toc().total_byte_size()
    } else {
        0
    };

    let offset = bitstream.num_read_bits() / 8 + skip_bytes;
    Ok(ContainerDeclaration {
        image_header,
        embedded_icc,
        offset,
    })
}

/// Builds a CMS-less render context and bundles it with the declaration into a
/// [`ContainerCtx`].
pub fn build_container_ctx(
    declaration: ContainerDeclaration,
    pool: JxlThreadPool,
) -> Result<ContainerCtx, DecodeError> {
    let mut builder = RenderContext::builder()
        .pool(pool)
        .force_wide_buffers(false);
    if let Some(icc) = declaration.embedded_icc.clone() {
        builder = builder.embedded_icc(icc);
    }
    let render_context = builder
        .build(declaration.image_header.clone())
        .map_err(|e| DecodeError::new(e.to_string()))?;

    Ok(ContainerCtx {
        declaration,
        render_context,
    })
}
