//! Frame parse stage: load frames and read the frame-global entropy structures.
//!
//! `read_frames` is an in-memory port of `JxlImageInner::feed_bytes_inner` for
//! the fully-buffered case. The `read_*_global` / `read_low_frequency_groups`
//! helpers parse the frame-global (`LfGlobal`/`HfGlobal`) and LF-group entropy
//! structures via the vendored building blocks; these are frame-global, not
//! image-global, so they live here rather than in `container`.

use std::collections::HashMap;

use jxl_bitstream::Bitstream;
use jxl_modular::{Sample, image::TransformedModularSubimage};
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use crate::vendor::jxl_render::{
    Error, ImageWithRegion, IndexedFrame, Region, RenderContext, Result, util,
};

use crate::DecodeError;

/// Loads all frames in the codestream into the render context.
pub fn read_frames(
    ctx: &mut RenderContext,
    frame_bytes: &[u8],
) -> std::result::Result<(), DecodeError> {
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

/// Parses the frame's `LfGlobal` (global entropy/LF metadata + modular header).
pub fn read_low_frequency_global<S: Sample>(frame: &IndexedFrame) -> Result<LfGlobal<S>> {
    Ok(frame
        .try_parse_lf_global()
        .ok_or(Error::IncompleteFrame)??)
}

/// Parses the frame's `HfGlobal` (HF passes / quantizer weights), if present.
pub fn read_high_frequency_global<S: Sample>(
    frame: &IndexedFrame,
    low_frequency_global: &LfGlobal<S>,
) -> Result<Option<HfGlobal>> {
    Ok(frame
        .try_parse_hf_global(Some(low_frequency_global))
        .transpose()?)
}

/// Reads (entropy-decodes) the LF groups into `low_frequency_groups`, returning
/// the LF image when this is not an LF-frame reference.
pub fn read_low_frequency_groups<S: Sample>(
    frame: &IndexedFrame,
    low_frequency_global: &LfGlobal<S>,
    low_frequency_groups: &mut HashMap<u32, LfGroup<S>>,
    modular_lf_groups: Vec<TransformedModularSubimage<S>>,
    lf_region: Region,
    pool: &JxlThreadPool,
) -> Result<Option<ImageWithRegion>> {
    util::load_lf_groups(
        frame,
        low_frequency_global,
        low_frequency_groups,
        modular_lf_groups,
        lf_region,
        pool,
    )
}
