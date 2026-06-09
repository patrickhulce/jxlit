//! Entropy-decode stage of the VarDCT path.
//!
//! This is the "decode" half of the VarDCT pipeline (kept separate from the
//! dequant/inverse-DCT half in [`super::dequant`]/[`super::idct`]). It parses
//! the frame-global entropy structures (`LfGlobal`/`HfGlobal`), loads the LF
//! groups, and runs per-pass-group entropy decode, all via the vendored
//! `jxl-frame`/`jxl-render` building blocks. The output of pass-group decode is
//! the quantized HF coefficient grid that the dequant stage consumes.

use std::collections::HashMap;

use jxl_bitstream::Bitstream;
use jxl_modular::{Sample, image::TransformedModularSubimage};
use jxl_threadpool::JxlThreadPool;

use crate::vendor::jxl_frame::data::{HfGlobal, LfGlobal, LfGroup, PassGroupParams};
use crate::vendor::jxl_render::{Error, ImageWithRegion, IndexedFrame, Region, Result, util};

/// Parses the frame's `LfGlobal` (global entropy/LF metadata + modular header).
pub(crate) fn parse_lf_global<S: Sample>(frame: &IndexedFrame) -> Result<LfGlobal<S>> {
    Ok(frame
        .try_parse_lf_global()
        .ok_or(Error::IncompleteFrame)??)
}

/// Parses the frame's `HfGlobal` (HF passes / quantizer weights), if present.
pub(crate) fn parse_hf_global<S: Sample>(
    frame: &IndexedFrame,
    lf_global: &LfGlobal<S>,
) -> Result<Option<HfGlobal>> {
    Ok(frame.try_parse_hf_global(Some(lf_global)).transpose()?)
}

/// Loads (entropy-decodes) the LF groups into `lf_groups`, returning the LF
/// image when this is not an LF-frame reference.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_lf_groups<S: Sample>(
    frame: &IndexedFrame,
    lf_global: &LfGlobal<S>,
    lf_groups: &mut HashMap<u32, LfGroup<S>>,
    mlf_groups: Vec<TransformedModularSubimage<S>>,
    lf_region: Region,
    pool: &JxlThreadPool,
) -> Result<Option<ImageWithRegion>> {
    util::load_lf_groups(frame, lf_global, lf_groups, mlf_groups, lf_region, pool)
}

/// Entropy-decodes a single pass group, writing quantized HF coefficients into
/// `params.vardct.hf_coeff_output`. Thin forwarder over the vendored decoder.
pub(crate) fn decode_pass_group<S: Sample>(
    bitstream: &mut Bitstream,
    params: PassGroupParams<S>,
) -> crate::vendor::jxl_frame::Result<()> {
    crate::vendor::jxl_frame::data::decode_pass_group(bitstream, params)
}
