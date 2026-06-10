//! Per-tile entropy (ANS) decode of a single pass-group.
//!
//! Reads the quantized HF coefficients for one tile into
//! `params.vardct.hf_coeff_output`. Thin forwarder over the vendored decoder;
//! this is the start of the future GPU offload region.

use jxl_bitstream::Bitstream;
use jxl_modular::Sample;

use crate::vendor::jxl_frame::data::PassGroupParams;

/// Entropy-decodes a single pass-group (one JXL group within one pass).
pub fn read_pass_group<S: Sample>(
    bitstream: &mut Bitstream,
    params: PassGroupParams<S>,
) -> crate::vendor::jxl_frame::Result<()> {
    crate::vendor::jxl_frame::data::decode_pass_group(bitstream, params)
}
