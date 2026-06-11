//! Per-tile entropy (ANS) decode of a single pass-group.
//!
//! Reads the quantized HF coefficients for one tile into
//! `params.vardct.hf_coeff_output`. Thin forwarder over the vendored decoder;
//! this is the start of the future GPU offload region.

use jxl_bitstream::Bitstream;
use jxl_modular::Sample;

use crate::pipeline::gpu::{Device, GpuEnvironment, availability, kernels};
use crate::types::DecodeOptions;
use crate::vendor::jxl_frame::data::PassGroupParams;

/// Entropy-decodes a single pass-group (one JXL group within one pass).
pub fn read_pass_group<S: Sample>(
    device: Device,
    bitstream: &mut Bitstream,
    params: PassGroupParams<S>,
    group_idx: u32,
    pass_idx: u32,
    options: &DecodeOptions,
    env: GpuEnvironment,
) -> crate::vendor::jxl_frame::Result<()> {
    if device.is_gpu()
        && availability::read_pass_group_available(
            params.frame_header,
            group_idx,
            pass_idx,
            options,
            env,
        )
    {
        let _ = (bitstream, params);
        kernels::read_pass_group_on_gpu(group_idx, pass_idx);
        Ok(())
    } else {
        crate::vendor::jxl_frame::data::decode_pass_group(bitstream, params)
    }
}
