// Vendored from jxl-oxide (jxl-frame 0.13.3), (c) Wonwoo Choi, licensed MIT OR Apache-2.0.
// Source: https://github.com/tirr-c/jxl-oxide/blob/f8ae722ef2d6b782941c89517d19cfbf605c4a9d/crates/jxl-frame/src/data/noise.rs
// Copied as-is; only crate-path references changed.

#[derive(Debug)]
pub struct NoiseParameters {
    pub lut: [f32; 8],
}

impl<Ctx> jxl_oxide_common::Bundle<Ctx> for NoiseParameters {
    type Error = crate::vendor::jxl_frame::Error;

    fn parse(
        bitstream: &mut jxl_bitstream::Bitstream,
        _: Ctx,
    ) -> crate::vendor::jxl_frame::Result<Self> {
        let mut lut = [0.0f32; 8];
        for slot in &mut lut {
            *slot = bitstream.read_bits(10)? as f32 / (1 << 10) as f32;
        }

        Ok(Self { lut })
    }
}
