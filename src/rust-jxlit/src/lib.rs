mod pipeline;
mod types;
mod vendor;

pub use types::{DecodeError, DecodeOptions, DecodedImage};

pub fn decode(input: &[u8]) -> Result<DecodedImage, DecodeError> {
    decode_with_options(input, &DecodeOptions::default())
}

pub fn decode_with_options(
    input: &[u8],
    options: &DecodeOptions,
) -> Result<DecodedImage, DecodeError> {
    pipeline::decode(input, options)
}

#[cfg(test)]
mod tests;
