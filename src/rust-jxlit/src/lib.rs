mod pipeline;
mod types;
mod vendor;

pub use types::{DecodeError, DecodedImage};

pub fn decode(input: &[u8]) -> Result<DecodedImage, DecodeError> {
    pipeline::decode(input)
}

#[cfg(test)]
mod tests;
