mod pipeline;
mod telemetry;
mod types;
mod vendor;

pub use telemetry::{RebasedMeasure, RebasingTelemetry, rebase_telemetry};
pub use types::{
    DecodeError, DecodeMetadata, DecodeOptions, DecodeTelemetry, DecodedImage, JxlitMeta, Measure,
    PixelLayout,
};

pub fn decode(input: &[u8]) -> Result<DecodedImage, DecodeError> {
    decode_with_options(input, &DecodeOptions::default())
}

pub fn decode_with_options(
    input: &[u8],
    options: &DecodeOptions,
) -> Result<DecodedImage, DecodeError> {
    let version = env!("CARGO_PKG_VERSION");

    if options.telemetry {
        let (image, telemetry) = telemetry::with_timing_subscriber(|| {
            let _decode = phase_guard!("decode");
            pipeline::decode(input, options)
        });
        Ok(image?.attach_metadata(DecodeMetadata::with_telemetry(version, Some(telemetry))))
    } else {
        let image = pipeline::decode(input, options)?;
        Ok(image.attach_metadata(DecodeMetadata::with_version(version)))
    }
}

#[cfg(test)]
mod tests;
