use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct DecodeOptions {
    pub threads: Option<u32>,
}

#[napi(object)]
pub struct DecodedImage {
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub pixels: Float32Array,
}

fn decode_options_from_napi(options: Option<DecodeOptions>) -> jxlit::DecodeOptions {
    match options {
        Some(opts) => jxlit::DecodeOptions {
            threads: opts.threads.map(|n| n as usize),
        },
        None => jxlit::DecodeOptions::default(),
    }
}

#[napi]
pub fn decode(input: Buffer, options: Option<DecodeOptions>) -> Result<DecodedImage> {
    let decode_options = decode_options_from_napi(options);
    let decoded = jxlit::decode_with_options(input.as_ref(), &decode_options)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(DecodedImage {
        height: decoded.height as u32,
        width: decoded.width as u32,
        channels: decoded.channels as u32,
        pixels: Float32Array::new(decoded.pixels),
    })
}
