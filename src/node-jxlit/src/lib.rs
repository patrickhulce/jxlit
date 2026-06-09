use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct DecodedImage {
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub pixels: Float32Array,
}

#[napi]
pub fn decode(input: Buffer) -> Result<DecodedImage> {
    let decoded = jxlit::decode(input.as_ref()).map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(DecodedImage {
        height: decoded.height as u32,
        width: decoded.width as u32,
        channels: decoded.channels as u32,
        pixels: Float32Array::new(decoded.pixels),
    })
}
