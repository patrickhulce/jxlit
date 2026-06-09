use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct DecodedImage {
    height: usize,
    width: usize,
    channels: usize,
    pixels: Vec<f32>,
}

#[wasm_bindgen]
impl DecodedImage {
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height as u32
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width as u32
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u32 {
        self.channels as u32
    }

    #[wasm_bindgen(getter)]
    pub fn pixels(&self) -> Vec<f32> {
        self.pixels.clone()
    }
}

#[wasm_bindgen]
pub fn decode(input: &[u8]) -> Result<DecodedImage, JsError> {
    let decoded = jxlit::decode(input).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(DecodedImage {
        height: decoded.height,
        width: decoded.width,
        channels: decoded.channels,
        pixels: decoded.pixels,
    })
}
