use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct DecodeOptions {
    threads: Option<usize>,
}

#[wasm_bindgen]
impl DecodeOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(threads: Option<usize>) -> Self {
        Self { threads }
    }
}

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
pub fn decode(input: &[u8], options: Option<DecodeOptions>) -> Result<DecodedImage, JsError> {
    let decode_options = match options {
        Some(opts) => jxlit::DecodeOptions {
            threads: opts.threads,
        },
        None => jxlit::DecodeOptions::default(),
    };
    let decoded = jxlit::decode_with_options(input, &decode_options)
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(DecodedImage {
        height: decoded.height,
        width: decoded.width,
        channels: decoded.channels,
        pixels: decoded.pixels,
    })
}
