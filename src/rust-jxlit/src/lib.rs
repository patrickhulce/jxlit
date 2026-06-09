use std::fmt;
use std::io::Cursor;

use jxl_oxide::JxlImage;

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedImage {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub pixels: Vec<f32>,
}

#[derive(Debug)]
pub struct DecodeError(String);

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DecodeError {}

pub fn decode(input: &[u8]) -> Result<DecodedImage, DecodeError> {
    let image = JxlImage::builder()
        .read(Cursor::new(input))
        .map_err(|e| DecodeError(e.to_string()))?;
    let render = image
        .render_frame(0)
        .map_err(|e| DecodeError(e.to_string()))?;
    let mut stream = render.stream();
    let height = stream.height() as usize;
    let width = stream.width() as usize;
    let channels = stream.channels() as usize;
    let mut pixels = vec![0.0f32; height * width * channels];
    let written = stream.write_to_buffer(&mut pixels);
    if written != pixels.len() {
        return Err(DecodeError(format!(
            "expected to write {} samples, wrote {written}",
            pixels.len()
        )));
    }

    Ok(DecodedImage {
        height,
        width,
        channels,
        pixels,
    })
}

#[cfg(test)]
mod tests;
