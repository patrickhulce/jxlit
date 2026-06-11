use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct DecodeOptions {
    pub threads: Option<u32>,
    pub telemetry: Option<bool>,
    pub layout: Option<String>,
}

fn layout_from_napi(layout: Option<String>) -> napi::Result<jxlit::PixelLayout> {
    match layout.as_deref() {
        None | Some("interleaved") => Ok(jxlit::PixelLayout::Interleaved),
        Some("planar") => Ok(jxlit::PixelLayout::Planar),
        Some(other) => Err(Error::from_reason(format!(
            "invalid layout: {other} (expected \"interleaved\" or \"planar\")"
        ))),
    }
}

#[napi(object)]
pub struct Measure {
    pub name: String,
    pub start_ms: f64,
    pub duration_ms: f64,
}

#[napi(object)]
pub struct DecodeTelemetry {
    pub rust_timebase: f64,
    pub total_ms: f64,
    pub measures: Vec<Measure>,
}

#[napi(object)]
pub struct JxlitMeta {
    pub version: String,
    pub telemetry: Option<DecodeTelemetry>,
}

#[napi(object)]
pub struct DecodeMetadata {
    #[napi(js_name = "_jxlit")]
    pub jxlit: JxlitMeta,
}

#[napi(object)]
pub struct DecodedImage {
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub pixels: Float32Array,
    pub metadata: DecodeMetadata,
}

fn decode_options_from_napi(options: Option<DecodeOptions>) -> napi::Result<jxlit::DecodeOptions> {
    match options {
        Some(opts) => Ok(jxlit::DecodeOptions {
            threads: opts.threads.map(|n| n as usize),
            telemetry: opts.telemetry.unwrap_or(false),
            layout: layout_from_napi(opts.layout)?,
            ..jxlit::DecodeOptions::default()
        }),
        None => Ok(jxlit::DecodeOptions::default()),
    }
}

fn measure_from_rust(measure: &jxlit::Measure) -> Measure {
    Measure {
        name: measure.name.to_string(),
        start_ms: measure.start_ms,
        duration_ms: measure.duration_ms,
    }
}

fn telemetry_from_rust(telemetry: &jxlit::DecodeTelemetry) -> DecodeTelemetry {
    DecodeTelemetry {
        rust_timebase: telemetry.rust_timebase,
        total_ms: telemetry.total_ms,
        measures: telemetry.measures.iter().map(measure_from_rust).collect(),
    }
}

fn jxlit_meta_from_rust(meta: &jxlit::JxlitMeta) -> JxlitMeta {
    JxlitMeta {
        version: meta.version.to_string(),
        telemetry: meta.telemetry.as_ref().map(telemetry_from_rust),
    }
}

fn metadata_from_rust(metadata: &jxlit::DecodeMetadata) -> DecodeMetadata {
    DecodeMetadata {
        jxlit: jxlit_meta_from_rust(&metadata.jxlit),
    }
}

fn pixels_from_rust(pixels: jxlit::DecodedPixels) -> napi::Result<Vec<f32>> {
    pixels.cpu().ok_or_else(|| {
        Error::from_reason("GPU pixel buffers are not supported in Node bindings")
    })
}

#[napi]
pub fn decode(input: Buffer, options: Option<DecodeOptions>) -> Result<DecodedImage> {
    let decode_options = decode_options_from_napi(options)?;
    let decoded = jxlit::decode_with_options(input.as_ref(), &decode_options)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(DecodedImage {
        height: decoded.height as u32,
        width: decoded.width as u32,
        channels: decoded.channels as u32,
        pixels: Float32Array::new(pixels_from_rust(decoded.pixels)?),
        metadata: metadata_from_rust(&decoded.metadata),
    })
}
