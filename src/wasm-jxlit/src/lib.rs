use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct DecodeOptions {
    threads: Option<usize>,
    telemetry: bool,
    layout: jxlit::PixelLayout,
}

fn layout_from_wasm(layout: Option<String>) -> Result<jxlit::PixelLayout, JsError> {
    match layout.as_deref() {
        None | Some("interleaved") => Ok(jxlit::PixelLayout::Interleaved),
        Some("planar") => Ok(jxlit::PixelLayout::Planar),
        Some(other) => Err(JsError::new(&format!(
            "invalid layout: {other} (expected \"interleaved\" or \"planar\")"
        ))),
    }
}

#[wasm_bindgen]
impl DecodeOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(
        threads: Option<usize>,
        telemetry: bool,
        layout: Option<String>,
    ) -> Result<DecodeOptions, JsError> {
        Ok(Self {
            threads,
            telemetry,
            layout: layout_from_wasm(layout)?,
        })
    }
}

#[wasm_bindgen]
pub struct Measure {
    name: String,
    start_ms: f64,
    duration_ms: f64,
}

#[wasm_bindgen]
impl Measure {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter, js_name = startMs)]
    pub fn start_ms(&self) -> f64 {
        self.start_ms
    }

    #[wasm_bindgen(getter, js_name = durationMs)]
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms
    }
}

#[wasm_bindgen]
pub struct DecodeTelemetry {
    rust_timebase: f64,
    total_ms: f64,
    measures: Vec<Measure>,
}

#[wasm_bindgen]
impl DecodeTelemetry {
    #[wasm_bindgen(getter, js_name = rustTimebase)]
    pub fn rust_timebase(&self) -> f64 {
        self.rust_timebase
    }

    #[wasm_bindgen(getter, js_name = totalMs)]
    pub fn total_ms(&self) -> f64 {
        self.total_ms
    }

    #[wasm_bindgen(getter)]
    pub fn measures(&self) -> Vec<Measure> {
        self.measures
            .iter()
            .map(|measure| Measure {
                name: measure.name.clone(),
                start_ms: measure.start_ms,
                duration_ms: measure.duration_ms,
            })
            .collect()
    }
}

#[wasm_bindgen]
pub struct JxlitMeta {
    version: String,
    telemetry: Option<DecodeTelemetry>,
}

#[wasm_bindgen]
impl JxlitMeta {
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        self.version.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn telemetry(&self) -> Option<DecodeTelemetry> {
        self.telemetry.as_ref().map(|telemetry| DecodeTelemetry {
            rust_timebase: telemetry.rust_timebase,
            total_ms: telemetry.total_ms,
            measures: telemetry
                .measures
                .iter()
                .map(|measure| Measure {
                    name: measure.name.clone(),
                    start_ms: measure.start_ms,
                    duration_ms: measure.duration_ms,
                })
                .collect(),
        })
    }
}

#[wasm_bindgen]
pub struct DecodeMetadata {
    jxlit: JxlitMeta,
}

#[wasm_bindgen]
impl DecodeMetadata {
    #[wasm_bindgen(getter, js_name = _jxlit)]
    pub fn jxlit(&self) -> JxlitMeta {
        JxlitMeta {
            version: self.jxlit.version.clone(),
            telemetry: self
                .jxlit
                .telemetry
                .as_ref()
                .map(|telemetry| DecodeTelemetry {
                    rust_timebase: telemetry.rust_timebase,
                    total_ms: telemetry.total_ms,
                    measures: telemetry
                        .measures
                        .iter()
                        .map(|measure| Measure {
                            name: measure.name.clone(),
                            start_ms: measure.start_ms,
                            duration_ms: measure.duration_ms,
                        })
                        .collect(),
                }),
        }
    }
}

#[wasm_bindgen]
pub struct DecodedImage {
    height: usize,
    width: usize,
    channels: usize,
    pixels: Vec<f32>,
    metadata: DecodeMetadata,
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

    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> DecodeMetadata {
        DecodeMetadata {
            jxlit: JxlitMeta {
                version: self.metadata.jxlit.version.clone(),
                telemetry: self.metadata.jxlit.telemetry.as_ref().map(|telemetry| {
                    DecodeTelemetry {
                        rust_timebase: telemetry.rust_timebase,
                        total_ms: telemetry.total_ms,
                        measures: telemetry
                            .measures
                            .iter()
                            .map(|measure| Measure {
                                name: measure.name.clone(),
                                start_ms: measure.start_ms,
                                duration_ms: measure.duration_ms,
                            })
                            .collect(),
                    }
                }),
            },
        }
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

#[wasm_bindgen]
pub fn decode(input: &[u8], options: Option<DecodeOptions>) -> Result<DecodedImage, JsError> {
    let decode_options = match options {
        Some(opts) => jxlit::DecodeOptions {
            threads: opts.threads,
            telemetry: opts.telemetry,
            layout: opts.layout,
            ..jxlit::DecodeOptions::default()
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
        metadata: metadata_from_rust(&decoded.metadata),
    })
}
