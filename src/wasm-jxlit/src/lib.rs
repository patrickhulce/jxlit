use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct DecodeOptions {
    threads: Option<usize>,
    telemetry: bool,
}

#[wasm_bindgen]
impl DecodeOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(threads: Option<usize>, telemetry: bool) -> Self {
        Self { threads, telemetry }
    }
}

#[wasm_bindgen]
pub struct Measure {
    name: String,
    start_ns: u64,
    duration_ns: u64,
}

#[wasm_bindgen]
impl Measure {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter, js_name = startNs)]
    pub fn start_ns(&self) -> f64 {
        self.start_ns as f64
    }

    #[wasm_bindgen(getter, js_name = durationNs)]
    pub fn duration_ns(&self) -> f64 {
        self.duration_ns as f64
    }
}

#[wasm_bindgen]
pub struct DecodeTelemetry {
    rust_timebase: u64,
    total_ns: u64,
    measures: Vec<Measure>,
}

#[wasm_bindgen]
impl DecodeTelemetry {
    #[wasm_bindgen(getter, js_name = rustTimebase)]
    pub fn rust_timebase(&self) -> f64 {
        self.rust_timebase as f64
    }

    #[wasm_bindgen(getter, js_name = totalNs)]
    pub fn total_ns(&self) -> f64 {
        self.total_ns as f64
    }

    #[wasm_bindgen(getter)]
    pub fn measures(&self) -> Vec<Measure> {
        self.measures
            .iter()
            .map(|measure| Measure {
                name: measure.name.clone(),
                start_ns: measure.start_ns,
                duration_ns: measure.duration_ns,
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
            total_ns: telemetry.total_ns,
            measures: telemetry
                .measures
                .iter()
                .map(|measure| Measure {
                    name: measure.name.clone(),
                    start_ns: measure.start_ns,
                    duration_ns: measure.duration_ns,
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
            telemetry: self.jxlit.telemetry.as_ref().map(|telemetry| DecodeTelemetry {
                rust_timebase: telemetry.rust_timebase,
                total_ns: telemetry.total_ns,
                measures: telemetry
                    .measures
                    .iter()
                    .map(|measure| Measure {
                        name: measure.name.clone(),
                        start_ns: measure.start_ns,
                        duration_ns: measure.duration_ns,
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
                telemetry: self
                    .metadata
                    .jxlit
                    .telemetry
                    .as_ref()
                    .map(|telemetry| DecodeTelemetry {
                        rust_timebase: telemetry.rust_timebase,
                        total_ns: telemetry.total_ns,
                        measures: telemetry
                            .measures
                            .iter()
                            .map(|measure| Measure {
                                name: measure.name.clone(),
                                start_ns: measure.start_ns,
                                duration_ns: measure.duration_ns,
                            })
                            .collect(),
                    }),
            },
        }
    }
}

fn measure_from_rust(measure: &jxlit::Measure) -> Measure {
    Measure {
        name: measure.name.to_string(),
        start_ns: measure.start_ns,
        duration_ns: measure.duration_ns,
    }
}

fn telemetry_from_rust(telemetry: &jxlit::DecodeTelemetry) -> DecodeTelemetry {
    DecodeTelemetry {
        rust_timebase: telemetry.rust_timebase,
        total_ns: telemetry.total_ns,
        measures: telemetry
            .measures
            .iter()
            .map(measure_from_rust)
            .collect(),
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
