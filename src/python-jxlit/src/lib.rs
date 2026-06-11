use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[pyclass]
#[derive(Clone, Copy)]
struct DecodeOptions {
    #[pyo3(get, set)]
    threads: Option<usize>,
    #[pyo3(get, set)]
    telemetry: bool,
}

#[pymethods]
impl DecodeOptions {
    #[new]
    #[pyo3(signature = (threads=None, telemetry=false))]
    fn new(threads: Option<usize>, telemetry: bool) -> Self {
        Self { threads, telemetry }
    }
}

#[pyclass]
#[derive(Clone)]
struct Measure {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    start_ms: f64,
    #[pyo3(get)]
    duration_ms: f64,
}

#[pyclass]
#[derive(Clone)]
struct DecodeTelemetry {
    #[pyo3(get)]
    rust_timebase: f64,
    #[pyo3(get)]
    total_ms: f64,
    #[pyo3(get)]
    measures: Vec<Measure>,
}

#[pyclass]
#[derive(Clone)]
struct JxlitMeta {
    #[pyo3(get)]
    version: String,
    #[pyo3(get)]
    telemetry: Option<DecodeTelemetry>,
}

#[pyclass]
#[derive(Clone)]
struct DecodeMetadata {
    #[pyo3(get)]
    _jxlit: JxlitMeta,
}

#[pyclass]
struct DecodedImage {
    #[pyo3(get)]
    height: usize,
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    channels: usize,
    pixels: Array3<f32>,
    #[pyo3(get)]
    metadata: DecodeMetadata,
}

#[pymethods]
impl DecodedImage {
    #[getter]
    fn pixels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.pixels.clone().into_pyarray(py)
    }
}

fn decode_options_from_py(options: Option<DecodeOptions>) -> jxlit::DecodeOptions {
    match options {
        Some(opts) => jxlit::DecodeOptions {
            threads: opts.threads,
            telemetry: opts.telemetry,
        },
        None => jxlit::DecodeOptions::default(),
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
        _jxlit: jxlit_meta_from_rust(&metadata.jxlit),
    }
}

fn decoded_image_from_rust(decoded: jxlit::DecodedImage) -> PyResult<DecodedImage> {
    let array = Array3::from_shape_vec(
        (decoded.height, decoded.width, decoded.channels),
        decoded.pixels,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(DecodedImage {
        height: decoded.height,
        width: decoded.width,
        channels: decoded.channels,
        pixels: array,
        metadata: metadata_from_rust(&decoded.metadata),
    })
}

#[pyfunction]
#[pyo3(signature = (data, *, options=None))]
fn decode(data: &[u8], options: Option<DecodeOptions>) -> PyResult<DecodedImage> {
    let decode_options = decode_options_from_py(options);
    let decoded = jxlit::decode_with_options(data, &decode_options)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    decoded_image_from_rust(decoded)
}

#[pymodule]
fn _jxlit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecodeOptions>()?;
    m.add_class::<Measure>()?;
    m.add_class::<DecodeTelemetry>()?;
    m.add_class::<JxlitMeta>()?;
    m.add_class::<DecodeMetadata>()?;
    m.add_class::<DecodedImage>()?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
