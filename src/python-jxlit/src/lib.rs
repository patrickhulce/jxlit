use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy)]
struct DecodeOptions {
    #[pyo3(get, set)]
    threads: Option<usize>,
}

#[pymethods]
impl DecodeOptions {
    #[new]
    #[pyo3(signature = (threads=None))]
    fn new(threads: Option<usize>) -> Self {
        Self { threads }
    }
}

fn decode_options_from_py(options: Option<DecodeOptions>) -> jxlit::DecodeOptions {
    match options {
        Some(opts) => jxlit::DecodeOptions {
            threads: opts.threads,
        },
        None => jxlit::DecodeOptions::default(),
    }
}

#[pyfunction]
#[pyo3(signature = (data, *, options=None))]
fn decode<'py>(
    py: Python<'py>,
    data: &[u8],
    options: Option<DecodeOptions>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let decode_options = decode_options_from_py(options);
    let decoded = jxlit::decode_with_options(data, &decode_options)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let array = Array3::from_shape_vec(
        (decoded.height, decoded.width, decoded.channels),
        decoded.pixels,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pymodule]
fn _jxlit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecodeOptions>()?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    Ok(())
}
