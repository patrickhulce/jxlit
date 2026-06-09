use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn decode<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let decoded = jxlit::decode(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let array = Array3::from_shape_vec(
        (decoded.height, decoded.width, decoded.channels),
        decoded.pixels,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pymodule]
fn _jxlit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    Ok(())
}
