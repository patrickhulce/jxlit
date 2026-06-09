use pyo3::prelude::*;

#[pyfunction]
fn decode(data: &[u8]) -> Vec<u8> {
    jxlit::decode(data)
}

#[pymodule]
fn _jxlit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    Ok(())
}
