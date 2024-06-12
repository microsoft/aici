use aici_ag2_ctrl::TokenParser;
use pyo3::{
    exceptions::{PyKeyError, PyValueError},
    prelude::*,
};

#[pyclass]
struct Parser {
    inner: TokenParser,
    #[pyo3(get, set)]
    log_level: isize,
}

pub(crate) fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Parser>()?;
    Ok(())
}
