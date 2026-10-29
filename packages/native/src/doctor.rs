use pyo3::prelude::*;

#[pyfunction]
pub fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
pub fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[pyfunction]
pub fn rustc_version() -> &'static str {
    option_env!("SGLANG_H3_RUSTC_VERSION").unwrap_or("unknown")
}
