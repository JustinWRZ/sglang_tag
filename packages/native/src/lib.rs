mod continuation;
mod doctor;
mod er_sde;
mod manifest;
mod nvenc;
mod request;
mod sigma;

pub use continuation::{validate_continuation_json, validate_state_manifest_json};
pub use er_sde::{er_sde_step_scalars, ErSdeStepScalars};
pub use manifest::{build_state_manifest_json, frame_count_from_video_latent_t};
pub use nvenc::{build_nvenc_command, merge_ld_library_path, probe_nvenc};
pub use sigma::{prepare_sigmas, retarget_sigma_shift, time_snr_shift};

use pyo3::prelude::*;

#[pymodule]
fn sglang_h3_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<er_sde::ErSdeStepScalars>()?;
    module.add_function(wrap_pyfunction!(doctor::native_version, module)?)?;
    module.add_function(wrap_pyfunction!(doctor::build_profile, module)?)?;
    module.add_function(wrap_pyfunction!(doctor::rustc_version, module)?)?;
    module.add_function(wrap_pyfunction!(request::validate_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(request::validate_sigmas, module)?)?;
    module.add_function(wrap_pyfunction!(request::validate_export_state, module)?)?;
    module.add_function(wrap_pyfunction!(
        request::validate_continuation_request,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sigma::time_snr_shift, module)?)?;
    module.add_function(wrap_pyfunction!(sigma::retarget_sigma_shift, module)?)?;
    module.add_function(wrap_pyfunction!(sigma::prepare_sigmas, module)?)?;
    module.add_function(wrap_pyfunction!(er_sde::er_sde_step_scalars, module)?)?;
    module.add_function(wrap_pyfunction!(
        continuation::validate_state_manifest_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        continuation::validate_continuation_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        manifest::frame_count_from_video_latent_t,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        manifest::build_state_manifest_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(nvenc::merge_ld_library_path, module)?)?;
    module.add_function(wrap_pyfunction!(nvenc::probe_nvenc, module)?)?;
    module.add_function(wrap_pyfunction!(nvenc::build_nvenc_command, module)?)?;
    Ok(())
}
