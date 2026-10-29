use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn invalid(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

#[pyfunction]
pub fn validate_sampler(value: String) -> PyResult<String> {
    if value == "euler" || value == "er_sde" {
        Ok(value)
    } else {
        Err(invalid("minimax_h3_sampler must be 'euler' or 'er_sde'"))
    }
}

#[pyfunction(signature = (video, audio=None))]
pub fn validate_sigmas(
    video: Vec<f64>,
    audio: Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
    validate_video_schedule(&video)?;
    if let Some(audio_schedule) = audio.as_deref() {
        if audio_schedule.len() != video.len() {
            return Err(invalid(
                "minimax_h3_sigmas.audio must match the video schedule length",
            ));
        }
    }
    Ok((video, audio))
}

#[pyfunction]
pub fn validate_export_state(value: bool) -> bool {
    value
}

#[pyfunction]
pub fn validate_continuation_request(
    state_path: String,
    anchor_latent_frames: i64,
) -> PyResult<(String, i64)> {
    if state_path.trim().is_empty() {
        return Err(invalid(
            "h3_continuation must be an object with a state_path",
        ));
    }
    if anchor_latent_frames < 1 {
        return Err(invalid("h3_continuation requires anchor_latent_frames"));
    }
    Ok((state_path, anchor_latent_frames))
}

pub(crate) fn validate_video_schedule(video: &[f64]) -> PyResult<()> {
    if video.len() < 2 {
        return Err(invalid(
            "minimax_h3_sigmas.video must contain at least two sigmas",
        ));
    }
    if video.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(invalid(
            "minimax_h3_sigmas.video must be finite and non-negative",
        ));
    }
    if video.windows(2).any(|pair| pair[1] > pair[0]) {
        return Err(invalid("minimax_h3_sigmas.video must be non-increasing"));
    }
    if video[0] <= 0.0 || video[video.len() - 1] != 0.0 {
        return Err(invalid(
            "minimax_h3_sigmas.video must start above zero and end at zero",
        ));
    }
    Ok(())
}
