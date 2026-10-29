use pyo3::prelude::*;

use crate::request::{invalid, validate_video_schedule};

pub(crate) fn time_snr_shift_impl(shift: f64, t: f64) -> f64 {
    if shift == 1.0 {
        t
    } else {
        shift * t / (1.0 + (shift - 1.0) * t)
    }
}

pub(crate) fn retarget_sigma_shift_impl(
    sigma: f64,
    from_shift: f64,
    to_shift: f64,
) -> PyResult<f64> {
    if !(from_shift.is_finite() && from_shift > 0.0) {
        return Err(invalid("flow shift must be positive"));
    }
    if !(to_shift.is_finite() && to_shift > 0.0) {
        return Err(invalid("audio flow shift must be positive"));
    }
    let uniform = sigma / (from_shift + sigma * (1.0 - from_shift));
    Ok(time_snr_shift_impl(to_shift, uniform))
}

fn default_shifted_sigmas(num_steps: usize, shift: f64) -> Vec<f64> {
    if num_steps == 0 {
        return Vec::new();
    }
    let points = num_steps + 1;
    let mut shifted = Vec::with_capacity(points + 1);
    for index in 0..points {
        let base = 1.0 - index as f64 / (points - 1) as f64;
        shifted.push(time_snr_shift_impl(shift, base));
    }
    let mut deduplicated = Vec::with_capacity(shifted.len());
    for value in shifted {
        if deduplicated.last().copied().as_ref() != Some(&value) {
            deduplicated.push(value);
        }
    }
    if deduplicated.last().copied().unwrap_or(0.0) > 0.0 {
        deduplicated.push(0.0);
    }
    deduplicated
}

#[pyfunction]
pub fn time_snr_shift(shift: f64, t: f64) -> f64 {
    time_snr_shift_impl(shift, t)
}

#[pyfunction]
pub fn retarget_sigma_shift(sigma: f64, from_shift: f64, to_shift: f64) -> PyResult<f64> {
    retarget_sigma_shift_impl(sigma, from_shift, to_shift)
}

#[pyfunction(signature = (video=None, audio=None, *, sampler, num_steps, flow_shift, audio_flow_shift))]
pub fn prepare_sigmas(
    video: Option<Vec<f64>>,
    audio: Option<Vec<f64>>,
    sampler: String,
    num_steps: usize,
    flow_shift: f64,
    audio_flow_shift: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if sampler != "euler" && sampler != "er_sde" {
        return Err(invalid("minimax_h3_sampler must be euler or er_sde"));
    }
    if !(flow_shift.is_finite() && flow_shift > 0.0) {
        return Err(invalid("flow shift must be positive"));
    }
    if !(audio_flow_shift.is_finite() && audio_flow_shift > 0.0) {
        return Err(invalid("audio flow shift must be positive"));
    }

    let mut video_schedule = match video {
        Some(values) => values,
        None => default_shifted_sigmas(num_steps, flow_shift),
    };
    validate_video_schedule(&video_schedule)?;
    if sampler == "er_sde" && video_schedule[0] >= 1.0 {
        video_schedule[0] = time_snr_shift_impl(12.0, 1.0 - 1e-4);
    }

    let audio_schedule = match audio {
        Some(values) => {
            if values.len() != video_schedule.len() {
                return Err(invalid(
                    "minimax_h3_sigmas.audio must match the video schedule length",
                ));
            }
            values
        }
        None => video_schedule
            .iter()
            .map(|sigma| retarget_sigma_shift_impl(*sigma, flow_shift, audio_flow_shift))
            .collect::<PyResult<Vec<_>>>()?,
    };
    Ok((video_schedule, audio_schedule))
}
