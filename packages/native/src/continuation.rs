use pyo3::prelude::*;
use serde_json::{json, Value};

use crate::request::invalid;

const STATE_PROTOCOL: &str = "h3-native-av-state/v1";

fn required_i64(value: &Value, path: &str) -> PyResult<i64> {
    value
        .as_i64()
        .ok_or_else(|| invalid(format!("{path} must be an integer")))
}

fn required_f64_slice(value: &Value, path: &str) -> PyResult<Vec<f64>> {
    value
        .as_array()
        .map(|values| {
            values
                .iter()
                .filter_map(|item| item.as_f64())
                .collect::<Vec<_>>()
        })
        .filter(|values| values.len() == value.as_array().map(Vec::len).unwrap_or(0))
        .ok_or_else(|| invalid(format!("{path} must be a number array")))
}

fn optional_f64(value: Option<&Value>) -> PyResult<Option<f64>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| invalid("anchored_timestep must be a number")),
    }
}

#[pyfunction]
pub fn validate_state_manifest_json(manifest_json: String) -> PyResult<()> {
    let manifest: Value = serde_json::from_str(&manifest_json)
        .map_err(|error| invalid(format!("invalid state manifest JSON: {error}")))?;
    if manifest.get("protocol").and_then(Value::as_str) != Some(STATE_PROTOCOL) {
        return Err(invalid(format!(
            "h3_continuation state is not {STATE_PROTOCOL}"
        )));
    }
    let tensor_spec = manifest
        .get("tensor_spec")
        .ok_or_else(|| invalid("state manifest is missing tensor_spec"))?;
    for key in ["video.latent", "audio.latent"] {
        if tensor_spec.get(key).is_none() {
            return Err(invalid(format!(
                "h3_continuation state missing tensor {key:?}"
            )));
        }
    }
    Ok(())
}

#[pyfunction]
pub fn validate_continuation_json(input_json: String) -> PyResult<String> {
    let input: Value = serde_json::from_str(&input_json)
        .map_err(|error| invalid(format!("invalid continuation input: {error}")))?;
    let continuation = input
        .get("continuation")
        .ok_or_else(|| invalid("continuation input is missing continuation"))?;
    let manifest = input
        .get("manifest")
        .ok_or_else(|| invalid("continuation input is missing manifest"))?;
    let target = input
        .get("target")
        .ok_or_else(|| invalid("continuation input is missing target"))?;

    let state_path = continuation
        .get("state_path")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("h3_continuation state_path must be a string"))?;
    let requested = continuation
        .get("anchor_latent_frames")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid("h3_continuation requires anchor_latent_frames"))?;
    let src_t = required_i64(
        manifest
            .get("geometry")
            .and_then(|value| value.get("latent_t"))
            .ok_or_else(|| invalid("state manifest is missing geometry.latent_t"))?,
        "geometry.latent_t",
    )?;
    let src_h = required_i64(
        manifest
            .get("geometry")
            .and_then(|value| value.get("latent_h"))
            .ok_or_else(|| invalid("state manifest is missing geometry.latent_h"))?,
        "geometry.latent_h",
    )?;
    let src_w = required_i64(
        manifest
            .get("geometry")
            .and_then(|value| value.get("latent_w"))
            .ok_or_else(|| invalid("state manifest is missing geometry.latent_w"))?,
        "geometry.latent_w",
    )?;
    let target_t = required_i64(
        target
            .get("latent_t")
            .ok_or_else(|| invalid("target is missing latent_t"))?,
        "target.latent_t",
    )?;
    let target_h = required_i64(
        target
            .get("latent_h")
            .ok_or_else(|| invalid("target is missing latent_h"))?,
        "target.latent_h",
    )?;
    let target_w = required_i64(
        target
            .get("latent_w")
            .ok_or_else(|| invalid("target is missing latent_w"))?,
        "target.latent_w",
    )?;

    if src_h != target_h || src_w != target_w {
        return Err(invalid(format!(
            "h3_continuation latent canvas mismatch: source {src_h}x{src_w} vs target {target_h}x{target_w}"
        )));
    }

    let manifest_sampling = manifest
        .get("sampling")
        .ok_or_else(|| invalid("state manifest is missing sampling"))?;
    let manifest_sampler = manifest_sampling
        .get("sampler")
        .and_then(Value::as_str)
        .unwrap_or("euler");
    let current_sampler = input
        .get("current_sampler")
        .and_then(Value::as_str)
        .unwrap_or("euler");
    if manifest_sampler != current_sampler {
        return Err(invalid(format!(
            "h3_continuation sampler mismatch: {manifest_sampler:?} != {current_sampler:?}"
        )));
    }

    let source_sigmas = required_f64_slice(
        manifest_sampling
            .get("sigmas")
            .ok_or_else(|| invalid("state manifest is missing sampling.sigmas"))?,
        "sampling.sigmas",
    )?;
    let current_sigmas = required_f64_slice(
        input
            .get("current_sigmas")
            .ok_or_else(|| invalid("continuation input is missing current_sigmas"))?,
        "current_sigmas",
    )?;
    if source_sigmas.len() != current_sigmas.len()
        || source_sigmas
            .iter()
            .zip(current_sigmas.iter())
            .any(|(left, right)| (left - right).abs() > 1e-6)
    {
        return Err(invalid("h3_continuation sigma schedule mismatch"));
    }

    let resolved = requested.min(src_t);
    if resolved < 1 || resolved >= target_t {
        return Err(invalid(
            "h3_continuation resolved to an invalid anchor length",
        ));
    }
    let anchor_row_count = required_i64(
        input
            .get("anchor_row_count")
            .ok_or_else(|| invalid("continuation input is missing anchor_row_count"))?,
        "anchor_row_count",
    )?;
    let anchored_timestep = optional_f64(continuation.get("anchored_timestep"))?;

    let report = json!({
        "source_state_path": state_path,
        "source_protocol": manifest.get("protocol").cloned().unwrap_or(Value::Null),
        "source_latent_t": src_t,
        "source_latent_h": src_h,
        "source_latent_w": src_w,
        "requested_anchor_latent_frames": requested,
        "resolved_anchor_latent_frames": resolved,
        "anchor_row_count": anchor_row_count,
        "anchored_timestep": anchored_timestep.unwrap_or(0.999),
        "mode": "masked_prefix"
    });
    serde_json::to_string(&report)
        .map_err(|error| invalid(format!("failed to encode continuation report: {error}")))
}
