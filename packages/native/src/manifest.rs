use pyo3::prelude::*;
use serde_json::Value;

use crate::request::invalid;

#[pyfunction]
pub fn frame_count_from_video_latent_t(latent_t: i64) -> PyResult<i64> {
    if latent_t == 1 {
        return Ok(1);
    }
    if latent_t < 2 || (latent_t - 2) % 5 != 0 {
        return Err(invalid("MiniMax H3 video latent T must be 1 or match 5n+2"));
    }
    Ok(17 * ((latent_t - 2) / 5) + 5)
}

#[pyfunction]
pub fn build_state_manifest_json(input_json: String) -> PyResult<String> {
    let input: Value = serde_json::from_str(&input_json)
        .map_err(|error| invalid(format!("invalid state manifest input: {error}")))?;
    for section in [
        "protocol",
        "artifact_kind",
        "created_at_unix",
        "request",
        "tensor_spec",
        "geometry",
        "sampling",
        "video_timeline",
        "audio_timeline",
        "continuation",
        "validity",
        "delivery",
        "export_elapsed_s",
    ] {
        if input.get(section).is_none() {
            return Err(invalid(format!(
                "state manifest input is missing {section:?}"
            )));
        }
    }
    if input.get("protocol").and_then(Value::as_str) != Some("h3-native-av-state/v1") {
        return Err(invalid("unsupported MiniMax H3 state protocol"));
    }
    serde_json::to_string_pretty(&input)
        .map_err(|error| invalid(format!("failed to encode state manifest: {error}")))
}
