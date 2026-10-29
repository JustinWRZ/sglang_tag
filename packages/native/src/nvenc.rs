use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::request::invalid;

#[pyfunction(signature = (existing=None, library_path=None))]
pub fn merge_ld_library_path(
    existing: Option<String>,
    library_path: Option<String>,
) -> PyResult<Option<String>> {
    let Some(library_path) = library_path.filter(|value| !value.trim().is_empty()) else {
        return Ok(existing);
    };
    Ok(Some(match existing {
        Some(existing) if !existing.trim().is_empty() => {
            format!("{library_path}:{existing}")
        }
        _ => library_path,
    }))
}

#[pyfunction(signature = (ffmpeg_path, library_path=None, timeout_ms=5000))]
pub fn probe_nvenc(
    ffmpeg_path: String,
    library_path: Option<String>,
    timeout_ms: u64,
) -> PyResult<bool> {
    if !Path::new(&ffmpeg_path).is_file() {
        return Err(invalid(format!("ffmpeg not found: {ffmpeg_path}")));
    }
    if timeout_ms == 0 {
        return Err(invalid("NVENC probe timeout must be positive"));
    }
    let mut command = Command::new(&ffmpeg_path);
    command
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .args(["-f", "lavfi"])
        .args(["-i", "testsrc2=size=256x256:rate=1"])
        .args(["-frames:v", "1"])
        .args(["-vcodec", "h264_nvenc"])
        .args(["-f", "null", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(library_path) = library_path
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        command.env("LD_LIBRARY_PATH", library_path);
    }

    let mut child = command
        .spawn()
        .map_err(|error| PyRuntimeError::new_err(format!("failed to start ffmpeg: {error}")))?;
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status.success()),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(false);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => {
                return Err(PyRuntimeError::new_err(format!(
                    "failed to wait for ffmpeg: {error}"
                )))
            }
        }
    }
}

#[pyfunction]
pub fn build_nvenc_command(
    command: Vec<String>,
    ffmpeg_path: String,
    cq: i32,
) -> PyResult<Vec<String>> {
    if !(0..=51).contains(&cq) {
        return Err(invalid("H3_COMPAT_NVENC_CQ must be in [0, 51]"));
    }
    let codec_index = command
        .windows(2)
        .position(|window| window[0] == "-vcodec" && window[1] == "libx264")
        .ok_or_else(|| invalid("FFmpeg command does not contain libx264"))?;

    let mut output = command;
    output[0] = ffmpeg_path;
    output[codec_index + 1] = "h264_nvenc".to_string();

    let remove_pair = |flag: &str, output: &mut Vec<String>| -> PyResult<()> {
        if let Some(index) = output.iter().position(|value| value == flag) {
            if index + 1 >= output.len() {
                return Err(invalid(format!(
                    "FFmpeg command has {flag} without a value"
                )));
            }
            output.drain(index..index + 2);
        }
        Ok(())
    };
    remove_pair("-preset", &mut output)?;
    remove_pair("-crf", &mut output)?;

    let insert_at = output
        .iter()
        .skip(codec_index + 1)
        .position(|value| value == "-pix_fmt")
        .map(|offset| offset + codec_index + 1)
        .ok_or_else(|| invalid("FFmpeg command does not contain output -pix_fmt"))?;
    output.insert(insert_at, "-rc".to_string());
    output.insert(insert_at + 1, "vbr".to_string());
    output.insert(insert_at + 2, "-cq".to_string());
    output.insert(insert_at + 3, cq.to_string());
    Ok(output)
}
