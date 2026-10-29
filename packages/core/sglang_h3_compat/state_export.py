from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import torch

from .flags import (
    CONTINUATION_REPORT_KEY,
    EULER_SAMPLER,
    SAMPLER_EXTRA_KEY,
)
from .native import (
    build_state_manifest_json,
    frame_count_from_video_latent_t,
    validate_state_manifest_json,
)

MINIMAX_H3_STATE_PROTOCOL = "h3-native-av-state/v1"
MINIMAX_H3_STATE_FALLBACK_DIR = "/tmp/h3_state_exports"


def _safe_output_path(batch: Any) -> str | None:
    try:
        return batch.output_file_path(1, 0)
    except Exception:
        return None


def _state_base_path(batch: Any) -> str:
    output_path = _safe_output_path(batch)
    if not output_path:
        os.makedirs(MINIMAX_H3_STATE_FALLBACK_DIR, exist_ok=True)
        rid = getattr(batch, "rid", None) or uuid.uuid4().hex
        return os.path.join(MINIMAX_H3_STATE_FALLBACK_DIR, str(rid))
    base, _ext = os.path.splitext(str(output_path))
    return base


def export_state(batch: Any, _server_args: Any) -> dict[str, Any] | None:
    started = time.time()
    video = batch.latents.detach().clone()
    audio = batch.audio_latents.detach().clone()
    video_cpu = video.to("cpu", copy=True).contiguous()
    audio_cpu = audio.to("cpu", copy=True).contiguous()
    if video.is_cuda or audio.is_cuda:
        torch.cuda.synchronize()

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_SIGMAS_EXTRA_KEY,
    )

    sigmas = batch.extra.get(MINIMAX_H3_SIGMAS_EXTRA_KEY) or {}
    sampler = batch.extra.get(SAMPLER_EXTRA_KEY) or EULER_SAMPLER
    latent_t = int(video_cpu.shape[2])
    latent_h = int(video_cpu.shape[3])
    latent_w = int(video_cpu.shape[4])
    audio_t = int(audio_cpu.shape[2])
    try:
        frame_count = frame_count_from_video_latent_t(latent_t)
    except ValueError:
        frame_count = None
    sigmas_video = [float(value) for value in sigmas.get("video") or []]
    sigmas_audio = [float(value) for value in sigmas.get("audio") or []]
    plan = batch.extra.get("minimax_h3_resolved_plan")
    plan_seed = getattr(plan, "seed", None)
    if plan_seed is None:
        plan_seed = getattr(batch.sampling_params, "seed", None)
        if isinstance(plan_seed, list) and plan_seed:
            plan_seed = plan_seed[0]

    base = _state_base_path(batch)
    state_path = f"{base}.state.safetensors"
    manifest_path = f"{base}.state.manifest.json"
    from safetensors.torch import save_file

    save_file(
        {"video.latent": video_cpu, "audio.latent": audio_cpu},
        state_path,
        metadata={"protocol": MINIMAX_H3_STATE_PROTOCOL},
    )
    manifest_input = {
        "protocol": MINIMAX_H3_STATE_PROTOCOL,
        "artifact_kind": "h3-native-av-state",
        "created_at_unix": time.time(),
        "request": {
            "rid": getattr(batch, "rid", None),
            "seed": plan_seed,
            "sampler": str(sampler),
        },
        "tensor_spec": {
            "video.latent": {
                "shape": list(video_cpu.shape),
                "dtype": str(video_cpu.dtype),
                "space": "standardized (pre-reverse-normalization)",
            },
            "audio.latent": {
                "shape": list(audio_cpu.shape),
                "dtype": str(audio_cpu.dtype),
                "space": "standardized (pre-reverse-normalization)",
            },
        },
        "geometry": {
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "audio_t": audio_t,
            "canvas": f"{latent_h * 16}x{latent_w * 16}",
        },
        "sampling": {
            "sampler": str(sampler),
            "sigmas": sigmas_video,
            "sigmas_audio": sigmas_audio,
        },
        "video_timeline": {
            "fps": 24,
            "frame_count": frame_count,
            "video_latent_t": latent_t,
            "rule": "((fc-5)//17)*5+2",
        },
        "audio_timeline": {
            "latent_rate_hz": 40,
            "audio_latent_t": audio_t,
            "sample_rate": 32000,
            "channels": 2,
        },
        "continuation": batch.extra.get(CONTINUATION_REPORT_KEY),
        "validity": {
            "final_denoised_pre_decode": True,
            "terminal_sigma": sigmas_video[-1] if sigmas_video else None,
        },
        "delivery": {
            "mp4_output_path": _safe_output_path(batch),
            "state_file": state_path,
        },
        "export_elapsed_s": round(time.time() - started, 3),
    }
    manifest_json = build_state_manifest_json(
        json.dumps(manifest_input, ensure_ascii=False)
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_json + "\n")
    return json.loads(manifest_json)


def load_state_for_continuation(
    state_path: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors.torch import load_file

    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"h3_continuation state not found: {state_path}")
    tensors = load_file(state_path)
    if state_path.endswith(".state.safetensors"):
        manifest_path = state_path[: -len(".state.safetensors")] + (
            ".state.manifest.json"
        )
    else:
        manifest_path = os.path.splitext(state_path)[0] + ".manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    validate_state_manifest_json(json.dumps(manifest, ensure_ascii=False))
    for key in ("video.latent", "audio.latent"):
        if key not in tensors:
            raise ValueError(f"h3_continuation state missing {key!r}")
    video = tensors["video.latent"]
    if video.ndim != 5 or int(video.shape[1]) != 24:
        raise ValueError(
            "h3_continuation video.latent must be [1,24,T,H,W], got "
            f"{list(video.shape)}"
        )
    return tensors, manifest


__all__ = [
    "MINIMAX_H3_STATE_PROTOCOL",
    "export_state",
    "load_state_for_continuation",
]
