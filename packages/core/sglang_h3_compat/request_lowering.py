from __future__ import annotations

import json
from typing import Any

from .compat_constants import (
    H3_COMPAT_CONTINUATION_FIELD,
    H3_COMPAT_EXPORT_FIELD,
    H3_COMPAT_SAMPLER_FIELD,
    H3_COMPAT_SIGMAS_FIELD,
)
from .native import (
    validate_continuation_request,
    validate_sampler,
    validate_sigmas,
)


def _request_extra(request: Any, name: str) -> Any:
    value = getattr(request, name, None)
    if value is not None:
        return value
    extras = getattr(request, "model_extra", None) or {}
    return extras.get(name)


def _parse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return value


def _sigmas(request: Any) -> dict[str, Any] | None:
    value = _parse_value(_request_extra(request, H3_COMPAT_SIGMAS_FIELD))
    if value is None:
        return None
    if not isinstance(value, dict) or "video" not in value:
        raise ValueError(
            "minimax_h3_sigmas must be an object with a 'video' sigma list"
        )
    if set(value) - {"video", "audio"}:
        raise ValueError(
            "minimax_h3_sigmas supports only 'video' and optional 'audio'"
        )
    video_raw = value.get("video")
    audio_raw = value.get("audio")
    if not isinstance(video_raw, list):
        raise ValueError(
            "minimax_h3_sigmas.video must contain at least two sigmas"
        )
    if audio_raw is not None and not isinstance(audio_raw, list):
        raise ValueError(
            "minimax_h3_sigmas.audio must contain only numbers"
        )
    try:
        video = [float(item) for item in video_raw]
        audio = (
            None
            if audio_raw is None
            else [float(item) for item in audio_raw]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "minimax_h3_sigmas must contain only numbers"
        ) from exc
    video, audio = validate_sigmas(video, audio)
    result: dict[str, Any] = {"video": video}
    if audio is not None:
        result["audio"] = audio
    return result


def _sampler(request: Any) -> str | None:
    value = _parse_value(_request_extra(request, H3_COMPAT_SAMPLER_FIELD))
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "minimax_h3_sampler must be 'euler' or 'er_sde'"
        )
    return validate_sampler(value)


def _export_state(request: Any) -> bool | None:
    value = _parse_value(_request_extra(request, H3_COMPAT_EXPORT_FIELD))
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("export_state must be a boolean")
    return value


def _continuation(request: Any) -> dict[str, Any] | None:
    value = _parse_value(_request_extra(request, H3_COMPAT_CONTINUATION_FIELD))
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            "h3_continuation must be an object with a state_path"
        )
    state_path = value.get("state_path")
    anchor_latent_frames = value.get("anchor_latent_frames")
    if not isinstance(state_path, str):
        raise ValueError(
            "h3_continuation must be an object with a state_path"
        )
    if isinstance(anchor_latent_frames, bool) or not isinstance(
        anchor_latent_frames, int
    ):
        raise ValueError("h3_continuation requires anchor_latent_frames")
    state_path, anchor_latent_frames = validate_continuation_request(
        state_path, anchor_latent_frames
    )
    result = dict(value)
    result["state_path"] = state_path
    result["anchor_latent_frames"] = anchor_latent_frames
    return result


def lower_video_request_kwargs_after(
    result: dict[str, Any],
    _self: Any,
    request: Any,
    _kwargs: dict[str, Any],
) -> dict[str, Any]:
    compat = {
        H3_COMPAT_SIGMAS_FIELD: _sigmas(request),
        H3_COMPAT_SAMPLER_FIELD: _sampler(request),
        H3_COMPAT_EXPORT_FIELD: _export_state(request),
        H3_COMPAT_CONTINUATION_FIELD: _continuation(request),
    }
    if all(value is None for value in compat.values()):
        return result
    diffusers_kwargs = dict(result.get("diffusers_kwargs") or {})
    for name, value in compat.items():
        if value is not None:
            diffusers_kwargs[name] = value
    result["diffusers_kwargs"] = diffusers_kwargs
    return result
