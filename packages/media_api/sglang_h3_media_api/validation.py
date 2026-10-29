"""Strict media v1 request validation producing the B.6 422 shape."""

from __future__ import annotations

import re
from typing import Any

from . import capabilities as caps

_JOB_ID_RE = re.compile(caps.JOB_ID_PATTERN)
_URL_RE = re.compile(r"^https?://")


def _err(errors: list[dict], loc: list[str], msg: str, type_: str) -> None:
    errors.append({"loc": loc, "msg": msg, "type": type_})


def validate_request(body: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    """Validate a media.generate_request body.

    Returns (normalized, errors). ``normalized`` is a plain dict with the
    accepted fields only; ``errors`` follows B.6 (loc/msg/type) and is
    non-empty exactly when the request must be rejected with 422.
    """
    errors: list[dict] = []
    normalized: dict[str, Any] = {}

    for key in body:
        if key not in caps.ROOT_ALLOWED:
            _err(
                errors,
                [key],
                f"unknown field {key!r} (additionalProperties: false)",
                "extra_forbidden",
            )

    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        _err(errors, ["prompt"], "prompt is required and must be a non-empty string", "missing")
    elif len(prompt) > caps.PROMPT_MAX_LENGTH:
        _err(
            errors,
            ["prompt"],
            f"prompt must be at most {caps.PROMPT_MAX_LENGTH} characters",
            "string_too_long",
        )
    else:
        normalized["prompt"] = prompt

    negative_prompt = body.get("negative_prompt")
    if negative_prompt is not None:
        if not isinstance(negative_prompt, str):
            _err(errors, ["negative_prompt"], "negative_prompt must be a string", "string_type")
        else:
            normalized["negative_prompt"] = negative_prompt

    model = body.get("model")
    if model is not None:
        if not isinstance(model, str) or not model:
            _err(errors, ["model"], "model must be a non-empty string", "string_type")
        else:
            allowed = caps.service_models()
            if model not in allowed:
                _err(
                    errors,
                    ["model"],
                    caps.MODEL_UNSUPPORTED_MESSAGE.format(models=", ".join(allowed)),
                    "value_error",
                )
            else:
                normalized["model"] = model

    seed = body.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            _err(errors, ["seed"], "seed must be an integer", "int_type")
        elif not (caps.SEED_MIN <= seed <= caps.SEED_MAX):
            _err(
                errors,
                ["seed"],
                f"seed must be between {caps.SEED_MIN} and {caps.SEED_MAX}",
                "int_parsing",
            )
        else:
            normalized["seed"] = seed

    has_batch = "batch" in body and body["batch"] is not None
    has_n = "n" in body and body["n"] is not None
    if has_batch and has_n:
        _err(
            errors,
            ["batch"],
            "'batch' and its deprecated alias 'n' are mutually exclusive",
            "value_error",
        )
    count = body.get("batch") if has_batch else body.get("n") if has_n else 1
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int):
            _err(errors, ["batch" if has_batch else "n"], "batch must be an integer", "int_type")
        else:
            if not (1 <= count <= caps.BATCH_MAX):
                _err(
                    errors,
                    ["batch" if has_batch else "n"],
                    f"batch must be between 1 and {caps.BATCH_MAX}",
                    "int_parsing",
                )
            elif count > 1:
                _err(
                    errors,
                    ["batch" if has_batch else "n"],
                    caps.BATCH_UNSUPPORTED_MESSAGE,
                    "value_error",
                )
            else:
                normalized["batch"] = 1

    background = body.get("background")
    if background is not None:
        if not isinstance(background, bool):
            _err(errors, ["background"], "background must be a boolean", "bool_type")
        else:
            normalized["background"] = background

    job_id = body.get("job_id")
    if job_id is not None:
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            _err(
                errors,
                ["job_id"],
                "job_id must be a 32-character hex string (lower or upper case)",
                "string_pattern_mismatch",
            )
        else:
            normalized["job_id"] = job_id.lower()

    normalized["assets"] = _validate_assets(body.get("assets"), errors)
    normalized["output"] = _validate_output(body.get("output"), errors)

    if not errors and not normalized["assets"]:
        _err(errors, ["assets"], caps.ASSET_REQUIRED_MESSAGE, "missing")
    return normalized, errors


def _validate_assets(assets: Any, errors: list[dict]) -> list[dict[str, Any]]:
    if assets is None:
        return []
    if not isinstance(assets, list):
        _err(errors, ["assets"], "assets must be an array", "iterable_type")
        return []
    accepted: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            _err(errors, ["assets", str(index)], "asset must be an object", "model_type")
            continue
        for key in asset:
            if key not in caps.ASSET_ALLOWED:
                _err(
                    errors,
                    ["assets", str(index), key],
                    f"unknown asset field {key!r} (additionalProperties: false)",
                    "extra_forbidden",
                )
        asset_type = asset.get("type")
        if asset_type not in caps.ASSET_TYPES:
            message = caps.ASSET_TYPE_UNSUPPORTED.get(
                str(asset_type), "asset.type must be one of image|video|audio"
            )
            _err(errors, ["assets", str(index), "type"], message, "enum")
            continue
        url = asset.get("url")
        if not isinstance(url, str) or not _URL_RE.match(url):
            _err(
                errors,
                ["assets", str(index), "url"],
                "asset.url must match ^https?://",
                "string_pattern_mismatch",
            )
            continue
        role = asset.get("role", "reference")
        if role not in caps.ASSET_ROLES:
            message = caps.ASSET_ROLE_UNSUPPORTED.get(
                str(role), "asset.role must be reference or mask"
            )
            _err(errors, ["assets", str(index), "role"], message, "enum")
            continue
        accepted.append({"type": asset_type, "url": url, "role": role})
    return accepted


def _validate_output(output: Any, errors: list[dict]) -> dict[str, Any]:
    if output is None:
        return {}
    if not isinstance(output, dict):
        _err(errors, ["output"], "output must be an object", "model_type")
        return {}
    accepted: dict[str, Any] = {}
    for key, value in output.items():
        if key not in caps.OUTPUT_ALLOWED:
            message = caps.OUTPUT_UNSUPPORTED_MESSAGES.get(
                key, f"output.{key} is not supported by minimax_h3"
            )
            _err(errors, ["output", key], message, "unsupported_field")
            continue
        if key == "duration":
            if isinstance(value, bool) or not isinstance(value, int) or not (
                caps.DURATION_MIN <= value <= caps.DURATION_MAX
            ):
                _err(
                    errors,
                    ["output", key],
                    f"duration must be an integer between {caps.DURATION_MIN} and {caps.DURATION_MAX}",
                    "int_parsing",
                )
            else:
                accepted[key] = value
        elif key == "short_edge":
            if value != caps.SHORT_EDGE_FIXED:
                _err(
                    errors,
                    ["output", key],
                    f"minimax_h3 only supports short_edge={caps.SHORT_EDGE_FIXED}",
                    "value_error",
                )
            else:
                accepted[key] = value
        elif key == "ratio":
            if value not in caps.ASPECT_RATIOS:
                _err(
                    errors,
                    ["output", key],
                    f"ratio must be one of {', '.join(caps.ASPECT_RATIOS)}",
                    "enum",
                )
            else:
                accepted[key] = value
        elif key == "num_inference_steps":
            if isinstance(value, bool) or not isinstance(value, int) or not (
                caps.STEPS_MIN <= value <= caps.STEPS_MAX
            ):
                _err(
                    errors,
                    ["output", key],
                    f"num_inference_steps must be an integer between {caps.STEPS_MIN} and {caps.STEPS_MAX}",
                    "int_parsing",
                )
            else:
                accepted[key] = value
        elif key == "guidance_scale":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                _err(
                    errors,
                    ["output", key],
                    "guidance_scale must be a number >= 0",
                    "number_parsing",
                )
            else:
                accepted[key] = float(value)
    return accepted
