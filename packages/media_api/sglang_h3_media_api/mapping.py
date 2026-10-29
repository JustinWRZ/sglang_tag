"""Mapping from a validated media v1 request to the engine payload."""

from __future__ import annotations

from typing import Any

from . import capabilities as caps


def to_engine_payload(normalized: dict[str, Any]) -> dict[str, Any]:
    output = normalized.get("output") or {}
    steps = output.get("num_inference_steps")
    payload: dict[str, Any] = {
        "task": "ref2va",
        "prompt": normalized["prompt"],
        "target": {
            "short_edge": output.get("short_edge", caps.SHORT_EDGE_FIXED),
            "aspect_ratio": output.get("ratio", "16:9"),
            "duration_seconds": output.get("duration", caps.DURATION_MIN),
        },
        "conditions": [
            {"type": asset["type"], "uri": asset["url"], "role": asset["role"]}
            for asset in normalized["assets"]
        ],
        "num_inference_steps": steps if steps is not None else caps.DEFAULT_STEPS,
        "minimax_h3_sampler": "er_sde",
        "minimax_h3_sigmas": {"video": list(caps.WORKFLOW_SIGMAS)},
    }
    if "negative_prompt" in normalized:
        payload["negative_prompt"] = normalized["negative_prompt"]
    if "seed" in normalized:
        payload["seed"] = normalized["seed"]
    if "guidance_scale" in output:
        payload["guidance_scale"] = output["guidance_scale"]
    return payload
