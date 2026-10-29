"""MiniMax-H3 capability matrix for the media v1 protocol.

Single source of truth for what this service supports and the exact
rejection message returned for every unsupported feature. The B.6 422
responses quote these strings verbatim; the package README mirrors this
matrix in human-readable form. Change support here and only here.
"""

from __future__ import annotations

import os

SERVICE_NAME = os.environ.get("MEDIA_SERVICE_NAME", "minimax-h3")
API_NAME = "media"
API_VERSION = "1"
MODALITIES = ["video", "audio"]
MODE = "sync+background"

DEFAULT_BACKGROUND = os.environ.get("MEDIA_DEFAULT_BACKGROUND", "true").lower() in (
    "1",
    "true",
    "yes",
)
SYNC_WAIT_S = float(os.environ.get("MEDIA_SYNC_WAIT_S", "300"))
SUBMIT_TIMEOUT_S = float(os.environ.get("MEDIA_SUBMIT_TIMEOUT_S", "30"))
ARTIFACT_TTL_S = float(os.environ.get("MEDIA_ARTIFACT_TTL_S", "86400"))


def service_models() -> list[str]:
    raw = os.environ.get("MEDIA_MODELS", "")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return ["minimax-h3-cobijada-fp8"]


PROMPT_MAX_LENGTH = 32000
SEED_MIN = 0
SEED_MAX = 2147483647
BATCH_MAX = 10
JOB_ID_PATTERN = r"^[0-9a-fA-F]{32}$"

SHORT_EDGE_FIXED = 768
ASPECT_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4")
DURATION_MIN = 4
DURATION_MAX = 15
STEPS_MIN = 1
STEPS_MAX = 50

DEFAULT_STEPS = 10
WORKFLOW_SIGMAS = (
    1.0, 0.9882, 0.973, 0.9524, 0.9231,
    0.878, 0.8, 0.6316, 0.4737, 0.1579, 0.0,
)

ROOT_ALLOWED = frozenset(
    {
        "model",
        "prompt",
        "negative_prompt",
        "assets",
        "seed",
        "batch",
        "n",
        "output",
        "background",
        "job_id",
    }
)
ASSET_ALLOWED = frozenset({"type", "url", "role"})
ASSET_TYPES = ("image",)
ASSET_ROLES = ("reference",)

OUTPUT_ALLOWED = frozenset(
    {
        "duration",
        "short_edge",
        "ratio",
        "num_inference_steps",
        "guidance_scale",
    }
)

OUTPUT_UNSUPPORTED_MESSAGES = {
    "size": (
        "minimax_h3 derives output size from short_edge (768) and ratio; "
        "output.size is not supported"
    ),
    "quality": (
        "minimax_h3 serves a fixed quality tier on this deployment; "
        "output.quality is not supported"
    ),
    "output_format": "minimax_h3 always produces MP4; output.output_format is not supported",
    "output_compression": "minimax_h3 always produces MP4; output.output_compression is not supported",
    "background": "output.background (opaque/transparent) is not supported by minimax_h3",
    "moderation": "output.moderation is not supported by minimax_h3",
    "input_fidelity": "output.input_fidelity is not supported by minimax_h3",
    "style": "output.style is not supported by minimax_h3",
    "resolution": (
        "minimax_h3 resolution is fixed by short_edge=768; "
        "output.resolution is not supported"
    ),
    "frames": "output.frames is not supported; use output.duration (4-15 seconds)",
    "camera_fixed": "output.camera_fixed is not supported by minimax_h3",
    "generate_audio": (
        "minimax_h3 ref2va always generates audio with the video; "
        "output.generate_audio is not applicable"
    ),
    "return_last_frame": "output.return_last_frame is not supported by minimax_h3",
    "watermark": "output.watermark is not supported by minimax_h3",
    "prompt_extend": "output.prompt_extend is not supported by minimax_h3",
}

ASSET_TYPE_UNSUPPORTED = {
    "video": "video assets are not supported by minimax_h3; use image assets as reference",
    "audio": "audio assets are not supported by minimax_h3; use image assets as reference",
}
ASSET_ROLE_UNSUPPORTED = {
    "mask": "mask assets are not supported by minimax_h3; only role=reference is supported",
}
BATCH_UNSUPPORTED_MESSAGE = (
    "minimax_h3 produces exactly one output per job; batch > 1 is not supported"
)
ASSET_REQUIRED_MESSAGE = (
    "minimax_h3 serves task ref2va; at least one image asset with "
    "role=reference is required"
)
MODEL_UNSUPPORTED_MESSAGE = (
    "model must be one of the service model whitelist ({models}); "
    "omit it to use the service default"
)
