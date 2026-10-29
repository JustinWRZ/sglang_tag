from __future__ import annotations

import functools
import os
from typing import Any

from .flags import ffmpeg_path, nvenc_cq, nvenc_enabled, nvenc_library_path
from .native import (
    build_nvenc_command,
    merge_ld_library_path,
    probe_nvenc,
)


def _update_ffmpeg_environment() -> None:
    merged = merge_ld_library_path(
        os.environ.get("LD_LIBRARY_PATH"), nvenc_library_path()
    )
    if merged is not None:
        os.environ["LD_LIBRARY_PATH"] = merged


@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    return nvenc_enabled() and bool(
        probe_nvenc(
            ffmpeg_path(),
            nvenc_library_path(),
            timeout_ms=5000,
        )
    )


def encoder_open_before(*args: Any, **kwargs: Any):
    if nvenc_available():
        _update_ffmpeg_environment()
    return None


def ffmpeg_command_after(
    result: list[str], save_file_path: str, **kwargs: Any
):
    del save_file_path
    if not nvenc_available():
        return result
    return build_nvenc_command(
        list(result),
        ffmpeg_path(),
        nvenc_cq(),
    )
