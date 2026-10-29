from __future__ import annotations

import importlib.metadata
import os
import sys
from collections.abc import Callable

from .flags import (
    ffmpeg_path,
    loudness_enabled,
    mlp_chunk_enabled,
    nvenc_cq,
    nvenc_enabled,
    nvenc_library_path,
    request_fields_enabled,
    sampler_enabled,
    state_io_enabled,
)
from .native import build_profile, native_version, probe_nvenc

PLUGIN_GROUP = "sglang.multimodal_gen.plugins"
SUPPORTED_SGLANG = "0.5.22"
FEATURE_PACKAGES = {
    "request_fields": "sglang-h3-request-fields",
    "denoise_compat": "sglang-h3-denoise-compat",
    "state_export": "sglang-h3-state-export",
    "low_memory": "sglang-h3-low-memory",
    "nvenc": "sglang-h3-nvenc",
    "audio_loudness": "sglang-h3-audio-loudness",
}


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_nvenc() -> tuple[bool, str]:
    if not nvenc_enabled():
        return False, "disabled"
    try:
        available = probe_nvenc(
            ffmpeg_path(),
            nvenc_library_path(),
            timeout_ms=8000,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return available, "available" if available else "probe failed"


def check_sageattention() -> tuple[bool, str]:
    try:
        import sageattention
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    location = getattr(sageattention, "__file__", "installed")
    return True, location


def print_section(title: str, check: Callable[[], tuple[bool, str]]) -> None:
    ok, detail = check()
    print(f"{'PASS' if ok else 'WARN'} {title}: {detail}")


def main() -> int:
    print("SGLang H3 compatibility doctor")
    print(f"Python executable: {sys.executable}")
    print(f"native backend: rust")
    print(f"native version: {native_version()}")
    print(f"native build profile: {build_profile()}")
    sglang_version = installed_version("sglang")
    print(f"sglang: {sglang_version}")
    if sglang_version != SUPPORTED_SGLANG:
        print(f"ERROR expected SGLang {SUPPORTED_SGLANG}")
    core_version = installed_version("sglang-h3-compat-core")
    print(f"core: {core_version}")
    for feature, package in FEATURE_PACKAGES.items():
        version = installed_version(package)
        print(f"{feature}: {version if version else 'not installed'}")
    print("discovered plugin entry points:")
    for entry_point in importlib.metadata.entry_points(group=PLUGIN_GROUP):
        print(f"  {entry_point.name} -> {entry_point.value}")
    print_section("SageAttention", check_sageattention)
    print_section("NVENC", check_nvenc)
    print(
        "flags: "
        f"request_fields={request_fields_enabled()} "
        f"denoise={sampler_enabled()} "
        f"state_io={state_io_enabled()} "
        f"mlp_chunk={mlp_chunk_enabled()} "
        f"audio_loudness={loudness_enabled()} "
        f"nvenc={nvenc_enabled()} "
        f"nvenc_cq={nvenc_cq()} "
        f"ffmpeg={ffmpeg_path()} "
        f"nvenc_lib={nvenc_library_path()} "
        f"ld_library_path={os.environ.get('LD_LIBRARY_PATH')}"
    )
    return 0 if sglang_version == SUPPORTED_SGLANG else 1


if __name__ == "__main__":
    raise SystemExit(main())
