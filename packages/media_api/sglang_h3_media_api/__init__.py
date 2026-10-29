"""sglang-h3-media-api: media v1 protocol plugin for MiniMax-H3 on SGLang."""

from __future__ import annotations

__version__ = "0.1.0"

_CREATE_APP_TARGET = (
    "sglang.multimodal_gen.runtime.entrypoints.http_server.create_app"
)
_SUPPORTED_SGLANG_VERSION = "0.5.22"


def _ensure_sglang_version() -> None:
    from importlib.metadata import version

    installed = version("sglang")
    if installed != _SUPPORTED_SGLANG_VERSION:
        raise RuntimeError(
            f"sglang-h3-media-api requires sglang=={_SUPPORTED_SGLANG_VERSION}, "
            f"found {installed}"
        )


def register() -> None:
    _ensure_sglang_version()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        HookRegistry,
        HookType,
    )

    from .routes import wrap_create_app

    HookRegistry.register(_CREATE_APP_TARGET, wrap_create_app, HookType.AROUND)
