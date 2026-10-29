import os

SAMPLER_EXTRA_KEY = "h3_compat:minimax_h3_sampler"
EXPORT_STATE_FIELD = "export_state"
CONTINUATION_FIELD = "h3_continuation"
CONTINUATION_REPORT_KEY = "h3_compat:minimax_h3_continuation_report"
EULER_SAMPLER = "euler"
ER_SDE_SAMPLER = "er_sde"
MLP_CHUNK_TOKENS = 16384
MLP_CHUNK_THRESHOLD = 32768


def env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def request_fields_enabled() -> bool:
    return env_enabled("H3_COMPAT_REQUEST_FIELDS", True)


def sampler_enabled() -> bool:
    return env_enabled("H3_COMPAT_SAMPLER", True)


def state_io_enabled() -> bool:
    return env_enabled("H3_COMPAT_STATE_IO", True)


def loudness_enabled() -> bool:
    return env_enabled("H3_COMPAT_AUDIO_LOUDNESS", True)


def mlp_chunk_enabled() -> bool:
    return env_enabled("H3_COMPAT_MLP_CHUNK", True)


def vae_bf16_enabled() -> bool:
    return (
        os.environ.get("MINIMAX_H3_VAE_ENCODE_DTYPE", "").strip().lower()
        == "bf16"
    )


def nvenc_enabled() -> bool:
    return env_enabled("H3_COMPAT_NVENC", True)


def nvenc_library_path() -> str | None:
    value = os.environ.get("H3_COMPAT_NVENC_LIB_PATH", "").strip()
    return value or None


def ffmpeg_path() -> str:
    return os.environ.get("H3_COMPAT_FFMPEG_PATH", "").strip() or "/usr/bin/ffmpeg"


def nvenc_cq() -> int:
    try:
        return int(os.environ.get("H3_COMPAT_NVENC_CQ", "25"))
    except ValueError:
        return 25
