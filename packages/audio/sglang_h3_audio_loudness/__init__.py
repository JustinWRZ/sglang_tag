from sglang_h3_compat.registration import register_audio_loudness

__version__ = "0.3.1"
__all__ = ["register", "__version__"]


def register() -> None:
    register_audio_loudness()
