from sglang_h3_compat.registration import register_denoise_compat

__version__ = "0.3.1"
__all__ = ["register", "__version__"]


def register() -> None:
    register_denoise_compat()
