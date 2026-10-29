from sglang_h3_compat.registration import register_request_fields

__version__ = "0.3.1"
__all__ = ["register", "__version__"]


def register() -> None:
    register_request_fields()
