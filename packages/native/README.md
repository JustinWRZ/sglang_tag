# sglang-h3-native

Rust/PyO3 core used by the SGLang MiniMax-H3 compatibility plugins.

The SGLang plugin entry points remain Python packages. This module implements the portable computation and validation core:

- request value validation
- MiniMax-H3 sigma schedules
- ER-SDE step coefficients
- continuation metadata validation and reports
- `h3-native-av-state/v1` manifest generation
- NVENC probing and FFmpeg command construction

The Python packages hard-depend on this module and do not provide a fallback implementation.
