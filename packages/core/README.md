# sglang-h3-compat-core

Python wrapper package for the MiniMax-H3 SGLang Diffusion compatibility plugins.

This package has no SGLang plugin entry point. It:

- wraps the Rust `sglang_h3_native` module;
- installs SGLang HookRegistry callbacks;
- handles PyTorch tensors and SGLang request objects;
- provides `h3-compat-doctor`.

Install feature packages such as `sglang-h3-denoise-compat` or the `sglang-h3-compat` bundle rather than registering this package directly.

The implementation targets SGLang `0.5.22` and Linux x86-64.
