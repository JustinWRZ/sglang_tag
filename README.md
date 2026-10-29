# SGLang MiniMax-H3 compatibility plugins

Mixed Python/Rust plugins for SGLang Diffusion `0.5.22`. The public SGLang plugin interface remains Python, while portable MiniMax-H3 computation and validation logic is implemented in Rust through PyO3.

The package set installs as normal Python wheels and does not modify installed SGLang source files.

## Architecture

```text
SGLang Diffusion
  ↓ sglang.multimodal_gen.plugins
Python feature packages and HookRegistry wrappers
  ↓ sglang-h3-compat-core
Rust sglang_h3_native PyO3 extension
```

Python keeps responsibilities that require direct SGLang or PyTorch object access:

- SGLang entry points
- HookRegistry wrappers
- SGLang request and batch objects
- PyTorch tensors and generators
- safetensors tensor I/O
- pipeline-stage monkey patches

Rust implements:

- request field validation
- manual sigma validation and retargeting
- ER-SDE step coefficients
- continuation metadata validation
- continuation report construction
- `h3-native-av-state/v1` manifest generation
- MiniMax-H3 latent-time conversion
- NVENC probing and FFmpeg command construction
- native build diagnostics

There is intentionally no Python fallback for the migrated logic. `sglang-h3-compat-core` hard-depends on the Linux x86-64 `sglang-h3-native` wheel.

## Compatibility

- SGLang: `0.5.22`
- Python: `>=3.11`
- Native platform: Linux x86_64
- PyO3 ABI: `abi3-py311`

The hooks target internal SGLang MiniMax-H3 APIs and are not compatible with other SGLang releases.

## Requirements and notes

- SGLang and all plugin distributions must be installed into the **same Python environment**. SGLang discovers plugins through installed distribution metadata, not by scanning arbitrary `.py` files.
- SGLang must be exactly `0.5.22`, with the `diffusion` extra enabled.
- The supported native platform is Linux x86_64, glibc `>=2.34`, Python `>=3.11`, with a compatible MiniMax-H3 model directory.
- Prebuilt wheels do not require Cargo, rustc, or maturin. Those tools are needed only when rebuilding from source.
- SageAttention is optional and is not bundled. If it is unavailable, SGLang falls back to FlashAttention; output behavior and performance may differ.
- NVENC requires an FFmpeg with `h264_nvenc`. If probing fails or NVENC is unavailable, output falls back to x264.
- Replace `/path/to/minimax-h3` in the launch examples with the actual local model path.

## Install SGLang

Use the official wheel when possible:

```bash
pip install "sglang[diffusion]==0.5.22"
```

If a specific SGLang source tree is required, run this from the SGLang repository root:

```bash
pip install -e 'python[diffusion]'
```

The SGLang 0.5.22 x86-64 wheel uses the `manylinux_2_34` platform tag. Use a host with glibc `>=2.34`, or install SGLang from source on an older host.

## Packages

| Package | Entry point | Purpose |
|---|---|---|
| `sglang-h3-native` | none | Rust/PyO3 core |
| `sglang-h3-compat-core` | none | shared Python wrappers and doctor |
| `sglang-h3-request-fields` | `h3_request_fields` | legacy request fields |
| `sglang-h3-denoise-compat` | `h3_denoise_compat` | manual sigmas, Euler/ER-SDE, continuation anchoring |
| `sglang-h3-state-export` | `h3_state_export` | latent state export |
| `sglang-h3-low-memory` | `h3_low_memory` | 32GB memory behavior |
| `sglang-h3-nvenc` | `h3_nvenc` | h264_nvenc output |
| `sglang-h3-audio-loudness` | `h3_audio_loudness` | audio loudness normalization |
| `sglang-h3-compat` | none | full bundle |

`h3_denoise_compat` and `h3_state_export` automatically register the request adapter when request fields are enabled. If all feature packages are installed, each target still receives exactly one hook.

## Package map (source → wheel)

Eight source packages produce nine wheels: the repo-root meta bundle is a package of its own.

| Wheel (0.3.0) | Source | Contents |
|---|---|---|
| `sglang-h3-compat` | repo root `pyproject.toml` + `sglang_h3_compat_bundle/` | Meta bundle: `__version__` only, no logic; depends on the six feature plugins |
| `sglang-h3-compat-core` | `packages/core/` | All real Python hooks: package `sglang_h3_compat` (13 modules — `registration`, `er_sde`, `sigma_schedule`, `state_export`, `request_lowering`, `nvenc`, `memory_compat`, …) |
| `sglang-h3-denoise-compat` | `packages/denoise/` | Thin register shell (`register()` → core `register_denoise_compat`) + plugin entry-point |
| `sglang-h3-request-fields` | `packages/request_fields/` | Same shell pattern |
| `sglang-h3-state-export` | `packages/state_encoder/` | Same shell pattern |
| `sglang-h3-low-memory` | `packages/low_memory/` | Same shell pattern |
| `sglang-h3-nvenc` | `packages/nvenc/` | Same shell pattern |
| `sglang-h3-audio-loudness` | `packages/audio/` | Same shell pattern |
| `sglang-h3-native` | `packages/native/` (Cargo + `src/`) | Rust core, compiled by maturin into `sglang_h3_native.abi3.so` |

SGLang discovers plugins through the `sglang.multimodal_gen.plugins` entry-points declared in each feature wheel's `pyproject.toml`; the implementations they call live in `sglang-h3-compat-core`. Only the six feature wheels are loaded via `SGLANG_PLUGINS`; bundle/core/native are dependency infrastructure. Rebuild everything with `python tools/build_packages.py` (output lands in `packages/dist/`, untracked).

## Install all features

### From the internal Gitea repository

Clone the plugin repository for the source; all prebuilt wheels — the nine plugin wheels (in the `sglang-h3-compat-0.3.1-wheels.zip` archive) and the third-party `sageattention-2.2.0` kernel wheel (PyPI has no 2.x) — are attached to the internal Gitea Release. Internal users set `GITEA_COMPAT_URL` to the repository clone URL and `GITEA_COMPAT_RELEASE` to the release download base:

```bash
GITEA_COMPAT_URL=<internal gitea sglang_compat clone url>
GITEA_COMPAT_RELEASE=<internal gitea sglang_compat release download base>
git clone "$GITEA_COMPAT_URL"
```

Public readers without access to the internal Gitea can build all nine wheels from this source tree with `python tools/build_packages.py`.

### From the internal Gitea Release archive

Download the complete wheel archive:

```bash
curl -fL -O "$GITEA_COMPAT_RELEASE/sglang-h3-compat-0.3.1-wheels.zip"
unzip sglang-h3-compat-0.3.1-wheels.zip
pip install --no-index --find-links ./dist sglang-h3-compat
```

With no `SGLANG_PLUGINS` value, every installed feature package is discovered automatically.

### From another local wheel directory

```bash
python -m venv .venv
source .venv/bin/activate

pip install "sglang[diffusion]==0.5.22"
pip install --find-links /path/to/sglang_h3_compat/packages/dist sglang-h3-compat
```

## Selective install

```bash
pip install --find-links ./dist sglang-h3-denoise-compat
pip install --find-links ./dist sglang-h3-state-export
pip install --find-links ./dist sglang-h3-low-memory
pip install --find-links ./dist sglang-h3-nvenc
pip install --find-links ./dist sglang-h3-audio-loudness
pip install --find-links ./dist sglang-h3-request-fields
```

To filter installed plugins:

```bash
SGLANG_PLUGINS=h3_denoise_compat
SGLANG_PLUGINS=h3_state_export
SGLANG_PLUGINS=h3_request_fields,h3_denoise_compat,h3_state_export,h3_low_memory,h3_nvenc,h3_audio_loudness
```

## Request fields

The request-fields feature accepts:

```json
{
  "minimax_h3_sampler": "er_sde",
  "minimax_h3_sigmas": {
    "video": [1.0, 0.9844, 0.9627, 0.9304, 0.878, 0.7579, 0.5527, 0.2368, 0.0]
  },
  "export_state": true,
  "h3_continuation": {
    "state_path": "/path/to/previous.state.safetensors",
    "anchor_latent_frames": 5
  }
}
```

Rust validates and prepares the schedules; Python forwards the resulting values into the MiniMax-H3 pipeline.

## Recommended launch

```bash
MINIMAX_H3_VAE_ENCODE_DTYPE=bf16 \
SGLANG_PLUGINS=h3_request_fields,h3_denoise_compat,h3_state_export,h3_low_memory,h3_nvenc,h3_audio_loudness \
sglang serve \
  --model-type diffusion \
  --model-path /path/to/minimax-h3 \
  --attention-backend sage_attn \
  --vae-precision bf16 \
  --performance-mode auto \
  --num-gpus 1 \
  --base-gpu-id 0 \
  --dit-cpu-offload \
  --dit-layerwise-offload \
  --layerwise-offload-components dit text_encoder image_encoder vae
```

SageAttention is not bundled. Install a compatible SageAttention build separately when the old attention performance and behavior are required.

## Environment switches

| Variable | Default | Meaning |
|---|---:|---|
| `H3_COMPAT_REQUEST_FIELDS` | `1` | request-field lowering |
| `H3_COMPAT_SAMPLER` | `1` | denoise compatibility |
| `H3_COMPAT_STATE_IO` | `1` | state export |
| `H3_COMPAT_MLP_CHUNK` | `1` | MLP chunking |
| `H3_COMPAT_AUDIO_LOUDNESS` | `1` | audio loudness normalization |
| `H3_COMPAT_NVENC` | `1` | use NVENC when available |
| `H3_COMPAT_FFMPEG_PATH` | `/usr/bin/ffmpeg` | FFmpeg executable |
| `H3_COMPAT_NVENC_CQ` | `25` | NVENC CQ value |
| `H3_COMPAT_NVENC_LIB_PATH` | unset | optional `LD_LIBRARY_PATH` entry |
| `MINIMAX_H3_VAE_ENCODE_DTYPE` | unset | set to `bf16` for the 32GB VAE path |

## Doctor

```bash
h3-compat-doctor
```

The doctor reports:

- SGLang and native versions
- native build profile
- installed feature packages
- discovered plugin entry points
- SageAttention availability
- NVENC probe result
- active feature switches

## Build

Requirements:

- rustup
- Rust `1.75.0`, selected automatically by `packages/native/rust-toolchain.toml`
- `maturin[zig]`, invoked automatically through `uv tool run`
- uv

From the repository root:

```bash
python tools/build_packages.py
```

The script runs:

```text
cargo fmt --check
cargo clippy -D warnings
cargo test
native wheel build
all Python wheel builds
```

Wheels are written to:

```text
packages/dist/
```

Current native wheel:

```text
sglang_h3_native-0.3.0-cp311-abi3-manylinux_2_34_x86_64.whl
```

## Tests

Rust is checked with Cargo. Python tests live in `packages/core/tests` and cover:

- request lowering
- feature flags
- native module import
- sigma preparation
- ER-SDE coefficient parity against the former Torch implementation
- state-manifest generation
- latent-time conversion
- NVENC command construction
- continuation report generation

```bash
cd packages/native
cargo test

cd ../..
PYTHONPATH=packages/core python -m unittest discover -s packages/core/tests -v
```
