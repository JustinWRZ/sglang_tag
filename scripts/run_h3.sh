#!/usr/bin/env bash
set -euo pipefail

# MiniMax-H3 launch script (plugin-standalone branch).
# Usage: MODEL_PATH=/path/to/model H3_VENV=/path/to/sglang/venv bash scripts/run_h3.sh
# All knobs are environment variables; DRY_RUN=1 prints the command without starting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- resolve the sglang venv (plugins must be installed in it) ---
H3_VENV="${H3_VENV:-${VENV:-}}"
if [[ -z "$H3_VENV" ]]; then
  for _cand in "$SCRIPT_DIR/../venv" "$SCRIPT_DIR/venv" "$PWD/venv"; do
    if [[ -x "$_cand/bin/sglang" ]]; then H3_VENV="$_cand"; break; fi
  done
fi
if [[ -z "$H3_VENV" || ! -x "$H3_VENV/bin/sglang" ]]; then
  printf 'error: set H3_VENV (or VENV) to the sglang venv that has the plugins installed\n' >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- CUDA_HOME: prefer venv-bundled nvcc (pip cuda-toolkit), else system CUDA ---
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _cand in "$H3_VENV"/lib/python*/site-packages/nvidia/cu13 /usr/local/cuda; do
    if [[ -x "$_cand/bin/nvcc" ]]; then export CUDA_HOME="$_cand"; break; fi
  done
fi
export PATH="$H3_VENV/bin${CUDA_HOME:+:$CUDA_HOME/bin}:$PATH"

export SGLANG_PLUGINS="${SGLANG_PLUGINS:-h3_request_fields,h3_denoise_compat,h3_state_export,h3_low_memory,h3_nvenc,h3_audio_loudness}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the minimax-h3 model directory}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sage_attn}"
VAE_PRECISION="${VAE_PRECISION:-bf16}"
export MINIMAX_H3_VAE_ENCODE_DTYPE="${MINIMAX_H3_VAE_ENCODE_DTYPE:-$VAE_PRECISION}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-auto}"
# OFFLOAD_MODE: full = 全量 offload（低显存推荐）；minimal = 只 dit layerwise；
#               off = 不传 offload 参数，交给引擎按显存自动决定
OFFLOAD_MODE="${OFFLOAD_MODE:-full}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_GPU_ID="${BASE_GPU_ID:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30314}"
MASTER_PORT="${MASTER_PORT:-31274}"
SCHEDULER_PORT="${SCHEDULER_PORT:-5578}"
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/outputs/}"
INPUT_SAVE_PATH="${INPUT_SAVE_PATH:-$SCRIPT_DIR/inputs/uploads}"

mkdir -p "$OUTPUT_PATH" "$INPUT_SAVE_PATH"

offload_args=()
case "$OFFLOAD_MODE" in
  full)
    offload_args=(
      --dit-cpu-offload
      --dit-layerwise-offload
      --layerwise-offload-components dit text_encoder image_encoder vae
      --dit-offload-prefetch-size 2.0
    )
    ;;
  minimal)
    offload_args=(
      --dit-layerwise-offload
      --layerwise-offload-components dit
      --dit-offload-prefetch-size 2.0
    )
    ;;
  off)
    ;;
  *)
    printf 'error: OFFLOAD_MODE must be full, minimal, or off (got: %s)\n' "$OFFLOAD_MODE" >&2
    exit 2
    ;;
esac

cmd=(
  "$H3_VENV/bin/sglang" serve
  --model-type diffusion
  --model-path "$MODEL_PATH"
  --attention-backend "$ATTENTION_BACKEND"
  --vae-precision "$VAE_PRECISION"
  --performance-mode "$PERFORMANCE_MODE"
  --num-gpus "$NUM_GPUS"
  --base-gpu-id "$BASE_GPU_ID"
  "${offload_args[@]}"
  --host "$HOST"
  --port "$PORT"
  --master-port "$MASTER_PORT"
  --scheduler-port "$SCHEDULER_PORT"
  --output-path "$OUTPUT_PATH"
  --input-save-path "$INPUT_SAVE_PATH"
  --warmup-mode off
  --log-level info
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
