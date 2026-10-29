from __future__ import annotations

import importlib.metadata
import logging

from . import memory_compat
from .continuation import (
    branch_init_after,
    denoise_loop_around,
    expand_initial_rows_after,
    maybe_enable_cache_dit_around,
    prepare_timestep_plan_around,
    run_full_loop_around,
)
from .decoding_hooks import canonical_audio_after, decoding_forward_before
from .flags import (
    loudness_enabled,
    mlp_chunk_enabled,
    request_fields_enabled,
    sampler_enabled,
    state_io_enabled,
)
from .nvenc import encoder_open_before, ffmpeg_command_after
from .request_lowering import lower_video_request_kwargs_after
from .sigma_schedule import install_schedules_before

logger = logging.getLogger(__name__)

_SUPPORTED_SGLANG_VERSION = "0.5.22"
_request_fields_registered = False

_ADAPTER_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter.MiniMaxH3VideoModelAdapter.lower_video_request_kwargs"
_TIMESTEP_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation.MiniMaxH3TimestepPreparationStage.forward"
_DECODING_FORWARD_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding.MiniMaxH3DecodingStage.forward"
_DENoise_RUN_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising.MiniMaxH3DenoisingStage._run_full_loop"
_CACHE_DIT_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising.MiniMaxH3DenoisingStage._maybe_enable_cache_dit_and_torch_compile"
_BRANCH_INIT_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop.MiniMaxH3DenoiseBranch.__init__"
_BRANCH_PLAN_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop.MiniMaxH3DenoiseBranch.prepare_timestep_plan"
_INITIAL_ROWS_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising._expand_initial_rows"
_DENOISE_LOOP_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop.minimax_h3_denoise_loop"
_AUDIO_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding._canonical_output_audio_waveform"
_MLP_TARGET = "sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3MLP.forward"
_VAE_ENCODE_TARGET = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae.AutoencoderKLLegacy.encode_base"
_SCOPED_FP32_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding.minimax_h3_scoped_encode_fp32"
_KEYFRAME_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding.minimax_h3_encode_keyframe_cond_rows"
_REFERENCE_TARGET = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding.minimax_h3_encode_reference_video_rows"
_ENCODER_OPEN_TARGET = "sglang.multimodal_gen.runtime.entrypoints.utils.CudaVideoEncoder.open"
_FFMPEG_COMMAND_TARGET = "sglang.multimodal_gen.runtime.entrypoints.utils._cuda_video_ffmpeg_command"


def ensure_sglang_version() -> None:
    version = importlib.metadata.version("sglang")
    if version != _SUPPORTED_SGLANG_VERSION:
        raise RuntimeError(
            f"sglang-h3-compat requires SGLang {_SUPPORTED_SGLANG_VERSION}, "
            f"found {version}"
        )


def register_request_fields() -> None:
    global _request_fields_registered
    if _request_fields_registered:
        return
    if not request_fields_enabled():
        return
    ensure_sglang_version()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        HookRegistry,
        HookType,
    )

    logger.info("registering h3_request_fields")
    HookRegistry.register(_ADAPTER_TARGET, lower_video_request_kwargs_after, HookType.AFTER)
    _request_fields_registered = True


def register_denoise_compat() -> None:
    if not sampler_enabled():
        return
    ensure_sglang_version()
    register_request_fields()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        HookRegistry,
        HookType,
    )

    logger.info("registering h3_denoise_compat")
    HookRegistry.register(_TIMESTEP_TARGET, install_schedules_before, HookType.BEFORE)
    HookRegistry.register(_DENoise_RUN_TARGET, run_full_loop_around, HookType.AROUND)
    HookRegistry.register(_CACHE_DIT_TARGET, maybe_enable_cache_dit_around, HookType.AROUND)
    HookRegistry.register(_BRANCH_INIT_TARGET, branch_init_after, HookType.AFTER)
    HookRegistry.register(_BRANCH_PLAN_TARGET, prepare_timestep_plan_around, HookType.AROUND)
    HookRegistry.register(_INITIAL_ROWS_TARGET, expand_initial_rows_after, HookType.AFTER)
    HookRegistry.register(_DENOISE_LOOP_TARGET, denoise_loop_around, HookType.AROUND)


def register_state_export() -> None:
    if not state_io_enabled():
        return
    ensure_sglang_version()
    register_request_fields()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        HookRegistry,
        HookType,
    )

    logger.info("registering h3_state_export")
    HookRegistry.register(_DECODING_FORWARD_TARGET, decoding_forward_before, HookType.BEFORE)


def register_low_memory() -> None:
    ensure_sglang_version()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        HookRegistry,
        HookType,
    )

    logger.info("registering h3_low_memory")
    if mlp_chunk_enabled():
        HookRegistry.register(_MLP_TARGET, memory_compat.mlp_forward_around, HookType.AROUND)
    HookRegistry.register(_VAE_ENCODE_TARGET, memory_compat.vae_encode_base_around, HookType.AROUND)
    HookRegistry.register(_SCOPED_FP32_TARGET, memory_compat.scoped_encode_fp32_around, HookType.AROUND)
    HookRegistry.register(_KEYFRAME_TARGET, memory_compat.keyframe_encode_around, HookType.AROUND)
    HookRegistry.register(_REFERENCE_TARGET, memory_compat.reference_video_encode_around, HookType.AROUND)


def register_nvenc() -> None:
    ensure_sglang_version()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        HookRegistry,
        HookType,
    )

    logger.info("registering h3_nvenc")
    HookRegistry.register(_ENCODER_OPEN_TARGET, encoder_open_before, HookType.BEFORE)
    HookRegistry.register(_FFMPEG_COMMAND_TARGET, ffmpeg_command_after, HookType.AFTER)


def register_audio_loudness() -> None:
    if not loudness_enabled():
        return
    ensure_sglang_version()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        HookRegistry,
        HookType,
    )

    logger.info("registering h3_audio_loudness")
    HookRegistry.register(_AUDIO_TARGET, canonical_audio_after, HookType.AFTER)


def register() -> None:
    register_request_fields()
    register_denoise_compat()
    register_state_export()
    register_low_memory()
    register_nvenc()
    register_audio_loudness()
