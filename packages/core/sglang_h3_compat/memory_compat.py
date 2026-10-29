from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from .flags import (
    MLP_CHUNK_THRESHOLD,
    MLP_CHUNK_TOKENS,
    mlp_chunk_enabled,
    vae_bf16_enabled,
)


def mlp_forward_around(original_fn: Any, self: Any, x: torch.Tensor):
    if not mlp_chunk_enabled() or x.shape[0] <= MLP_CHUNK_THRESHOLD:
        return original_fn(self, x)
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _silu_mul

    output = torch.empty(
        (x.shape[0], self.fc2.output_size), dtype=x.dtype, device=x.device
    )
    for start in range(0, x.shape[0], MLP_CHUNK_TOKENS):
        stop = min(start + MLP_CHUNK_TOKENS, x.shape[0])
        hidden, _ = self.fc1(x[start:stop])
        hidden = _silu_mul(hidden, reuse_input=self.reuse_fc1_activation)
        output[start:stop], _ = self.fc2(hidden)
    return output


def vae_encode_base_around(
    original_fn: Any,
    self: Any,
    input: torch.Tensor,
    process_image: bool = False,
):
    if vae_bf16_enabled():
        weight_dtype = next(self.parameters()).dtype
        if input.dtype != weight_dtype:
            input = input.to(weight_dtype)
    return original_fn(self, input, process_image=process_image)


def scoped_encode_fp32_around(original_fn: Any, video_vae: Any):
    if not vae_bf16_enabled():
        return original_fn(video_vae)
    return nullcontext()


def keyframe_encode_around(
    original_fn: Any,
    video_vae: Any,
    image: Any,
    arch_config: Any,
):
    if not vae_bf16_enabled():
        return original_fn(video_vae, image, arch_config)
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
        MINIMAX_H3_KEYFRAME_ENCODE_SEED,
        MINIMAX_H3_KEYFRAME_PATCH_SIZE,
        _cached_latent_mean_std,
        minimax_h3_scoped_encode_rng,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )

    parameter = next(video_vae.parameters())
    with minimax_h3_scoped_encode_rng(
        MINIMAX_H3_KEYFRAME_ENCODE_SEED, parameter.device
    ):
        latent = video_vae.encode_images(image, use_fp16_latent=True)[0]
    latent = latent.cpu().float()
    if latent.dim() == 4:
        latent = latent[None]
    channels = int(arch_config.latent_channels)
    mean, std = _cached_latent_mean_std(
        tuple(arch_config.latents_mean),
        tuple(arch_config.latents_std),
        (1, channels, 1, 1, 1),
    )
    latent.sub_(mean).div_(std)
    rows = minimax_h3_patchify_video_latent(
        latent, patch_size=list(MINIMAX_H3_KEYFRAME_PATCH_SIZE)
    )
    return rows.to(torch.float32)


def reference_video_encode_around(
    original_fn: Any,
    video_vae: Any,
    frames: Any,
    arch_config: Any,
):
    if not vae_bf16_enabled():
        return original_fn(video_vae, frames, arch_config)
    import numpy as np

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
        _cached_latent_mean_std,
        minimax_h3_scoped_encode_rng,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
        MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED,
        MINIMAX_H3_REFERENCE_VIDEO_PATCH_SIZE,
    )

    frames = np.asarray(frames)
    parameter = next(video_vae.parameters())
    with minimax_h3_scoped_encode_rng(
        MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED, parameter.device
    ):
        latent = video_vae.encode_videos(frames, use_fp16_latent=True)[0]
    latent = latent.cpu().float()
    if latent.dim() == 4:
        latent = latent[None]
    channels = int(arch_config.latent_channels)
    latent_t, latent_h, latent_w = (
        int(latent.shape[2]),
        int(latent.shape[3]),
        int(latent.shape[4]),
    )
    mean, std = _cached_latent_mean_std(
        tuple(arch_config.latents_mean),
        tuple(arch_config.latents_std),
        (1, channels, 1, 1, 1),
    )
    latent.sub_(mean).div_(std)
    rows = minimax_h3_patchify_video_latent(
        latent, patch_size=list(MINIMAX_H3_REFERENCE_VIDEO_PATCH_SIZE)
    )
    return rows.to(torch.float32), latent_t, latent_h, latent_w
