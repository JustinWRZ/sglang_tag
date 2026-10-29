from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass
from typing import Any

import torch

from .compat_constants import H3_COMPAT_CONTINUATION_FIELD
from .er_sde import er_sde_step_scalars, predict_denoised_rows, update_target_rows
from .flags import (
    CONTINUATION_REPORT_KEY,
    ER_SDE_SAMPLER,
    EULER_SAMPLER,
    SAMPLER_EXTRA_KEY,
)
from .native import validate_continuation_json
from .state_export import load_state_for_continuation


@dataclass
class ActiveH3Request:
    batch_id: Any
    sampler: str
    anchor_rows: torch.Tensor | None = None
    anchor_report: dict[str, Any] | None = None
    anchored_timestep: float | None = None
    noise_generator: torch.Generator | None = None


_active_request: contextvars.ContextVar[ActiveH3Request | None] = (
    contextvars.ContextVar("h3_compat_active_request", default=None)
)


def _compat_options(batch: Any) -> dict[str, Any]:
    options = getattr(batch.sampling_params, "diffusers_kwargs", None) or {}
    return options if isinstance(options, dict) else {}


def _load_anchor(
    batch: Any, continuation: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any], float | None]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
        _resolve_full_loop_context,
    )

    ctx = _resolve_full_loop_context(batch)
    tensors, manifest = load_state_for_continuation(
        str(continuation["state_path"])
    )
    video = tensors["video.latent"]
    src_t = int(video.shape[2])
    sampler = str(
        batch.extra.get(SAMPLER_EXTRA_KEY)
        or _compat_options(batch).get("minimax_h3_sampler")
        or EULER_SAMPLER
    )
    current_sigmas = [float(value) for value in ctx.sigmas["video"]]
    requested = int(continuation["anchor_latent_frames"])
    count = min(requested, src_t)
    tail = video[:, :, src_t - count :, :, :]
    rows = minimax_h3_patchify_video_latent(tail, patch_size=[1, 2, 2])
    rows = rows.to(torch.float32).contiguous()
    continuation_input = {
        "continuation": continuation,
        "manifest": manifest,
        "target": {
            "latent_t": int(ctx.latent_t),
            "latent_h": int(ctx.latent_h),
            "latent_w": int(ctx.latent_w),
        },
        "current_sampler": sampler,
        "current_sigmas": current_sigmas,
        "anchor_row_count": int(rows.shape[0]),
    }
    report_json = validate_continuation_json(
        json.dumps(continuation_input, ensure_ascii=False)
    )
    report = json.loads(report_json)
    anchored_timestep_raw = continuation.get("anchored_timestep")
    anchored_timestep = (
        None
        if anchored_timestep_raw is None
        else float(anchored_timestep_raw)
    )
    return rows, report, anchored_timestep


def run_full_loop_around(
    original_fn: Any,
    self: Any,
    batch: Any,
    server_args: Any,
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
        minimax_h3_plan_from_batch,
    )

    from .sigma_schedule import install_schedules_before

    install_schedules_before(batch, minimax_h3_plan_from_batch(batch))
    options = _compat_options(batch)
    sampler = str(
        batch.extra.get(SAMPLER_EXTRA_KEY)
        or options.get("minimax_h3_sampler")
        or EULER_SAMPLER
    )
    batch.extra[SAMPLER_EXTRA_KEY] = sampler
    continuation = options.get(H3_COMPAT_CONTINUATION_FIELD)
    anchor_rows = None
    report = None
    anchored_timestep = None
    if continuation is not None:
        anchor_rows, report, anchored_timestep = _load_anchor(batch, continuation)
    generator = None
    if sampler == ER_SDE_SAMPLER:
        plan = batch.extra.get("minimax_h3_resolved_plan")
        seed = getattr(plan, "seed", None)
        if seed is None:
            seed = getattr(batch.sampling_params, "seed", 42)
            if isinstance(seed, list) and seed:
                seed = seed[0]
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
    active = ActiveH3Request(
        batch_id=id(batch),
        sampler=sampler,
        anchor_rows=anchor_rows,
        anchor_report=report,
        anchored_timestep=anchored_timestep,
        noise_generator=generator,
    )
    token = _active_request.set(active)
    try:
        return original_fn(self, batch, server_args)
    finally:
        current = _active_request.get()
        if current is not None and current.anchor_report is not None:
            batch.extra[CONTINUATION_REPORT_KEY] = current.anchor_report
        _active_request.reset(token)


def maybe_enable_cache_dit_around(
    original_fn: Any,
    self: Any,
    *args: Any,
    **kwargs: Any,
):
    active = _active_request.get()
    if active is not None and active.sampler == ER_SDE_SAMPLER:
        return None
    return original_fn(self, *args, **kwargs)


def branch_init_after(result: Any, self: Any, *args: Any, **kwargs: Any):
    active = _active_request.get()
    if active is None or active.anchor_rows is None:
        return result
    count = int(active.anchor_rows.shape[0])
    if count < 0 or count > int(self.img_target_seq_idx.numel()):
        raise ValueError("h3_continuation anchor rows exceed video target rows")
    self.anchor_target_rows = count
    self.anchored_seq_idx = self.img_target_seq_idx[:count]
    self.n_video_timestep_rows = (
        int(self.seq_len)
        - int(self.img_cond_seq_idx.numel())
        - int(self.audio_target_seq_idx.numel())
        - int(self.audio_ref_seq_idx.numel())
        - count
    )
    return result


def prepare_timestep_plan_around(
    original_fn: Any,
    self: Any,
    **kwargs: Any,
):
    plans = original_fn(self, **kwargs)
    active = _active_request.get()
    if active is None or active.anchor_rows is None:
        return plans
    anchored_timestep = (
        active.anchored_timestep
        if active.anchored_timestep is not None
        else 0.999
    )
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MINIMAX_H3_ADALN_MODALITY_NUM,
    )

    patched: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for unique_timesteps, inverse_indices, block_combined in plans:
        values = [float(value) for value in unique_timesteps.tolist()]
        if anchored_timestep not in values:
            old_values = values
            new_values = sorted(old_values + [anchored_timestep])
            old_to_new = {
                old_index: new_index
                for new_index, value in enumerate(new_values)
                if (old_index := next(
                    (i for i, old in enumerate(old_values) if old == value),
                    None,
                )) is not None
            }
            anchored_index = new_values.index(anchored_timestep)
            new_inverse = torch.empty_like(inverse_indices)
            for old_index, new_index in old_to_new.items():
                new_inverse.masked_fill_(
                    inverse_indices == old_index, new_index
                )
            new_inverse.index_fill_(
                0, self.anchored_seq_idx, anchored_index
            )
            new_unique = torch.tensor(
                new_values,
                dtype=unique_timesteps.dtype,
                device=unique_timesteps.device,
            )
            new_block = torch.add(
                self.block_token_tags,
                new_inverse[self.local_row_slice],
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
            patched.append((new_unique, new_inverse, new_block))
        else:
            new_inverse = inverse_indices.clone()
            new_inverse.index_fill_(
                0,
                self.anchored_seq_idx,
                values.index(anchored_timestep),
            )
            new_block = torch.add(
                self.block_token_tags,
                new_inverse[self.local_row_slice],
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
            patched.append(
                (unique_timesteps, new_inverse, new_block)
            )
    return patched


def expand_initial_rows_after(result: Any, *args: Any, **kwargs: Any):
    active = _active_request.get()
    if active is None or active.anchor_rows is None:
        return result
    positive = kwargs.get("positive") or (
        args[1] if len(args) > 1 else None
    )
    if positive is None:
        raise ValueError("MiniMax H3 initial-row context is missing positive")
    initial_video, initial_audio = result
    count = int(active.anchor_rows.shape[0])
    initial_video = initial_video.clone()
    initial_video[
        positive.video_target_start : positive.video_target_start + count
    ] = active.anchor_rows.to(
        device=initial_video.device, dtype=initial_video.dtype
    )
    return initial_video, initial_audio


def denoise_loop_around(original_fn: Any, **kwargs: Any):
    active = _active_request.get()
    needs_compat = (
        active is not None
        and (
            active.sampler == ER_SDE_SAMPLER or active.anchor_rows is not None
        )
        and kwargs.get("rollout_ctx") is None
    )
    if not needs_compat:
        return original_fn(**kwargs)
    return _compat_denoise_loop(original_fn=original_fn, active=active, **kwargs)


def _compat_denoise_loop(
    *,
    original_fn: Any,
    active: ActiveH3Request,
    model: Any,
    model_forward: Any,
    positive: Any,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    audio_ref_rows: torch.Tensor | None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    imgvid_cond_noise_aug_for_inference: float,
    audio_cond_noise_aug_for_inference: float,
    attn_metadata: Any = None,
    on_step: Any = None,
    step_profiler: Any = None,
    **_unused: Any,
):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    del original_fn
    if len(sigmas_video) != len(sigmas_audio) or len(sigmas_video) < 2:
        raise ValueError("invalid MiniMax H3 sigma schedules")
    video_rows = initial_video_rows.to(
        device=device, dtype=torch.float32, copy=True
    )
    audio_rows = initial_audio_rows.to(
        device=device, dtype=torch.float32, copy=True
    )
    if keyframe_cond_rows is not None:
        video_rows.index_copy_(
            0,
            positive.cond_row_idx,
            keyframe_cond_rows.to(device=device, dtype=torch.float32),
        )
    if audio_ref_rows is not None:
        audio_rows.index_copy_(
            0,
            positive.audio_ref_row_idx,
            audio_ref_rows.to(device=device, dtype=torch.float32),
        )
    anchor_state = None
    if active.anchor_rows is not None:
        anchor_state = active.anchor_rows.to(device=device, dtype=torch.float32)
        video_rows[positive.video_target_slice][: anchor_state.shape[0]].copy_(
            anchor_state
        )

    video_target_slice = positive.video_target_slice
    audio_target_slice = positive.audio_target_slice
    video_timesteps = [1.0 - sigma for sigma in sigmas_video[:-1]]
    audio_timesteps = [1.0 - sigma for sigma in sigmas_audio[:-1]]
    video_step_t = torch.tensor(
        video_timesteps, dtype=torch.float32, device=device
    )
    audio_step_t = torch.tensor(
        audio_timesteps, dtype=torch.float32, device=device
    )
    timestep_plan = positive.prepare_timestep_plan(
        video_timesteps=video_timesteps,
        audio_timesteps=audio_timesteps,
        imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
        audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
    )
    adaln_plan_slots = model.prepare_adaln_plans(
        [entry[0] for entry in timestep_plan]
    )
    video_sigmas = torch.tensor(sigmas_video, dtype=torch.float32, device=device)
    audio_sigmas = torch.tensor(sigmas_audio, dtype=torch.float32, device=device)
    video_sigma_t = 1.0 - video_step_t
    audio_sigma_t = 1.0 - audio_step_t
    video_denoised = torch.empty_like(video_rows[video_target_slice])
    audio_denoised = torch.empty_like(audio_rows[audio_target_slice])
    er_video_steps = None
    er_audio_steps = None
    er_prev_video = None
    er_prev_video_d = None
    er_prev_audio = None
    er_prev_audio_d = None
    if active.sampler == ER_SDE_SAMPLER:
        er_video_steps = er_sde_step_scalars(sigmas_video)
        er_audio_steps = er_sde_step_scalars(sigmas_audio)
        er_prev_video = torch.zeros_like(video_denoised)
        er_prev_video_d = torch.zeros_like(video_denoised)
        er_prev_audio = torch.zeros_like(audio_denoised)
        er_prev_audio_d = torch.zeros_like(audio_denoised)

    for step in range(len(sigmas_video) - 1):
        step_context = (
            step_profiler(step) if step_profiler is not None else nullcontext()
        )
        with step_context:
            forward_kwargs = positive.forward_kwargs(
                video_rows=video_rows,
                audio_rows=audio_rows,
                step_timesteps=timestep_plan[step],
                adaln_slot=None
                if adaln_plan_slots is None
                else adaln_plan_slots[step],
            )
            if attn_metadata is not None:
                attn_metadata.current_timestep = step
            if model_forward is None and attn_metadata is not None:
                forward_context = set_forward_context(
                    current_timestep=step, attn_metadata=attn_metadata
                )
            else:
                forward_context = nullcontext()
            with forward_context, torch.inference_mode():
                if model_forward is None:
                    model_video, model_audio = model(**forward_kwargs)
                else:
                    model_video, model_audio = model_forward(
                        model, forward_kwargs, step
                    )
                model_video = model_video.float()
                model_audio = model_audio[audio_target_slice].float()
                video_target = video_rows[video_target_slice]
                audio_target = audio_rows[audio_target_slice]
                if er_video_steps is not None:
                    predict_denoised_rows(
                        video_target,
                        model_video,
                        sigma_t=video_sigma_t[step],
                        denoised_scratch=video_denoised,
                    )
                    predict_denoised_rows(
                        audio_target,
                        model_audio,
                        sigma_t=audio_sigma_t[step],
                        denoised_scratch=audio_denoised,
                    )
                    update_target_rows(
                        video_target,
                        video_denoised,
                        er_video_steps[step],
                        prev_denoised=er_prev_video,
                        prev_denoised_d=er_prev_video_d,
                        noise_generator=active.noise_generator,
                    )
                    update_target_rows(
                        audio_target,
                        audio_denoised,
                        er_audio_steps[step],
                        prev_denoised=er_prev_audio,
                        prev_denoised_d=er_prev_audio_d,
                        noise_generator=active.noise_generator,
                    )
                else:
                    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
                        _minimax_h3_update_target_rows_,
                    )

                    video_ratios = video_sigmas[1:] / video_sigmas[:-1]
                    audio_ratios = audio_sigmas[1:] / audio_sigmas[:-1]
                    _minimax_h3_update_target_rows_(
                        video_target,
                        model_video,
                        sigma_t=video_sigma_t[step],
                        sigma_curr=sigmas_video[step],
                        sigma_ratio=video_ratios[step],
                        one_minus_sigma_ratio=1.0 - video_ratios[step],
                        denoised_scratch=video_denoised,
                    )
                    _minimax_h3_update_target_rows_(
                        audio_target,
                        model_audio,
                        sigma_t=audio_sigma_t[step],
                        sigma_curr=sigmas_audio[step],
                        sigma_ratio=audio_ratios[step],
                        one_minus_sigma_ratio=1.0 - audio_ratios[step],
                        denoised_scratch=audio_denoised,
                    )
                if anchor_state is not None:
                    video_target[: anchor_state.shape[0]].copy_(anchor_state)
            if on_step is not None:
                on_step(step, video_rows, audio_rows)
    return video_rows, audio_rows
