from __future__ import annotations

from typing import Any

from .compat_constants import (
    H3_COMPAT_CONTINUATION_FIELD,
    H3_COMPAT_EXPORT_FIELD,
    H3_COMPAT_SIGMAS_FIELD,
)
from .flags import ER_SDE_SAMPLER, EULER_SAMPLER, SAMPLER_EXTRA_KEY
from .native import prepare_sigmas


def _compat_options(batch: Any) -> dict[str, Any]:
    sampling = getattr(batch, "sampling_params", None)
    options = getattr(sampling, "diffusers_kwargs", None) or {}
    return options if isinstance(options, dict) else {}


def _num_steps(batch: Any) -> int:
    value = getattr(batch, "num_inference_steps", None)
    if value is None:
        value = getattr(batch.sampling_params, "num_inference_steps", None)
    return int(value or 50)


def install_schedules_before(*args: Any, **kwargs: Any):
    batch = args[0] if args else kwargs.get("batch")
    if batch is None:
        return None
    options = _compat_options(batch)
    sampler = options.get("minimax_h3_sampler")
    if sampler is not None and sampler not in (EULER_SAMPLER, ER_SDE_SAMPLER):
        raise ValueError(
            f"minimax_h3_sampler must be euler or er_sde, got {sampler!r}"
        )
    if sampler is not None:
        batch.extra[SAMPLER_EXTRA_KEY] = sampler

    schedules = options.get("minimax_h3_sigmas")
    if schedules is None and sampler != ER_SDE_SAMPLER:
        return None
    plan = args[1] if len(args) > 1 else kwargs.get("plan")
    if plan is None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
    if plan is None:
        return None

    video_values = None
    audio_values = None
    if schedules is not None:
        if not isinstance(schedules, dict) or "video" not in schedules:
            raise ValueError(
                "minimax_h3_sigmas must be an object with a video sigma list"
            )
        video_values = [float(value) for value in schedules["video"]]
        audio_raw = schedules.get("audio")
        audio_values = (
            None
            if audio_raw is None
            else [float(value) for value in audio_raw]
        )

    video, audio = prepare_sigmas(
        video=video_values,
        audio=audio_values,
        sampler=str(sampler or EULER_SAMPLER),
        num_steps=_num_steps(batch),
        flow_shift=float(getattr(plan, "flow_shift", 12.0) or 12.0),
        audio_flow_shift=float(
            getattr(plan, "audio_flow_shift", 3.0) or 3.0
        ),
    )

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_SIGMAS_EXTRA_KEY,
    )

    batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = {
        "video": video,
        "audio": audio,
    }
    if options.get(H3_COMPAT_EXPORT_FIELD) is not None:
        batch.extra[H3_COMPAT_EXPORT_FIELD] = bool(
            options[H3_COMPAT_EXPORT_FIELD]
        )
    if options.get(H3_COMPAT_CONTINUATION_FIELD) is not None:
        batch.extra[H3_COMPAT_CONTINUATION_FIELD] = options[
            H3_COMPAT_CONTINUATION_FIELD
        ]
    return args, kwargs
