from __future__ import annotations

from typing import Any

from .compat_constants import H3_COMPAT_EXPORT_FIELD
from .state_export import export_state


def decoding_forward_before(self: Any, batch: Any, server_args: Any):
    sampling_params = getattr(batch, "sampling_params", None)
    if (
        batch.extra.get(H3_COMPAT_EXPORT_FIELD) is not True
        and (getattr(sampling_params, "diffusers_kwargs", None) or {}).get(
            H3_COMPAT_EXPORT_FIELD
        )
        is not True
    ):
        return None
    try:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_replica_group,
            model_parallel_is_initialized,
        )

        owner = True
        if model_parallel_is_initialized():
            owner = get_replica_group().rank_in_group == 0
    except ImportError:
        owner = True
    if owner:
        manifest = export_state(batch, server_args)
        if manifest is not None:
            batch.extra["h3_compat:state_manifest"] = manifest
    return None


def canonical_audio_after(result: Any, *args: Any, **kwargs: Any):
    import torch

    if not isinstance(result, torch.Tensor) or result.numel() == 0:
        return result
    scale = result.std() * 5.0
    if scale > 1.0:
        return result / scale
    return result
