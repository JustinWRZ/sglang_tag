from __future__ import annotations

import torch

from .native import ErSdeStepScalars, er_sde_step_scalars

__all__ = [
    "ErSdeStepScalars",
    "er_sde_step_scalars",
    "predict_denoised_rows",
    "update_target_rows",
]


def predict_denoised_rows(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_t: torch.Tensor,
    denoised_scratch: torch.Tensor,
) -> None:
    torch.mul(sigma_t, velocity, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)


def update_target_rows(
    state: torch.Tensor,
    denoised: torch.Tensor,
    scalars: ErSdeStepScalars,
    *,
    prev_denoised: torch.Tensor,
    prev_denoised_d: torch.Tensor,
    noise_generator: torch.Generator,
) -> None:
    if scalars.is_final:
        state.copy_(denoised)
        return
    state.mul_(scalars.stage1_x).add_(denoised, alpha=scalars.stage1_d)
    if scalars.stage2_x is not None:
        denoised_d = (denoised - prev_denoised).mul_(
            scalars.stage2_inv_denom
        )
        state.add_(denoised_d, alpha=scalars.stage2_x)
        if scalars.stage3_x is not None:
            denoised_u = (denoised_d - prev_denoised_d).mul_(
                scalars.stage3_inv_denom
            )
            state.add_(denoised_u, alpha=scalars.stage3_x)
        prev_denoised_d.copy_(denoised_d)
    if scalars.noise_scale > 0.0:
        noise = torch.randn(
            state.shape,
            generator=noise_generator,
            dtype=state.dtype,
            device=state.device,
        )
        state.add_(noise, alpha=scalars.noise_scale)
    prev_denoised.copy_(denoised)
