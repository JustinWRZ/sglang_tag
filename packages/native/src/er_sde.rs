use pyo3::prelude::*;

use crate::request::invalid;

#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct ErSdeStepScalars {
    pub is_final: bool,
    pub stage1_x: f64,
    pub stage1_d: f64,
    pub stage2_x: Option<f64>,
    pub stage2_inv_denom: Option<f64>,
    pub stage3_x: Option<f64>,
    pub stage3_inv_denom: Option<f64>,
    pub noise_scale: f64,
}

fn noise_scaler(value: f32) -> f32 {
    value * ((value.powf(0.3).exp()) + 10.0)
}

#[pyfunction(signature = (sigmas, *, max_stage=3, num_integration_points=200.0))]
pub fn er_sde_step_scalars(
    sigmas: Vec<f64>,
    max_stage: usize,
    num_integration_points: f32,
) -> PyResult<Vec<ErSdeStepScalars>> {
    if sigmas.len() < 2 {
        return Err(invalid("ER-SDE sigma schedule needs at least two values"));
    }
    if !(num_integration_points.is_finite() && num_integration_points > 0.0) {
        return Err(invalid("ER-SDE integration point count must be positive"));
    }
    if sigmas
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
    {
        return Err(invalid("ER-SDE sigmas must be finite and in [0, 1]"));
    }

    let values = sigmas.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let er_lambdas = values
        .iter()
        .map(|sigma| {
            let half_log_snr = ((1.0 - sigma) / sigma).ln();
            (-half_log_snr).exp()
        })
        .collect::<Vec<_>>();
    let point_count = num_integration_points as usize;
    let points: Vec<f32> = (0..point_count).map(|index| index as f32).collect();
    let mut result = Vec::with_capacity(values.len().saturating_sub(1));

    for index in 0..values.len().saturating_sub(1) {
        let mut scalar = ErSdeStepScalars {
            is_final: false,
            stage1_x: 0.0,
            stage1_d: 0.0,
            stage2_x: None,
            stage2_inv_denom: None,
            stage3_x: None,
            stage3_inv_denom: None,
            noise_scale: 0.0,
        };
        if values[index + 1] == 0.0 {
            scalar.is_final = true;
            result.push(scalar);
            continue;
        }

        let lambda_s = er_lambdas[index];
        let lambda_t = er_lambdas[index + 1];
        let alpha_s = values[index] / lambda_s;
        let alpha_t = values[index + 1] / lambda_t;
        let r_alpha = alpha_t / alpha_s;
        let ratio = noise_scaler(lambda_t) / noise_scaler(lambda_s);
        scalar.stage1_x = (r_alpha * ratio) as f64;
        scalar.stage1_d = (alpha_t * (1.0 - ratio)) as f64;
        let variance = lambda_t * lambda_t - lambda_s * lambda_s * ratio * ratio;
        scalar.noise_scale = if variance.is_nan() || variance <= 0.0 {
            0.0
        } else {
            (alpha_t * variance.sqrt()) as f64
        };

        let stage = max_stage.min(index + 1);
        if stage >= 2 {
            let dt = lambda_t - lambda_s;
            let step_size = -dt / num_integration_points;
            let positions: Vec<f32> = points
                .iter()
                .map(|point| lambda_t + point * step_size)
                .collect();
            let scaled: Vec<f32> = positions.iter().map(|value| noise_scaler(*value)).collect();
            let integral = scaled.iter().map(|value| 1.0 / value).sum::<f32>() * step_size;
            scalar.stage2_x = Some((alpha_t * (dt + integral * noise_scaler(lambda_t))) as f64);
            scalar.stage2_inv_denom = Some((1.0 / (lambda_s - er_lambdas[index - 1])) as f64);
            if stage >= 3 {
                let upper_integral = positions
                    .iter()
                    .zip(scaled.iter())
                    .map(|(position, denominator)| (position - lambda_s) / denominator)
                    .sum::<f32>()
                    * step_size;
                scalar.stage3_x = Some(
                    (alpha_t * ((dt * dt) / 2.0 + upper_integral * noise_scaler(lambda_t))) as f64,
                );
                scalar.stage3_inv_denom =
                    Some((1.0 / ((lambda_s - er_lambdas[index - 2]) / 2.0)) as f64);
            }
        }
        result.push(scalar);
    }
    Ok(result)
}
