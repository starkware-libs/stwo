use std::iter::zip;

use num_traits::Zero;

use crate::core::backend::cpu::lookups::gkr::gen_eq_evals as cpu_gen_eq_evals;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::simd::m31::{LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::{
    correct_sum_as_poly_in_first_variable, EqEvals, GkrMultivariatePolyOracle, GkrOps, Layer,
};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::{Fraction, Reciprocal, UnivariatePoly};

impl GkrOps for SimdBackend {
    #[allow(clippy::uninit_vec)]
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        if y.len() < LOG_N_LANES as usize {
            return Mle::new(cpu_gen_eq_evals(y, v).into_iter().collect());
        }

        // Start DP with CPU backend to avoid dealing with instances smaller than a SIMD vector.
        let (y_rem, y_last_chunk) = y.split_last_chunk::<{ LOG_N_LANES as usize }>().unwrap();
        let initial = SecureColumn::from_iter(cpu_gen_eq_evals(y_last_chunk, v));
        assert_eq!(initial.len(), N_LANES);

        let packed_len = 1 << y_rem.len();
        let mut data = initial.data;

        data.reserve(packed_len - data.len());
        unsafe { data.set_len(packed_len) };

        for (i, &y_j) in y_rem.iter().rev().enumerate() {
            let packed_y_j = PackedSecureField::broadcast(y_j);

            let (lhs_evals, rhs_evals) = data.split_at_mut(1 << i);

            for (lhs, rhs) in zip(lhs_evals, rhs_evals) {
                // Equivalent to:
                // `rhs = eq(1, y_j) * lhs`,
                // `lhs = eq(0, y_j) * lhs`
                *rhs = *lhs * packed_y_j;
                *lhs -= *rhs;
            }
        }

        let length = packed_len * N_LANES;
        Mle::new(SecureColumn { data, length })
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        // Offload to CPU backend to avoid dealing with instances smaller than a SIMD vector.
        if layer.n_variables() as u32 <= LOG_N_LANES {
            return into_simd_layer(layer.to_cpu().next_layer().unwrap());
        }

        match layer {
            Layer::GrandProduct(col) => next_grand_product_layer(col),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => next_logup_generic_layer(numerators, denominators),
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => next_logup_multiplicities_layer(numerators, denominators),
            Layer::LogUpSingles { denominators } => next_logup_singles_layer(denominators),
        }
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        let n_variables = h.n_variables();
        let n_terms = 1 << n_variables.saturating_sub(1);
        let eq_evals = h.eq_evals.as_ref();
        // Vector used to generate evaluations of `eq(x, y)` for `x` in the boolean hypercube.
        let y = eq_evals.y();

        // Offload to CPU backend to avoid dealing with instances smaller than a SIMD vector.
        if n_terms < N_LANES {
            return h.to_cpu().sum_as_poly_in_first_variable(claim);
        }

        let n_packed_terms = n_terms / N_LANES;
        let packed_lambda = PackedSecureField::broadcast(h.lambda);

        let (mut eval_at_0, mut eval_at_2) = match &h.input_layer {
            Layer::GrandProduct(col) => eval_grand_product_sum(eq_evals, col, n_packed_terms),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => eval_logup_generic_sum(
                eq_evals,
                numerators,
                denominators,
                n_packed_terms,
                packed_lambda,
            ),
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => eval_logup_multiplicities_sum(
                eq_evals,
                numerators,
                denominators,
                n_packed_terms,
                packed_lambda,
            ),
            Layer::LogUpSingles { denominators } => {
                eval_logup_singles_sum(eq_evals, denominators, n_packed_terms, packed_lambda)
            }
        };

        eval_at_0 *= h.eq_fixed_var_correction;
        eval_at_2 *= h.eq_fixed_var_correction;
        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
    }
}

/// Generates the next GKR layer for Grand Product.
///
/// Assumption: `len(layer) > N_LANES`.
fn next_grand_product_layer(layer: &Mle<SimdBackend, SecureField>) -> Layer<SimdBackend> {
    assert!(layer.len() > N_LANES);
    let next_layer_len = layer.len() / 2;

    let data = layer
        .data
        .array_chunks()
        .map(|&[a, b]| {
            let (evens, odds) = a.deinterleave(b);
            evens * odds
        })
        .collect();

    Layer::GrandProduct(Mle::new(SecureColumn {
        data,
        length: next_layer_len,
    }))
}

/// Generates the next GKR layer for LogUp.
///
/// Assumption: `len(denominators) > N_LANES`.
fn next_logup_generic_layer(
    numerators: &Mle<SimdBackend, SecureField>,
    denominators: &Mle<SimdBackend, SecureField>,
) -> Layer<SimdBackend> {
    assert!(denominators.len() > N_LANES);
    assert_eq!(numerators.len(), denominators.len());

    let next_layer_len = denominators.len() / 2;
    let next_layer_packed_len = next_layer_len / N_LANES;

    let mut next_numerators = Vec::with_capacity(next_layer_packed_len);
    let mut next_denominators = Vec::with_capacity(next_layer_packed_len);

    for i in 0..next_layer_packed_len {
        let (n_even, n_odd) = numerators.data[i * 2].deinterleave(numerators.data[i * 2 + 1]);
        let (d_even, d_odd) = denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

        let Fraction {
            numerator,
            denominator,
        } = Fraction::new(n_even, d_even) + Fraction::new(n_odd, d_odd);

        next_numerators.push(numerator);
        next_denominators.push(denominator);
    }

    let next_numerators = SecureColumn {
        data: next_numerators,
        length: next_layer_len,
    };

    let next_denominators = SecureColumn {
        data: next_denominators,
        length: next_layer_len,
    };

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

/// Generates the next GKR layer for LogUp.
///
/// Assumption: `len(denominators) > N_LANES`.
// TODO(andrew): Code duplication of `next_logup_generic_layer`. Consider unifying these.
fn next_logup_multiplicities_layer(
    numerators: &Mle<SimdBackend, BaseField>,
    denominators: &Mle<SimdBackend, SecureField>,
) -> Layer<SimdBackend> {
    assert!(denominators.len() > N_LANES);
    assert_eq!(numerators.len(), denominators.len());

    let next_layer_len = denominators.len() / 2;
    let next_layer_packed_len = next_layer_len / N_LANES;

    let mut next_numerators = Vec::with_capacity(next_layer_packed_len);
    let mut next_denominators = Vec::with_capacity(next_layer_packed_len);

    for i in 0..next_layer_packed_len {
        let (n_even, n_odd) = numerators.data[i * 2].deinterleave(numerators.data[i * 2 + 1]);
        let (d_even, d_odd) = denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

        let Fraction {
            numerator,
            denominator,
        } = Fraction::new(n_even, d_even) + Fraction::new(n_odd, d_odd);

        next_numerators.push(numerator);
        next_denominators.push(denominator);
    }

    let next_numerators = SecureColumn {
        data: next_numerators,
        length: next_layer_len,
    };

    let next_denominators = SecureColumn {
        data: next_denominators,
        length: next_layer_len,
    };

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

/// Generates the next GKR layer for LogUp.
///
/// Assumption: `len(denominators) > N_LANES`.
fn next_logup_singles_layer(denominators: &Mle<SimdBackend, SecureField>) -> Layer<SimdBackend> {
    assert!(denominators.len() > N_LANES);

    let next_layer_len = denominators.len() / 2;
    let next_layer_packed_len = next_layer_len / N_LANES;

    let mut next_numerators = Vec::with_capacity(next_layer_packed_len);
    let mut next_denominators = Vec::with_capacity(next_layer_packed_len);

    for i in 0..next_layer_packed_len {
        let (d_even, d_odd) = denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

        let Fraction {
            numerator,
            denominator,
        } = Reciprocal::new(d_even) + Reciprocal::new(d_odd);

        next_numerators.push(numerator);
        next_denominators.push(denominator);
    }

    let next_numerators = SecureColumn {
        data: next_numerators,
        length: next_layer_len,
    };

    let next_denominators = SecureColumn {
        data: next_denominators,
        length: next_layer_len,
    };

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

/// Evaluates `sum_x eq(({0}^|r|, 0, x), y) * inp(r, t, x, 0) * inp(r, t, x, 1)` at `t=0` and `t=2`.
///
/// Output of the form: `(eval_at_0, eval_at_2)`.
fn eval_grand_product_sum(
    eq_evals: &EqEvals<SimdBackend>,
    col: &Mle<SimdBackend, SecureField>,
    n_packed_terms: usize,
) -> (SecureField, SecureField) {
    let mut packed_eval_at_0 = PackedSecureField::zero();
    let mut packed_eval_at_2 = PackedSecureField::zero();

    for i in 0..n_packed_terms {
        // Input polynomial at points `(r, {0, 1, 2}, bits(i), v, {0, 1})`
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let (inp_at_r0iv0, inp_at_r0iv1) = col.data[i * 2].deinterleave(col.data[i * 2 + 1]);
        let (inp_at_r1iv0, inp_at_r1iv1) =
            col.data[(n_packed_terms + i) * 2].deinterleave(col.data[(n_packed_terms + i) * 2 + 1]);
        // Note `inp(r, t, x) = eq(t, 0) * inp(r, 0, x) + eq(t, 1) * inp(r, 1, x)`
        //   => `inp(r, 2, x) = 2 * inp(r, 1, x) - inp(r, 0, x)`
        let inp_at_r2iv0 = inp_at_r1iv0.double() - inp_at_r0iv0;
        let inp_at_r2iv1 = inp_at_r1iv1.double() - inp_at_r0iv1;

        // Product polynomial `prod(x) = inp(x, 0) * inp(x, 1)` at points `(r, {0, 2}, bits(i), v)`.
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let prod_at_r2iv = inp_at_r2iv0 * inp_at_r2iv1;
        let prod_at_r0iv = inp_at_r0iv0 * inp_at_r0iv1;

        let eq_eval_at_0iv = eq_evals.data[i];
        packed_eval_at_0 += eq_eval_at_0iv * prod_at_r0iv;
        packed_eval_at_2 += eq_eval_at_0iv * prod_at_r2iv;
    }

    (
        packed_eval_at_0.pointwise_sum(),
        packed_eval_at_2.pointwise_sum(),
    )
}

fn eval_logup_generic_sum(
    eq_evals: &EqEvals<SimdBackend>,
    numerators: &Mle<SimdBackend, SecureField>,
    denominators: &Mle<SimdBackend, SecureField>,
    n_packed_terms: usize,
    packed_lambda: PackedSecureField,
) -> (SecureField, SecureField) {
    let mut packed_eval_at_0 = PackedSecureField::zero();
    let mut packed_eval_at_2 = PackedSecureField::zero();

    let inp_numer = &numerators.data;
    let inp_denom = &denominators.data;

    for i in 0..n_packed_terms {
        // Input polynomials at points `(r, {0, 1, 2}, bits(i), v, {0, 1})`
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let (inp_numer_at_r0iv0, inp_numer_at_r0iv1) =
            inp_numer[i * 2].deinterleave(inp_numer[i * 2 + 1]);
        let (inp_denom_at_r0iv0, inp_denom_at_r0iv1) =
            inp_denom[i * 2].deinterleave(inp_denom[i * 2 + 1]);
        let (inp_numer_at_r1iv0, inp_numer_at_r1iv1) = inp_numer[(n_packed_terms + i) * 2]
            .deinterleave(inp_numer[(n_packed_terms + i) * 2 + 1]);
        let (inp_denom_at_r1iv0, inp_denom_at_r1iv1) = inp_denom[(n_packed_terms + i) * 2]
            .deinterleave(inp_denom[(n_packed_terms + i) * 2 + 1]);
        // Note `inp_denom(r, t, x) = eq(t, 0) * inp_denom(r, 0, x) + eq(t, 1) * inp_denom(r, 1, x)`
        //   => `inp_denom(r, 2, x) = 2 * inp_denom(r, 1, x) - inp_denom(r, 0, x)`
        let inp_numer_at_r2iv0 = inp_numer_at_r1iv0.double() - inp_numer_at_r0iv0;
        let inp_numer_at_r2iv1 = inp_numer_at_r1iv1.double() - inp_numer_at_r0iv1;
        let inp_denom_at_r2iv0 = inp_denom_at_r1iv0.double() - inp_denom_at_r0iv0;
        let inp_denom_at_r2iv1 = inp_denom_at_r1iv1.double() - inp_denom_at_r0iv1;

        // Fraction addition polynomials:
        // - `numer(x) = inp_numer(x, 0) * inp_denom(x, 1) + inp_numer(x, 1) * inp_denom(x, 0)`
        // - `denom(x) = inp_denom(x, 0) * inp_denom(x, 1)`.
        // at points `(r, {0, 2}, bits(i), v)` for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let Fraction {
            numerator: numer_at_r0iv,
            denominator: denom_at_r0iv,
        } = Fraction::new(inp_numer_at_r0iv0, inp_denom_at_r0iv0)
            + Fraction::new(inp_numer_at_r0iv1, inp_denom_at_r0iv1);
        let Fraction {
            numerator: numer_at_r2iv,
            denominator: denom_at_r2iv,
        } = Fraction::new(inp_numer_at_r2iv0, inp_denom_at_r2iv0)
            + Fraction::new(inp_numer_at_r2iv1, inp_denom_at_r2iv1);

        let eq_eval_at_0iv = eq_evals.data[i];
        packed_eval_at_0 += eq_eval_at_0iv * (numer_at_r0iv + packed_lambda * denom_at_r0iv);
        packed_eval_at_2 += eq_eval_at_0iv * (numer_at_r2iv + packed_lambda * denom_at_r2iv);
    }

    (
        packed_eval_at_0.pointwise_sum(),
        packed_eval_at_2.pointwise_sum(),
    )
}

// TODO(andrew): Code duplication of `eval_logup_generic_sum`. Consider unifying these.
fn eval_logup_multiplicities_sum(
    eq_evals: &EqEvals<SimdBackend>,
    numerators: &Mle<SimdBackend, BaseField>,
    denominators: &Mle<SimdBackend, SecureField>,
    n_packed_terms: usize,
    packed_lambda: PackedSecureField,
) -> (SecureField, SecureField) {
    let mut packed_eval_at_0 = PackedSecureField::zero();
    let mut packed_eval_at_2 = PackedSecureField::zero();

    let inp_numer = &numerators.data;
    let inp_denom = &denominators.data;

    for i in 0..n_packed_terms {
        // Input polynomials at points `(r, {0, 1, 2}, bits(i), v, {0, 1})`
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let (inp_numer_at_r0iv0, inp_numer_at_r0iv1) =
            inp_numer[i * 2].deinterleave(inp_numer[i * 2 + 1]);
        let (inp_denom_at_r0iv0, inp_denom_at_r0iv1) =
            inp_denom[i * 2].deinterleave(inp_denom[i * 2 + 1]);
        let (inp_numer_at_r1iv0, inp_numer_at_r1iv1) = inp_numer[(n_packed_terms + i) * 2]
            .deinterleave(inp_numer[(n_packed_terms + i) * 2 + 1]);
        let (inp_denom_at_r1iv0, inp_denom_at_r1iv1) = inp_denom[(n_packed_terms + i) * 2]
            .deinterleave(inp_denom[(n_packed_terms + i) * 2 + 1]);
        // Note `inp_denom(r, t, x) = eq(t, 0) * inp_denom(r, 0, x) + eq(t, 1) * inp_denom(r, 1, x)`
        //   => `inp_denom(r, 2, x) = 2 * inp_denom(r, 1, x) - inp_denom(r, 0, x)`
        let inp_numer_at_r2iv0 = inp_numer_at_r1iv0.double() - inp_numer_at_r0iv0;
        let inp_numer_at_r2iv1 = inp_numer_at_r1iv1.double() - inp_numer_at_r0iv1;
        let inp_denom_at_r2iv0 = inp_denom_at_r1iv0.double() - inp_denom_at_r0iv0;
        let inp_denom_at_r2iv1 = inp_denom_at_r1iv1.double() - inp_denom_at_r0iv1;

        // Fraction addition polynomials:
        // - `numer(x) = inp_numer(x, 0) * inp_denom(x, 1) + inp_numer(x, 1) * inp_denom(x, 0)`
        // - `denom(x) = inp_denom(x, 0) * inp_denom(x, 1)`.
        // at points `(r, {0, 2}, bits(i), v)` for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let Fraction {
            numerator: numer_at_r0iv,
            denominator: denom_at_r0iv,
        } = Fraction::new(inp_numer_at_r0iv0, inp_denom_at_r0iv0)
            + Fraction::new(inp_numer_at_r0iv1, inp_denom_at_r0iv1);
        let Fraction {
            numerator: numer_at_r2iv,
            denominator: denom_at_r2iv,
        } = Fraction::new(inp_numer_at_r2iv0, inp_denom_at_r2iv0)
            + Fraction::new(inp_numer_at_r2iv1, inp_denom_at_r2iv1);

        let eq_eval_at_0iv = eq_evals.data[i];
        packed_eval_at_0 += eq_eval_at_0iv * (numer_at_r0iv + packed_lambda * denom_at_r0iv);
        packed_eval_at_2 += eq_eval_at_0iv * (numer_at_r2iv + packed_lambda * denom_at_r2iv);
    }

    (
        packed_eval_at_0.pointwise_sum(),
        packed_eval_at_2.pointwise_sum(),
    )
}

/// Evaluates `sum_x eq(({0}^|r|, 0, x), y) * (inp_denom(r, t, x, 1) + inp_denom(r, t, x, 0) +
/// lambda * inp_denom(r, t, x, 0) * inp_denom(r, t, x, 1))` at `t=0` and `t=2`.
///
/// Output of the form: `(eval_at_0, eval_at_2)`.
fn eval_logup_singles_sum(
    eq_evals: &EqEvals<SimdBackend>,
    denominators: &Mle<SimdBackend, SecureField>,
    n_packed_terms: usize,
    packed_lambda: PackedSecureField,
) -> (SecureField, SecureField) {
    let mut packed_eval_at_0 = PackedSecureField::zero();
    let mut packed_eval_at_2 = PackedSecureField::zero();

    let inp_denom = &denominators.data;

    for i in 0..n_packed_terms {
        // Input polynomial at points `(r, {0, 1, 2}, bits(i), v, {0, 1})`
        // for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let (inp_denom_at_r0iv0, inp_denom_at_r0iv1) =
            inp_denom[i * 2].deinterleave(inp_denom[i * 2 + 1]);
        let (inp_denom_at_r1iv0, inp_denom_at_r1iv1) = inp_denom[(n_packed_terms + i) * 2]
            .deinterleave(inp_denom[(n_packed_terms + i) * 2 + 1]);
        // Note `inp_denom(r, t, x) = eq(t, 0) * inp_denom(r, 0, x) + eq(t, 1) * inp_denom(r, 1, x)`
        //   => `inp_denom(r, 2, x) = 2 * inp_denom(r, 1, x) - inp_denom(r, 0, x)`
        let inp_denom_at_r2iv0 = inp_denom_at_r1iv0.double() - inp_denom_at_r0iv0;
        let inp_denom_at_r2iv1 = inp_denom_at_r1iv1.double() - inp_denom_at_r0iv1;

        // Fraction addition polynomials:
        // - `numer(x) = inp_denom(x, 1) + inp_denom(x, 0)`
        // - `denom(x) = inp_denom(x, 0) * inp_denom(x, 1)`.
        // at points `(r, {0, 2}, bits(i), v)` for all `v` in `{0, 1}^LOG_N_SIMD_LANES`.
        let Fraction {
            numerator: numer_at_r0iv,
            denominator: denom_at_r0iv,
        } = Reciprocal::new(inp_denom_at_r0iv0) + Reciprocal::new(inp_denom_at_r0iv1);
        let Fraction {
            numerator: numer_at_r2iv,
            denominator: denom_at_r2iv,
        } = Reciprocal::new(inp_denom_at_r2iv0) + Reciprocal::new(inp_denom_at_r2iv1);

        let eq_eval_at_0iv = eq_evals.data[i];
        packed_eval_at_0 += eq_eval_at_0iv * (numer_at_r0iv + packed_lambda * denom_at_r0iv);
        packed_eval_at_2 += eq_eval_at_0iv * (numer_at_r2iv + packed_lambda * denom_at_r2iv);
    }

    (
        packed_eval_at_0.pointwise_sum(),
        packed_eval_at_2.pointwise_sum(),
    )
}

fn into_simd_layer(cpu_layer: Layer<CpuBackend>) -> Layer<SimdBackend> {
    match cpu_layer {
        Layer::GrandProduct(mle) => {
            Layer::GrandProduct(Mle::new(mle.into_evals().into_iter().collect()))
        }
        Layer::LogUpGeneric {
            numerators,
            denominators,
        } => Layer::LogUpGeneric {
            numerators: Mle::new(numerators.into_evals().into_iter().collect()),
            denominators: Mle::new(denominators.into_evals().into_iter().collect()),
        },
        Layer::LogUpMultiplicities {
            numerators,
            denominators,
        } => Layer::LogUpMultiplicities {
            numerators: Mle::new(numerators.into_evals().into_iter().collect()),
            denominators: Mle::new(denominators.into_evals().into_iter().collect()),
        },
        Layer::LogUpSingles { denominators } => Layer::LogUpSingles {
            denominators: Mle::new(denominators.into_evals().into_iter().collect()),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::channel::Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::{prove_batch, GkrOps, Layer};
    use crate::core::lookups::gkr_verifier::{partially_verify_batch, Gate, GkrArtifact, GkrError};
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::utils::Fraction;
    use crate::core::test_utils::test_channel;

    #[test]
    fn gen_eq_evals_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5, 6, 1, 1, 9].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_simd = SimdBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_simd.to_cpu(), *eq_evals_cpu);
    }

    #[test]
    fn gen_eq_evals_with_small_assignment_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_simd = SimdBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_simd.to_cpu(), *eq_evals_cpu);
    }

    #[test]
    fn grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product();
        let col = Mle::<SimdBackend, SecureField>::new(values.into_iter().collect());
        let input_layer = Layer::GrandProduct(col.clone());
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::GrandProduct], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_instance, [vec![product]]);
        assert_eq!(
            claims_to_verify_by_instance,
            [vec![col.eval_at_point(&ood_point)]]
        );
        Ok(())
    }

    #[test]
    fn logup_with_generic_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerators, &denominators)
            .map(|(&n, &d)| Fraction::new(n, d))
            .sum::<Fraction<SecureField, SecureField>>();
        let numerators = Mle::<SimdBackend, SecureField>::new(numerators.into_iter().collect());
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpGeneric {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [
                numerators.eval_at_point(&ood_point),
                denominators.eval_at_point(&ood_point)
            ]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }

    #[test]
    fn logup_with_multiplicities_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerators = (0..N).map(|_| rng.gen()).collect::<Vec<BaseField>>();
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerators, &denominators)
            .map(|(&n, &d)| Fraction::new(n.into(), d))
            .sum::<Fraction<SecureField, SecureField>>();
        let numerators = Mle::<SimdBackend, BaseField>::new(numerators.into_iter().collect());
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpMultiplicities {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [
                numerators.eval_at_point(&ood_point),
                denominators.eval_at_point(&ood_point)
            ]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }

    #[test]
    fn logup_with_singles_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = denominators
            .iter()
            .map(|&d| Fraction::new(SecureField::one(), d))
            .sum::<Fraction<SecureField, SecureField>>();
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpSingles {
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [SecureField::one(), denominators.eval_at_point(&ood_point)]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }
}
