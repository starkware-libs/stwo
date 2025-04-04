use std::iter::zip;

use num_traits::Zero;

use crate::core::backend::cpu::lookups::gkr::gen_eq_evals as cpu_gen_eq_evals;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::simd::m31::{LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::wgpu::WgpuBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::{
    correct_sum_as_poly_in_first_variable, EqEvals, GkrMultivariatePolyOracle, GkrOps, Layer,
};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::{Fraction, Reciprocal, UnivariatePoly};

impl GkrOps for WgpuBackend {
    #[allow(clippy::uninit_vec)]
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let simd_mle = SimdBackend::gen_eq_evals(y, v);
        simd_mle.to_wgpu()
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
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
        todo!()
    }
}

/// Generates the next GKR layer for Grand Product.
///
/// Assumption: `len(layer) > N_LANES`.
fn next_grand_product_layer(layer: &Mle<WgpuBackend, SecureField>) -> Layer<WgpuBackend> {
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
    numerators: &Mle<WgpuBackend, SecureField>,
    denominators: &Mle<WgpuBackend, SecureField>,
) -> Layer<WgpuBackend> {
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
    numerators: &Mle<WgpuBackend, BaseField>,
    denominators: &Mle<WgpuBackend, SecureField>,
) -> Layer<WgpuBackend> {
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
fn next_logup_singles_layer(denominators: &Mle<WgpuBackend, SecureField>) -> Layer<WgpuBackend> {
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
    eq_evals: &EqEvals<WgpuBackend>,
    col: &Mle<WgpuBackend, SecureField>,
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
    eq_evals: &EqEvals<WgpuBackend>,
    numerators: &Mle<WgpuBackend, SecureField>,
    denominators: &Mle<WgpuBackend, SecureField>,
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
    eq_evals: &EqEvals<WgpuBackend>,
    numerators: &Mle<WgpuBackend, BaseField>,
    denominators: &Mle<WgpuBackend, SecureField>,
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
    eq_evals: &EqEvals<WgpuBackend>,
    denominators: &Mle<WgpuBackend, SecureField>,
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
