use std::ops::{Add, Index};

use num_traits::{One, Zero};

use crate::core::backend::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, Field};
use crate::core::lookups::gkr_prover::{
    correct_sum_as_poly_in_first_variable, EqEvals, GkrMultivariatePolyOracle, GkrOps, Layer,
};
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::{Fraction, UnivariatePoly};

impl GkrOps for CpuBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        Mle::new(gen_eq_evals(y, v))
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        match layer {
            Layer::GrandProduct(layer) => next_grand_product_layer(layer),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => next_logup_layer(MleExpr::Mle(numerators), denominators),
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => next_logup_layer(MleExpr::Mle(numerators), denominators),
            Layer::LogUpSingles { denominators } => {
                next_logup_layer(MleExpr::Constant(BaseField::one()), denominators)
            }
        }
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        let k = h.n_variables();
        let n_terms = 1 << (k - 1);
        let eq_evals = h.eq_evals;
        let y = eq_evals.y();
        let lambda = h.lambda;
        let input_layer = &h.input_layer;

        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();

        match input_layer {
            Layer::GrandProduct(col) => {
                process_grand_product_sum(&mut eval_at_0, &mut eval_at_2, eq_evals, col, n_terms)
            }
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => process_logup_sum(
                &mut eval_at_0,
                &mut eval_at_2,
                eq_evals,
                numerators,
                denominators,
                n_terms,
                lambda,
            ),
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => process_logup_sum(
                &mut eval_at_0,
                &mut eval_at_2,
                eq_evals,
                numerators,
                denominators,
                n_terms,
                lambda,
            ),
            Layer::LogUpSingles { denominators } => process_logup_singles_sum(
                &mut eval_at_0,
                &mut eval_at_2,
                eq_evals,
                denominators,
                n_terms,
                lambda,
            ),
        }

        eval_at_0 *= h.eq_fixed_var_correction;
        eval_at_2 *= h.eq_fixed_var_correction;

        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, k)
    }
}

fn process_grand_product_sum(
    eval_at_0: &mut SecureField,
    eval_at_2: &mut SecureField,
    eq_evals: &EqEvals<CpuBackend>,
    col: &Mle<CpuBackend, SecureField>,
    n_terms: usize,
) {
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_terms {
        // Let `p` be the multilinear polynomial representing `col`.
        let p0x0 /* = p(0, x, 0) */ = col[i * 2];
        let p0x1 /* = p(0, x, 1) */ = col[i * 2 + 1];

        // We obtain `p(2, x)` for some `x` in the boolean
        // hypercube using `p(0, x)` and `p(1, x)`:
        //
        // ```text
        // p(t, x) = eq(t, 0) * p(0, x) + eq(t, 1) * p(1, x)
        //         = (1 - t) * p(0, x) + t * p(1, x)
        //
        // p(2, x) = 2 * p(1, x) - p(0, x)
        // ```
        let p1x0 /* = p(1, x, 0) */ = col[(n_terms + i) * 2];
        let p1x1 /* = p(1, x, 1) */ = col[(n_terms + i) * 2 + 1];
        let p2x0 /* = p(2, x, 0) */ = p1x0.double() - p0x0;
        let p2x1 /* = p(2, x, 1) */ = p1x1.double() - p0x1;

        let product2 = p2x0 * p2x1;
        let product0 = p0x0 * p0x1;

        let eq_eval = eq_evals[i];
        *eval_at_0 += eq_eval * product0;
        *eval_at_2 += eq_eval * product2;
    }
}

fn process_logup_sum<F: Field>(
    eval_at_0: &mut SecureField,
    eval_at_2: &mut SecureField,
    eq_evals: &EqEvals<CpuBackend>,
    numerators: &Mle<CpuBackend, F>,
    denominators: &Mle<CpuBackend, SecureField>,
    n_terms: usize,
    lambda: SecureField,
) where
    SecureField: ExtensionOf<F> + Field,
{
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_terms {
        // Let `p` be the multilinear polynomial representing `numerators`.
        // Let `q` be the multilinear polynomial representing `denominators`.
        let p0x0 /* = p(0, x, 0) */ = numerators[i * 2];
        let q0x0 /* = q(0, x, 0) */ = denominators[i * 2];
        let p0x1 /* = p(0, x, 1) */ = numerators[i * 2 + 1];
        let q0x1 /* = q(0, x, 1) */ = denominators[i * 2 + 1];

        // We obtain `p(2, x)` for some `x` in the boolean
        // hypercube using `p(0, x)` and `p(1, x)`:
        //
        // ```text
        // p(t, x) = eq(t, 0) * p(0, x) + eq(t, 1) * p(1, x)
        //         = (1 - t) * p(0, x) + t * p(1, x)
        //
        // p(2, x) = 2 * p(1, x) - p(0, x)
        // ```
        let p1x0 /* = p(1, x, 0) */ = numerators[(n_terms + i) * 2];
        let q1x0 /* = q(1, x, 0) */ = denominators[(n_terms + i) * 2];
        let p1x1 /* = p(1, x, 1) */ = numerators[(n_terms + i) * 2 + 1];
        let q1x1 /* = q(1, x, 1) */ = denominators[(n_terms + i) * 2 + 1];
        let p2x0 /* = p(2, x, 0) */ = p1x0.double() - p0x0;
        let q2x0 /* = q(2, x, 0) */ = q1x0.double() - q0x0;
        let p2x1 /* = p(2, x, 1) */ = p1x1.double() - p0x1;
        let q2x1 /* = q(2, x, 1) */ = q1x1.double() - q0x1;

        let res0 = Fraction::new(p0x0, q0x0) + Fraction::new(p0x1, q0x1);
        let res2 = Fraction::new(p2x0, q2x0) + Fraction::new(p2x1, q2x1);

        let eq_eval = eq_evals[i];
        *eval_at_0 += eq_eval * (res0.numerator + lambda * res0.denominator);
        *eval_at_2 += eq_eval * (res2.numerator + lambda * res2.denominator);
    }
}

fn process_logup_singles_sum(
    eval_at_0: &mut SecureField,
    eval_at_2: &mut SecureField,
    eq_evals: &EqEvals<CpuBackend>,
    denominators: &Mle<CpuBackend, SecureField>,
    n_terms: usize,
    lambda: SecureField,
) {
    /// Represents the fraction `1 / x`
    struct Reciprocal {
        x: SecureField,
    }

    impl Add for Reciprocal {
        type Output = Fraction<SecureField>;

        fn add(self, rhs: Self) -> Fraction<SecureField> {
            // `1/a + 1/b = (a + b)/(a * b)`
            Fraction {
                numerator: self.x + rhs.x,
                denominator: self.x * rhs.x,
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..n_terms {
        // Let `q` be the multilinear polynomial representing `denominators`.
        let q0x0 /* = q(0, x, 0) */ = denominators[i * 2];
        let q0x1 /* = q(0, x, 1) */ = denominators[i * 2 + 1];

        // We obtain `q(2, x)` for some `x` in the boolean
        // hypercube using `q(0, x)` and `q(1, x)`:
        //
        // ```text
        // q(t, x) = eq(t, 0) * q(0, x) + eq(t, 1) * q(1, x)
        //         = (1 - t) * q(0, x) + t * q(1, x)
        //
        // q(2, x) = 2 * q(1, x) - q(0, x)
        // ```
        let q1x0 /* = q(1, x, 0) */ = denominators[(n_terms + i) * 2];
        let q1x1 /* = q(1, x, 1) */ = denominators[(n_terms + i) * 2 + 1];
        let q2x0 /* = q(2, x, 0) */ = q1x0.double() - q0x0;
        let q2x1 /* = q(2, x, 1) */ = q1x1.double() - q0x1;

        let res0 = Reciprocal { x: q0x0 } + Reciprocal { x: q0x1 };
        let res2 = Reciprocal { x: q2x0 } + Reciprocal { x: q2x1 };

        let eq_eval = eq_evals[i];
        *eval_at_0 += eq_eval * (res0.numerator + lambda * res0.denominator);
        *eval_at_2 += eq_eval * (res2.numerator + lambda * res2.denominator);
    }
}

/// Returns evaluations `eq(x, y) * v` for all `x` in `{0, 1}^n`.
///
/// Evaluations are returned in bit-reversed order.
fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Vec<SecureField> {
    let mut evals = Vec::with_capacity(1 << y.len());
    evals.push(v);

    for &y_i in y.iter().rev() {
        for j in 0..evals.len() {
            // `lhs[j] = eq(0, y_i) * c[i]`
            // `rhs[j] = eq(1, y_i) * c[i]`
            let tmp = evals[j] * y_i;
            evals.push(tmp);
            evals[j] -= tmp;
        }
    }

    evals
}

fn next_grand_product_layer(layer: &Mle<CpuBackend, SecureField>) -> Layer<CpuBackend> {
    let res = layer.array_chunks().map(|&[a, b]| a * b).collect();
    Layer::GrandProduct(Mle::new(res))
}

fn next_logup_layer<F>(
    numerators: MleExpr<'_, F>,
    denominators: &Mle<CpuBackend, SecureField>,
) -> Layer<CpuBackend>
where
    F: Field,
    SecureField: ExtensionOf<F>,
    CpuBackend: MleOps<F>,
{
    let half_n = 1 << (denominators.n_variables() - 1);
    let mut next_numerators = Vec::with_capacity(half_n);
    let mut next_denominators = Vec::with_capacity(half_n);

    for i in 0..half_n {
        let a = Fraction::new(numerators[i * 2], denominators[i * 2]);
        let b = Fraction::new(numerators[i * 2 + 1], denominators[i * 2 + 1]);
        let res = a + b;
        next_numerators.push(res.numerator);
        next_denominators.push(res.denominator);
    }

    Layer::LogUpGeneric {
        numerators: Mle::new(next_numerators),
        denominators: Mle::new(next_denominators),
    }
}

enum MleExpr<'a, F: Field> {
    Constant(F),
    Mle(&'a Mle<CpuBackend, F>),
}

impl<'a, F: Field> Index<usize> for MleExpr<'a, F> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        match self {
            Self::Constant(v) => v,
            Self::Mle(mle) => &mle[index],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::{One, Zero};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::CpuBackend;
    use crate::core::channel::Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::{prove_batch, GkrOps, Layer};
    use crate::core::lookups::gkr_verifier::{partially_verify_batch, Gate, GkrArtifact, GkrError};
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::utils::{eq, Fraction};
    use crate::core::test_utils::test_channel;

    #[test]
    fn gen_eq_evals() {
        let zero = SecureField::zero();
        let one = SecureField::one();
        let y = [7, 3].map(|v| BaseField::from(v).into());

        let eq_evals = CpuBackend::gen_eq_evals(&y, one);

        assert_eq!(
            **eq_evals,
            [
                eq(&[zero, zero], &y),
                eq(&[zero, one], &y),
                eq(&[one, zero], &y),
                eq(&[one, one], &y),
            ]
        );
    }

    #[test]
    fn grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product::<SecureField>();
        let col = Mle::<CpuBackend, SecureField>::new(values);
        let top_layer = Layer::GrandProduct(col.clone());
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer]);

        let GkrArtifact {
            ood_point: r,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::GrandProduct], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_instance, [vec![product]]);
        assert_eq!(claims_to_verify_by_instance, [vec![col.eval_at_point(&r)]]);
        Ok(())
    }

    #[test]
    fn logup_with_generic_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerator_values = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let denominator_values = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerator_values, &denominator_values)
            .map(|(&n, &d)| Fraction::new(n, d))
            .sum::<Fraction<SecureField>>();
        let numerators = Mle::<CpuBackend, SecureField>::new(numerator_values);
        let denominators = Mle::<CpuBackend, SecureField>::new(denominator_values);
        let top_layer = Layer::LogUpGeneric {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer]);

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
        const N: usize = 1 << 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let denominator_values = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = denominator_values
            .iter()
            .map(|&d| Fraction::new(SecureField::one(), d))
            .sum::<Fraction<SecureField>>();
        let denominators = Mle::<CpuBackend, SecureField>::new(denominator_values);
        let top_layer = Layer::LogUpSingles {
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer]);

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

    #[test]
    fn logup_with_multiplicities_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerator_values = (0..N).map(|_| rng.gen()).collect::<Vec<BaseField>>();
        let denominator_values = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerator_values, &denominator_values)
            .map(|(&n, &d)| Fraction::new(n.into(), d))
            .sum::<Fraction<SecureField>>();
        let numerators = Mle::<CpuBackend, BaseField>::new(numerator_values);
        let denominators = Mle::<CpuBackend, SecureField>::new(denominator_values);
        let top_layer = Layer::LogUpMultiplicities {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer]);

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
}
