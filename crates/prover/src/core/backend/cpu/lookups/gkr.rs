use std::ops::Index;

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
use crate::core::lookups::utils::{Fraction, Reciprocal, UnivariatePoly};

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
        let n_variables = h.n_variables();
        assert!(!n_variables.is_zero());
        let n_terms = 1 << (n_variables - 1);
        let eq_evals = h.eq_evals.as_ref();
        // Vector used to generate evaluations of `eq(x, y)` for `x` in the boolean hypercube.
        let y = eq_evals.y();
        let lambda = h.lambda;

        let (mut eval_at_0, mut eval_at_2) = match &h.input_layer {
            Layer::GrandProduct(col) => eval_grand_product_sum(eq_evals, col, n_terms),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => eval_logup_sum(eq_evals, numerators, denominators, n_terms, lambda),
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => eval_logup_sum(eq_evals, numerators, denominators, n_terms, lambda),
            Layer::LogUpSingles { denominators } => {
                eval_logup_singles_sum(eq_evals, denominators, n_terms, lambda)
            }
        };

        eval_at_0 *= h.eq_fixed_var_correction;
        eval_at_2 *= h.eq_fixed_var_correction;
        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
    }
}

/// Evaluates `sum_x eq(({0}^|r|, 0, x), y) * inp(r, t, x, 0) * inp(r, t, x, 1)` at `t=0` and `t=2`.
///
/// Output of the form: `(eval_at_0, eval_at_2)`.
fn eval_grand_product_sum(
    eq_evals: &EqEvals<CpuBackend>,
    input_layer: &Mle<CpuBackend, SecureField>,
    n_terms: usize,
) -> (SecureField, SecureField) {
    let mut eval_at_0 = SecureField::zero();
    let mut eval_at_2 = SecureField::zero();

    for i in 0..n_terms {
        // Input polynomial at points `(r, {0, 1, 2}, bits(i), {0, 1})`.
        let inp_at_r0i0 = input_layer[i * 2];
        let inp_at_r0i1 = input_layer[i * 2 + 1];
        let inp_at_r1i0 = input_layer[(n_terms + i) * 2];
        let inp_at_r1i1 = input_layer[(n_terms + i) * 2 + 1];
        // Note `inp(r, t, x) = eq(t, 0) * inp(r, 0, x) + eq(t, 1) * inp(r, 1, x)`
        //   => `inp(r, 2, x) = 2 * inp(r, 1, x) - inp(r, 0, x)`
        // TODO(andrew): Consider evaluation at `1/2` to save an addition operation since
        // `inp(r, 1/2, x) = 1/2 * (inp(r, 1, x) + inp(r, 0, x))`. `1/2 * ...` can be factored
        // outside the loop.
        let inp_at_r2i0 = inp_at_r1i0.double() - inp_at_r0i0;
        let inp_at_r2i1 = inp_at_r1i1.double() - inp_at_r0i1;

        // Product polynomial `prod(x) = inp(x, 0) * inp(x, 1)` at points `(r, {0, 2}, bits(i))`.
        let prod_at_r2i = inp_at_r2i0 * inp_at_r2i1;
        let prod_at_r0i = inp_at_r0i0 * inp_at_r0i1;

        let eq_eval_at_0i = eq_evals[i];
        eval_at_0 += eq_eval_at_0i * prod_at_r0i;
        eval_at_2 += eq_eval_at_0i * prod_at_r2i;
    }

    (eval_at_0, eval_at_2)
}

/// Evaluates `sum_x eq(({0}^|r|, 0, x), y) * (inp_numer(r, t, x, 0) * inp_denom(r, t, x, 1) +
/// inp_numer(r, t, x, 1) * inp_denom(r, t, x, 0) + lambda * inp_denom(r, t, x, 0) * inp_denom(r, t,
/// x, 1))` at `t=0` and `t=2`.
///
/// Output of the form: `(eval_at_0, eval_at_2)`.
fn eval_logup_sum<F: Field>(
    eq_evals: &EqEvals<CpuBackend>,
    input_numerators: &Mle<CpuBackend, F>,
    input_denominators: &Mle<CpuBackend, SecureField>,
    n_terms: usize,
    lambda: SecureField,
) -> (SecureField, SecureField)
where
    SecureField: ExtensionOf<F> + Field,
{
    let mut eval_at_0 = SecureField::zero();
    let mut eval_at_2 = SecureField::zero();

    for i in 0..n_terms {
        // Input polynomials at points `(r, {0, 1, 2}, bits(i), {0, 1})`.
        let inp_numer_at_r0i0 = input_numerators[i * 2];
        let inp_denom_at_r0i0 = input_denominators[i * 2];
        let inp_numer_at_r0i1 = input_numerators[i * 2 + 1];
        let inp_denom_at_r0i1 = input_denominators[i * 2 + 1];
        let inp_numer_at_r1i0 = input_numerators[(n_terms + i) * 2];
        let inp_denom_at_r1i0 = input_denominators[(n_terms + i) * 2];
        let inp_numer_at_r1i1 = input_numerators[(n_terms + i) * 2 + 1];
        let inp_denom_at_r1i1 = input_denominators[(n_terms + i) * 2 + 1];
        // Note `inp_denom(r, t, x) = eq(t, 0) * inp_denom(r, 0, x) + eq(t, 1) * inp_denom(r, 1, x)`
        //   => `inp_denom(r, 2, x) = 2 * inp_denom(r, 1, x) - inp_denom(r, 0, x)`
        let inp_numer_at_r2i0 = inp_numer_at_r1i0.double() - inp_numer_at_r0i0;
        let inp_denom_at_r2i0 = inp_denom_at_r1i0.double() - inp_denom_at_r0i0;
        let inp_numer_at_r2i1 = inp_numer_at_r1i1.double() - inp_numer_at_r0i1;
        let inp_denom_at_r2i1 = inp_denom_at_r1i1.double() - inp_denom_at_r0i1;

        // Fraction addition polynomials:
        // - `numer(x) = inp_numer(x, 0) * inp_denom(x, 1) + inp_numer(x, 1) * inp_denom(x, 0)`
        // - `denom(x) = inp_denom(x, 1) * inp_denom(x, 0)`
        // at points `(r, {0, 2}, bits(i))`.
        let Fraction {
            numerator: numer_at_r0i,
            denominator: denom_at_r0i,
        } = Fraction::new(inp_numer_at_r0i0, inp_denom_at_r0i0)
            + Fraction::new(inp_numer_at_r0i1, inp_denom_at_r0i1);
        let Fraction {
            numerator: numer_at_r2i,
            denominator: denom_at_r2i,
        } = Fraction::new(inp_numer_at_r2i0, inp_denom_at_r2i0)
            + Fraction::new(inp_numer_at_r2i1, inp_denom_at_r2i1);

        let eq_eval_at_0i = eq_evals[i];
        eval_at_0 += eq_eval_at_0i * (numer_at_r0i + lambda * denom_at_r0i);
        eval_at_2 += eq_eval_at_0i * (numer_at_r2i + lambda * denom_at_r2i);
    }

    (eval_at_0, eval_at_2)
}

/// Evaluates `sum_x eq(({0}^|r|, 0, x), y) * (inp_denom(r, t, x, 1) + inp_denom(r, t, x, 0) +
/// lambda * inp_denom(r, t, x, 0) * inp_denom(r, t, x, 1))` at `t=0` and `t=2`.
///
/// Output of the form: `(eval_at_0, eval_at_2)`.
fn eval_logup_singles_sum(
    eq_evals: &EqEvals<CpuBackend>,
    input_denominators: &Mle<CpuBackend, SecureField>,
    n_terms: usize,
    lambda: SecureField,
) -> (SecureField, SecureField) {
    let mut eval_at_0 = SecureField::zero();
    let mut eval_at_2 = SecureField::zero();

    for i in 0..n_terms {
        // Input polynomial at points `(r, {0, 1, 2}, bits(i), {0, 1})`.
        let inp_denom_at_r0i0 = input_denominators[i * 2];
        let inp_denom_at_r0i1 = input_denominators[i * 2 + 1];
        let inp_denom_at_r1i0 = input_denominators[(n_terms + i) * 2];
        let inp_denom_at_r1i1 = input_denominators[(n_terms + i) * 2 + 1];
        // Note `inp_denom(r, t, x) = eq(t, 0) * inp_denom(r, 0, x) + eq(t, 1) * inp_denom(r, 1, x)`
        //   => `inp_denom(r, 2, x) = 2 * inp_denom(r, 1, x) - inp_denom(r, 0, x)`
        let inp_denom_at_r2i0 = inp_denom_at_r1i0.double() - inp_denom_at_r0i0;
        let inp_denom_at_r2i1 = inp_denom_at_r1i1.double() - inp_denom_at_r0i1;

        // Fraction addition polynomials at points:
        // - `numer(x) = inp_denom(x, 1) + inp_denom(x, 0)`
        // - `denom(x) = inp_denom(x, 1) * inp_denom(x, 0)`
        // at points `(r, {0, 2}, bits(i))`.
        let Fraction {
            numerator: numer_at_r0i,
            denominator: denom_at_r0i,
        } = Reciprocal::new(inp_denom_at_r0i0) + Reciprocal::new(inp_denom_at_r0i1);
        let Fraction {
            numerator: numer_at_r2i,
            denominator: denom_at_r2i,
        } = Reciprocal::new(inp_denom_at_r2i0) + Reciprocal::new(inp_denom_at_r2i1);

        let eq_eval_at_0i = eq_evals[i];
        eval_at_0 += eq_eval_at_0i * (numer_at_r0i + lambda * denom_at_r0i);
        eval_at_2 += eq_eval_at_0i * (numer_at_r2i + lambda * denom_at_r2i);
    }

    (eval_at_0, eval_at_2)
}

/// Returns evaluations `eq(x, y) * v` for all `x` in `{0, 1}^n`.
///
/// Evaluations are returned in bit-reversed order.
pub fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Vec<SecureField> {
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

impl<F: Field> Index<usize> for MleExpr<'_, F> {
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
        let two = BaseField::from(2).into();
        let y = [7, 3].map(|v| BaseField::from(v).into());

        let eq_evals = CpuBackend::gen_eq_evals(&y, two);

        assert_eq!(
            *eq_evals,
            [
                eq(&[zero, zero], &y) * two,
                eq(&[zero, one], &y) * two,
                eq(&[one, zero], &y) * two,
                eq(&[one, one], &y) * two,
            ]
        );
    }

    #[test]
    fn grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product::<SecureField>();
        let col = Mle::<CpuBackend, SecureField>::new(values);
        let input_layer = Layer::GrandProduct(col.clone());
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

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
            .sum::<Fraction<SecureField, SecureField>>();
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
            .sum::<Fraction<SecureField, SecureField>>();
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
            .sum::<Fraction<SecureField, SecureField>>();
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
