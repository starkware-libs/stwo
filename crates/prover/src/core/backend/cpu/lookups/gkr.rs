use num_traits::Zero;

use crate::core::backend::CpuBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::gkr_prover::{
    correct_sum_as_poly_in_first_variable, EqEvals, GkrMultivariatePolyOracle, GkrOps, Layer,
};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::UnivariatePoly;

impl GkrOps for CpuBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        Mle::new(gen_eq_evals(y, v))
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        match layer {
            Layer::GrandProduct(layer) => next_grand_product_layer(layer),
            Layer::_LogUp(_) => todo!(),
        }
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        let n_variables = h.n_variables();
        let n_terms = 1 << n_variables.saturating_sub(1);
        let eq_evals = h.eq_evals;
        // Vector used to generate evaluations of `eq(x, y)` for `x` in the boolean hypercube.
        let y = eq_evals.y();
        let input_layer = &h.input_layer;

        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();

        match input_layer {
            Layer::GrandProduct(col) => {
                process_grand_product_sum(&mut eval_at_0, &mut eval_at_2, eq_evals, col, n_terms)
            }
            Layer::_LogUp(_) => todo!(),
        }

        // Corrects the difference between two univariate sums in `t`:
        // 1. `sum_x eq(({0}^|r|, 0, x), y) * F(r, t, x)`
        // 2. `sum_x eq((r,       t, x), y) * F(r, t, x)`
        {
            eval_at_0 *= h.eq_fixed_var_correction;
            eval_at_2 *= h.eq_fixed_var_correction;
            correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
        }
    }
}

fn process_grand_product_sum(
    eval_at_0: &mut SecureField,
    eval_at_2: &mut SecureField,
    eq_evals: &EqEvals<CpuBackend>,
    input_layer: &Mle<CpuBackend, SecureField>,
    n_terms: usize,
) {
    for i in 0..n_terms {
        // Input polynomial at points `(r, {0, 1, 2}, bits(i), {0, 1})`.
        let inp_at_r0i0 = input_layer[i * 2];
        let inp_at_r0i1 = input_layer[i * 2 + 1];
        let inp_at_r1i0 = input_layer[(n_terms + i) * 2];
        let inp_at_r1i1 = input_layer[(n_terms + i) * 2 + 1];
        // Note `inp(r, t, x) = eq(t, 0) * inp(r, 0, x) + eq(t, 1) * inp(r, 1, x)`
        //   => `inp(r, 2, x) = 2 * inp(r, 1, x) - inp(r, 0, x)`
        let inp_at_r2i0 = inp_at_r1i0.double() - inp_at_r0i0;
        let inp_at_r2i1 = inp_at_r1i1.double() - inp_at_r0i1;

        // Product polynomial `prod(x) = inp(x, 0) * inp(x, 1)` at points `(r, {0, 2}, bits(i))`.
        let prod_at_r2i = inp_at_r2i0 * inp_at_r2i1;
        let prod_at_r0i = inp_at_r0i0 * inp_at_r0i1;

        let eq_eval_at_0i = eq_evals[i];
        *eval_at_0 += eq_eval_at_0i * prod_at_r0i;
        *eval_at_2 += eq_eval_at_0i * prod_at_r2i;
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

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::core::backend::CpuBackend;
    use crate::core::channel::Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::{prove_batch, GkrOps, Layer};
    use crate::core::lookups::gkr_verifier::{partially_verify_batch, Gate, GkrArtifact, GkrError};
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::utils::eq;
    use crate::core::test_utils::test_channel;

    #[test]
    fn gen_eq_evals() {
        let zero = SecureField::zero();
        let one = SecureField::one();
        let two = BaseField::from(2).into();
        let y = [7, 3].map(|v| BaseField::from(v).into());

        let eq_evals = CpuBackend::gen_eq_evals(&y, two);

        assert_eq!(
            **eq_evals,
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
            ..
        } = partially_verify_batch(vec![Gate::GrandProduct], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_instance, [vec![product]]);
        assert_eq!(claims_to_verify_by_instance, [vec![col.eval_at_point(&r)]]);
        Ok(())
    }
}
