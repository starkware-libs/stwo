use num_traits::Zero;

use crate::core::backend::CpuBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::gkr::correct_sum_as_poly_in_first_variable;
use crate::core::lookups::grandproduct::{GrandProductOps, GrandProductOracle, GrandProductTrace};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::UnivariatePoly;

impl GrandProductOps for CpuBackend {
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self> {
        let res = layer.array_chunks().map(|&[a, b]| a * b).collect();
        GrandProductTrace::new(Mle::new(res))
    }

    fn sum_as_poly_in_first_variable(
        h: &GrandProductOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        let k = h.n_variables();
        let n_terms = 1 << (k - 1);
        let eq_evals = h.eq_evals();
        let y = eq_evals.y();
        let trace = h.trace();

        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_terms {
            let lhs0 = trace[i * 2];
            let lhs1 = trace[i * 2 + 1];

            let rhs0 = trace[(n_terms + i) * 2];
            let rhs1 = trace[(n_terms + i) * 2 + 1];

            let product2 = (rhs0.double() - lhs0) * (rhs1.double() - lhs1);
            let product0 = lhs0 * lhs1;

            let eq_eval = eq_evals[i];
            eval_at_0 += eq_eval * product0;
            eval_at_2 += eq_eval * product2;
        }

        eval_at_0 *= h.eq_fixed_var_correction();
        eval_at_2 *= h.eq_fixed_var_correction();

        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, k)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::backend::CpuBackend;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr::{
        partially_verify_batch, prove_batch, GkrBatchVerificationArtifact, GkrError,
    };
    use crate::core::lookups::grandproduct::{GrandProductGate, GrandProductTrace};
    use crate::core::lookups::mle::Mle;
    use crate::core::test_utils::test_channel;

    #[test]
    fn grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product::<SecureField>();
        let top_layer = GrandProductTrace::<CpuBackend>::new(Mle::new(values));
        let proof = prove_batch(&mut test_channel(), vec![top_layer.clone()]);

        let GkrBatchVerificationArtifact {
            ood_point,
            claims_to_verify_by_component,
        } = partially_verify_batch(vec![&GrandProductGate], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_component, [vec![product]]);
        assert_eq!(
            claims_to_verify_by_component,
            [vec![top_layer.eval_at_point(&ood_point)]]
        );
        Ok(())
    }
}
