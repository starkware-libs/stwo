use num_traits::Zero;

use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::correct_sum_as_poly_in_first_variable;
use crate::core::lookups::grandproduct::{GrandProductOps, GrandProductOracle, GrandProductTrace};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::UnivariatePoly;

impl GrandProductOps for SimdBackend {
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self> {
        let next_layer_len = layer.len() / 2;

        // Offload to CPU backend to prevent dealing with instances smaller than a SIMD vector.
        if next_layer_len < N_LANES {
            return to_simd_trace(&CpuBackend::next_layer(&layer.to_cpu()));
        }

        let data = layer
            .data
            .array_chunks()
            .map(|&[a, b]| {
                let (evens, odds) = a.deinterleave(b);
                evens * odds
            })
            .collect();

        GrandProductTrace::new(Mle::new(SecureFieldVec {
            data,
            length: next_layer_len,
        }))
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

        // Offload to CPU backend to prevent dealing with instances smaller than a SIMD vector.
        if n_terms < 2 * N_LANES {
            return h.to_cpu().sum_as_poly_in_first_variable(claim);
        }

        let n_packed_terms = n_terms / N_LANES;
        let (lhs_data, rhs_data) = trace.data.split_at(trace.data.len() / 2);

        let mut packed_eval_at_0 = PackedSecureField::zero();
        let mut packed_eval_at_2 = PackedSecureField::zero();

        for i in 0..n_packed_terms {
            let (lhs0, lhs1) = lhs_data[i * 2].deinterleave(lhs_data[i * 2 + 1]);
            let (rhs0, rhs1) = rhs_data[i * 2].deinterleave(rhs_data[i * 2 + 1]);

            let product2 = (rhs0.double() - lhs0) * (rhs1.double() - lhs1);
            let product0 = lhs0 * lhs1;

            let eq_eval = eq_evals.data[i];
            packed_eval_at_0 += eq_eval * product0;
            packed_eval_at_2 += eq_eval * product2;
        }

        let eval_at_0 = packed_eval_at_0.pointwise_sum() * h.eq_fixed_var_correction();
        let eval_at_2 = packed_eval_at_2.pointwise_sum() * h.eq_fixed_var_correction();

        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, k)
    }
}

fn to_simd_trace(cpu_trace: &GrandProductTrace<CpuBackend>) -> GrandProductTrace<SimdBackend> {
    GrandProductTrace::new(Mle::new((**cpu_trace).to_cpu().into_iter().collect()))
}

#[cfg(test)]
mod tests {
    use crate::core::backend::simd::SimdBackend;
    use crate::core::channel::Channel;
    use crate::core::lookups::gkr::{partially_verify_batch, prove_batch, GkrArtifact, GkrError};
    use crate::core::lookups::grandproduct::{GrandProductGate, GrandProductTrace};
    use crate::core::lookups::mle::Mle;
    use crate::core::test_utils::test_channel;

    #[test]
    fn grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 6;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product();
        let mle = Mle::new(values.into_iter().collect());
        let top_layer = GrandProductTrace::<SimdBackend>::new(mle);
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer.clone()]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_component,
            n_variables_by_component: _,
        } = partially_verify_batch(vec![&GrandProductGate], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_component, [vec![product]]);
        assert_eq!(
            claims_to_verify_by_component,
            [vec![top_layer.eval_at_point(&ood_point)]]
        );
        Ok(())
    }
}
