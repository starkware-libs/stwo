//! Multilinear extension (MLE) eval at point.
use crate::constraint_framework::EvalAtRow;
use crate::core::fields::qm31::SecureField;
use crate::examples::xor::eq_evals::constraints::{
    eq_evals_check, EqEvalsMaskAt, EqEvalsMaskIs, PointMeta,
};
use crate::examples::xor::prefix_sum::{prefix_sum_check, PrefixSumMaskAt, PrefixSumMaskIs};

#[allow(clippy::too_many_arguments)]
pub fn mle_eval_check<E: EvalAtRow, const N_VARIABLES: usize>(
    eval: &mut E,
    point_meta: PointMeta<N_VARIABLES>,
    mle_coeff: E::EF,
    mle_claim: SecureField,
    eq_evals_at: &EqEvalsMaskAt<E, N_VARIABLES>,
    eq_evals_is: &EqEvalsMaskIs<E, N_VARIABLES>,
    prefix_sum_at: &PrefixSumMaskAt<E>,
    prefix_sum_is: &PrefixSumMaskIs<E>,
) where
    // Ensure the type exists.
    [(); N_VARIABLES + 1]:,
{
    eq_evals_check(eval, point_meta, eq_evals_at, eq_evals_is);
    let mle_term = mle_coeff * eq_evals_at.curr;
    prefix_sum_check(eval, mle_term, mle_claim, prefix_sum_at, prefix_sum_is);
}

#[cfg(test)]
mod tests {
    use std::array;

    use itertools::{zip_eq, Itertools};
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::constraint_framework::{assert_constraints, EvalAtRow};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::GkrOps;
    use crate::core::lookups::mle::Mle;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::core::utils::{
        bit_reverse, circle_domain_order_to_coset_order, coset_order_to_circle_domain_order,
    };
    use crate::examples::xor::eq_evals;
    use crate::examples::xor::eq_evals::constraints::PointMeta;
    use crate::examples::xor::mle_eval::{mle_eval_check, EqEvalsMaskAt, EqEvalsMaskIs};
    use crate::examples::xor::prefix_sum::{prefix_sum, PrefixSumMaskAt, PrefixSumMaskIs};

    const EVAL_TRACE: usize = 0;
    const CONST_TRACE: usize = 1;

    #[test]
    fn test_mle_eval_constraints_with_log_size_5() {
        const N_VARIABLES: usize = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let log_size = N_VARIABLES as u32;
        let size = 1 << log_size;
        let mle = Mle::new((0..size).map(|_| rng.gen::<SecureField>()).collect());
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let point_meta = PointMeta::new(eval_point);
        let base_trace = gen_base_trace(&mle, &eval_point);
        let claim = mle.eval_at_point(&eval_point);
        let constants_trace = gen_constants_trace(N_VARIABLES);
        let traces = TreeVec::new(vec![base_trace, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(log_size);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let [mle_coeff] = eval.next_extension_interaction_mask(EVAL_TRACE, [0]);
            // Flags for first and last point in trace domain.
            let [first, last] = eval.next_interaction_mask(CONST_TRACE, [0, 1]);
            let eq_evals_at = EqEvalsMaskAt::draw::<EVAL_TRACE>(&mut eval);
            let eq_evals_is = EqEvalsMaskIs::draw_steps::<CONST_TRACE>(&mut eval, first);
            let prefix_sum_at = PrefixSumMaskAt::draw::<EVAL_TRACE>(&mut eval);
            let prefix_sum_is = PrefixSumMaskIs { first, last };
            mle_eval_check(
                &mut eval,
                point_meta,
                mle_coeff,
                claim,
                &eq_evals_at,
                &eq_evals_is,
                &prefix_sum_at,
                &prefix_sum_is,
            );
        });
    }

    /// Generates a trace.
    ///
    /// Trace structure:
    ///
    /// ```text
    /// -------------------------------------------------------------------------------------
    /// |         MLE coeffs        |      EQ evals (basis)     |   MLE terms (prefix sum)  |
    /// -------------------------------------------------------------------------------------
    /// |  c0  |  c1  |  c2  |  c3  |  c0  |  c1  |  c2  |  c3  |  c0  |  c1  |  c2  |  c3  |
    /// -------------------------------------------------------------------------------------
    /// ```
    fn gen_base_trace(
        mle: &Mle<SimdBackend, SecureField>,
        eval_point: &[SecureField],
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let mle_coeffs = mle.to_cpu();
        let eq_evals = CpuBackend::gen_eq_evals(eval_point, SecureField::one()).into_evals();
        let mle_terms = zip_eq(&mle_coeffs, &eq_evals)
            .map(|(&coeff, &eq_eval)| coeff * eq_eval)
            .collect_vec();

        let mut mle_terms_circle_domain_order = mle_terms;
        bit_reverse(&mut mle_terms_circle_domain_order);
        let mle_terms_coset_order =
            circle_domain_order_to_coset_order(&mle_terms_circle_domain_order);

        let mle_terms_prefix_sum_coset_order = prefix_sum(&mle_terms_coset_order);
        let mle_terms_prefix_sum_circle_domain_order =
            coset_order_to_circle_domain_order(&mle_terms_prefix_sum_coset_order);
        let mut mle_terms_prefix_sum = mle_terms_prefix_sum_circle_domain_order;
        bit_reverse(&mut mle_terms_prefix_sum);

        let mut mle_coeff_col0 = Vec::new();
        let mut mle_coeff_col1 = Vec::new();
        let mut mle_coeff_col2 = Vec::new();
        let mut mle_coeff_col3 = Vec::new();

        for v in mle_coeffs {
            let [v0, v1, v2, v3] = v.to_m31_array();
            mle_coeff_col0.push(v0);
            mle_coeff_col1.push(v1);
            mle_coeff_col2.push(v2);
            mle_coeff_col3.push(v3);
        }

        let mut eq_evals_col0 = Vec::new();
        let mut eq_evals_col1 = Vec::new();
        let mut eq_evals_col2 = Vec::new();
        let mut eq_evals_col3 = Vec::new();

        for v in eq_evals {
            let [v0, v1, v2, v3] = v.to_m31_array();
            eq_evals_col0.push(v0);
            eq_evals_col1.push(v1);
            eq_evals_col2.push(v2);
            eq_evals_col3.push(v3);
        }

        let mut mle_terms_prefix_sum_col0 = Vec::new();
        let mut mle_terms_prefix_sum_col1 = Vec::new();
        let mut mle_terms_prefix_sum_col2 = Vec::new();
        let mut mle_terms_prefix_sum_col3 = Vec::new();

        for v in mle_terms_prefix_sum {
            let [v0, v1, v2, v3] = v.to_m31_array();
            mle_terms_prefix_sum_col0.push(v0);
            mle_terms_prefix_sum_col1.push(v1);
            mle_terms_prefix_sum_col2.push(v2);
            mle_terms_prefix_sum_col3.push(v3);
        }

        let log_size = mle.n_variables() as u32;
        let domain = CanonicCoset::new(log_size).circle_domain();

        vec![
            CircleEvaluation::new(domain, mle_coeff_col0.into_iter().collect()),
            CircleEvaluation::new(domain, mle_coeff_col1.into_iter().collect()),
            CircleEvaluation::new(domain, mle_coeff_col2.into_iter().collect()),
            CircleEvaluation::new(domain, mle_coeff_col3.into_iter().collect()),
            CircleEvaluation::new(domain, eq_evals_col0.into_iter().collect()),
            CircleEvaluation::new(domain, eq_evals_col1.into_iter().collect()),
            CircleEvaluation::new(domain, eq_evals_col2.into_iter().collect()),
            CircleEvaluation::new(domain, eq_evals_col3.into_iter().collect()),
            CircleEvaluation::new(domain, mle_terms_prefix_sum_col0.into_iter().collect()),
            CircleEvaluation::new(domain, mle_terms_prefix_sum_col1.into_iter().collect()),
            CircleEvaluation::new(domain, mle_terms_prefix_sum_col2.into_iter().collect()),
            CircleEvaluation::new(domain, mle_terms_prefix_sum_col3.into_iter().collect()),
        ]
    }

    fn gen_constants_trace(
        n_variables: usize,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        eq_evals::trace::gen_constants_trace(n_variables)
    }
}
