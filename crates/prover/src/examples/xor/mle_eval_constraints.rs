//! Multilinear extension (MLE) eval at point constraints.
use crate::constraint_framework::EvalAtRow;
use crate::core::fields::qm31::SecureField;
use crate::examples::xor::eq_eval_constraints::{
    eq_evals_check, EqEvalsMaskAt, EqEvalsMaskIs, PointMeta,
};
use crate::examples::xor::prefix_sum_constraints::{inclusive_prefix_sum_check, PrefixSumMask};

pub fn mle_eval_check<E: EvalAtRow, const N_VARIABLES: usize>(
    eval: &mut E,
    point_meta: PointMeta<N_VARIABLES>,
    mle_coeff: E::EF,
    mle_claim: SecureField,
    eq_evals_at: &EqEvalsMaskAt<E, N_VARIABLES>,
    eq_evals_is: &EqEvalsMaskIs<E, N_VARIABLES>,
    prefix_sum_at: &PrefixSumMask<E>,
) where
    // Ensure the type exists.
    [(); N_VARIABLES + 1]:,
{
    eq_evals_check(eval, point_meta, eq_evals_at, eq_evals_is);
    let mle_term = mle_coeff * eq_evals_at.curr;
    inclusive_prefix_sum_check(eval, mle_term, mle_claim, eq_evals_is.first, prefix_sum_at);
}

#[cfg(test)]
mod tests {
    use std::array;

    use itertools::{chain, zip_eq};
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::constraint_framework::{assert_constraints, EvalAtRow};
    use crate::core::backend::simd::column::SecureColumn;
    use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Col, Column};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::GkrOps;
    use crate::core::lookups::mle::Mle;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::examples::xor::eq_eval_constraints::{
        self, EqEvalsMaskAt, EqEvalsMaskIs, PointMeta,
    };
    use crate::examples::xor::mle_eval_constraints::mle_eval_check;
    use crate::examples::xor::prefix_sum_constraints::PrefixSumMask;

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
            let eq_evals_is = EqEvalsMaskIs::draw::<CONST_TRACE>(&mut eval);
            let eq_evals_at = EqEvalsMaskAt::draw::<EVAL_TRACE>(&mut eval);
            let prefix_sum_at = PrefixSumMask::draw::<EVAL_TRACE>(&mut eval);
            mle_eval_check(
                &mut eval,
                point_meta,
                mle_coeff,
                claim,
                &eq_evals_at,
                &eq_evals_is,
                &prefix_sum_at,
            );
        });
    }

    /// Generates a trace.
    ///
    /// Trace structure:
    ///
    /// ```text
    /// -------------------------------------------------------------------------------------
    /// |         MLE coeffs        |      eq evals (basis)     |   MLE terms (prefix sum)  |
    /// -------------------------------------------------------------------------------------
    /// |  c0  |  c1  |  c2  |  c3  |  c0  |  c1  |  c2  |  c3  |  c0  |  c1  |  c2  |  c3  |
    /// -------------------------------------------------------------------------------------
    /// ```
    fn gen_base_trace(
        mle: &Mle<SimdBackend, SecureField>,
        eval_point: &[SecureField],
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let mle_coeffs = mle.clone().into_evals();
        let eq_evals = SimdBackend::gen_eq_evals(eval_point, SecureField::one()).into_evals();
        let mle_terms = hadamard_product(&mle_coeffs, &eq_evals);

        let mle_coeff_columns = mle_coeffs.into_secure_column_by_coords().columns;
        let eq_evals_columns = eq_evals.into_secure_column_by_coords().columns;
        let mle_terms_columns = mle_terms.into_secure_column_by_coords().columns;
        let mle_terms_prefix_sum_columns = mle_terms_columns.map(inclusive_prefix_sum);

        let log_trace_domain_size = mle.n_variables() as u32;
        let trace_domain = CanonicCoset::new(log_trace_domain_size).circle_domain();

        chain!(
            mle_coeff_columns.map(|c| CircleEvaluation::new(trace_domain, c)),
            eq_evals_columns.map(|c| CircleEvaluation::new(trace_domain, c)),
            mle_terms_prefix_sum_columns.map(|c| CircleEvaluation::new(trace_domain, c)),
        )
        .collect()
    }

    /// Returns the element-wise product of `a` and `b`.
    fn hadamard_product(
        a: &Col<SimdBackend, SecureField>,
        b: &Col<SimdBackend, SecureField>,
    ) -> Col<SimdBackend, SecureField> {
        assert_eq!(a.len(), b.len());
        SecureColumn {
            data: zip_eq(&a.data, &b.data).map(|(&a, &b)| a * b).collect(),
            length: a.len(),
        }
    }

    fn gen_constants_trace(
        n_variables: usize,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        eq_eval_constraints::tests::gen_constants_trace(n_variables)
    }
}
