//! Multilinear extension (MLE) eval at point constraints.
use super::eq_eval_constraints::{eval_eq_constraints, MleEvalPoint};
use crate::constraint_framework::EvalAtRow;
use crate::core::fields::qm31::SecureField;
use crate::examples::xor::prefix_sum_constraints::eval_prefix_sum_constraints;

/// Evaluates constraints that grantee a MLE evaluates to a claim at a given point.
///
/// `mle_coeffs_col_eval` should be the evaluation of the column containing the coefficients of the
/// MLE in the multilinear Lagrange basis. `mle_claim_shift` should equal `claim / 2^N_VARIABLES`.
pub fn eval_mle_eval_constrants<E: EvalAtRow, const N_VARIABLES: usize>(
    mle_interaction: usize,
    selector_interaction: usize,
    eval: &mut E,
    mle_coeffs_col_eval: E::EF,
    mle_eval_point: MleEvalPoint<N_VARIABLES>,
    mle_claim_shift: SecureField,
) {
    let eq_col_eval =
        eval_eq_constraints(mle_interaction, selector_interaction, eval, mle_eval_point);
    let terms_col_eval = mle_coeffs_col_eval * eq_col_eval;
    eval_prefix_sum_constraints(mle_interaction, eval, terms_col_eval, mle_claim_shift)
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::iter::zip;

    use itertools::{chain, zip_eq};
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::constraint_framework::{assert_constraints, EvalAtRow};
    use crate::core::backend::simd::column::SecureColumn;
    use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
    use crate::core::backend::simd::qm31::PackedSecureField;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Col, Column};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::GkrOps;
    use crate::core::lookups::mle::Mle;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::examples::xor::eq_eval_constraints::{self, MleEvalPoint};
    use crate::examples::xor::mle_eval_constraints::eval_mle_eval_constrants;

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
        let mle_eval_point = MleEvalPoint::new(eval_point);
        let base_trace = gen_base_trace(&mle, &eval_point);
        let claim = mle.eval_at_point(&eval_point);
        let claim_shift = claim / BaseField::from(size);
        let constants_trace = gen_constants_trace(N_VARIABLES);
        let traces = TreeVec::new(vec![base_trace, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(log_size);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let [mle_coeff_col_eval] = eval.next_extension_interaction_mask(EVAL_TRACE, [0]);
            eval_mle_eval_constrants(
                EVAL_TRACE,
                CONST_TRACE,
                &mut eval,
                mle_coeff_col_eval,
                mle_eval_point,
                claim_shift,
            )
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
    /// |  c0  |  c1  |  c2  |  c3  |  c4  |  c5  |  c6  |  c7  |  c9  |  c9  |  c10 |  c11 |
    /// -------------------------------------------------------------------------------------
    /// ```
    fn gen_base_trace(
        mle: &Mle<SimdBackend, SecureField>,
        eval_point: &[SecureField],
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let mle_coeffs = mle.clone().into_evals();
        let eq_evals = SimdBackend::gen_eq_evals(eval_point, SecureField::one()).into_evals();
        let mle_terms = hadamard_product(&mle_coeffs, &eq_evals);

        let mle_coeff_cols = mle_coeffs.into_secure_column_by_coords().columns;
        let eq_evals_cols = eq_evals.into_secure_column_by_coords().columns;
        let mle_terms_cols = mle_terms.into_secure_column_by_coords().columns;

        let claim = mle.eval_at_point(eval_point);
        let shift = claim / BaseField::from(mle.len());
        let packed_shifts = PackedSecureField::broadcast(shift).into_packed_m31s();
        let mut shifted_mle_terms_cols = mle_terms_cols.clone();
        zip(&mut shifted_mle_terms_cols, packed_shifts)
            .for_each(|(col, shift)| col.data.iter_mut().for_each(|v| *v -= shift));
        let shifted_prefix_sum_cols = shifted_mle_terms_cols.map(inclusive_prefix_sum);

        let log_trace_domain_size = mle.n_variables() as u32;
        let trace_domain = CanonicCoset::new(log_trace_domain_size).circle_domain();

        chain![mle_coeff_cols, eq_evals_cols, shifted_prefix_sum_cols]
            .map(|c| CircleEvaluation::new(trace_domain, c))
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
