use std::array;

use num_traits::{One, Zero};

use crate::constraint_framework::EvalAtRow;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::eq;

/// Evaluates EqEvals constraints on a column.
///
/// Returns the evaluation at offset 0 on the column.
///
/// Given a column `c(P)` defined on a circle domain `D`, and an MLE eval point `(r0, r1, ...)`
/// evaluates constraints that guarantee: `c(D[b0, b1, ...]) = eq((b0, b1, ...), (r0, r1, ...))`.
///
/// See <https://eprint.iacr.org/2023/1284.pdf> (Section 5.1).
pub fn eval_eq_constraints<E: EvalAtRow, const N_VARIABLES: usize>(
    eq_interaction: usize,
    selector_interaction: usize,
    eval: &mut E,
    mle_eval_point: MleEvalPoint<N_VARIABLES>,
) -> E::EF {
    let [curr, next_next] = eval.next_extension_interaction_mask(eq_interaction, [0, 2]);
    let [is_first, is_second] = eval.next_interaction_mask(selector_interaction, [0, -1]);

    // Check the initial value on half_coset0 and final value on half_coset1.
    // Combining these constraints is safe because `is_first` and `is_second` are never
    // non-zero at the same time on the trace.
    let half_coset0_initial_check = (curr - mle_eval_point.eq_0_p) * is_first;
    let half_coset1_final_check = (curr - mle_eval_point.eq_1_p) * is_second;
    eval.add_constraint(half_coset0_initial_check + half_coset1_final_check);

    // Check all variables except the last (last variable is handled by the constraint above).
    #[allow(clippy::needless_range_loop)]
    for variable_i in 0..N_VARIABLES.saturating_sub(1) {
        let half_coset0_next = next_next;
        let half_coset1_prev = next_next;
        let [half_coset0_step, half_coset1_step] =
            eval.next_interaction_mask(selector_interaction, [0, -1]);
        let carry_quotient = mle_eval_point.eq_carry_quotients[variable_i];
        // Safe to combine these constraints as `is_step.half_coset0` and `is_step.half_coset1`
        // are never non-zero at the same time on the trace.
        let half_coset0_check = (curr - half_coset0_next * carry_quotient) * half_coset0_step;
        let half_coset1_check = (curr * carry_quotient - half_coset1_prev) * half_coset1_step;
        eval.add_constraint(half_coset0_check + half_coset1_check);
    }

    curr
}

#[derive(Debug, Clone, Copy)]
pub struct MleEvalPoint<const N_VARIABLES: usize> {
    // Equals `eq({0}^|p|, p)`.
    eq_0_p: SecureField,
    // Equals `eq({1}^|p|, p)`.
    eq_1_p: SecureField,
    // Index `i` stores `eq(({1}^|i|, 0), p[0..i+1]) / eq(({0}^|i|, 1), p[0..i+1])`.
    eq_carry_quotients: [SecureField; N_VARIABLES],
    // Point `p`.
    _p: [SecureField; N_VARIABLES],
}

impl<const N_VARIABLES: usize> MleEvalPoint<N_VARIABLES> {
    /// Creates new metadata from point `p`.
    pub fn new(p: [SecureField; N_VARIABLES]) -> Self {
        let zero = SecureField::zero();
        let one = SecureField::one();

        Self {
            eq_0_p: eq(&[zero; N_VARIABLES], &p),
            eq_1_p: eq(&[one; N_VARIABLES], &p),
            eq_carry_quotients: array::from_fn(|i| {
                let mut numer_assignment = vec![one; i + 1];
                numer_assignment[i] = zero;
                let mut denom_assignment = vec![zero; i + 1];
                denom_assignment[i] = one;
                eq(&numer_assignment, &p[..i + 1]) / eq(&denom_assignment, &p[..i + 1])
            }),
            _p: p,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use std::array;

    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use super::MleEvalPoint;
    use crate::constraint_framework::assert_constraints;
    use crate::constraint_framework::constant_columns::{gen_is_first, gen_is_step_with_offset};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::GkrOps;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::core::ColumnVec;
    use crate::examples::xor::eq_eval_constraints::eval_eq_constraints;

    const EVALS_TRACE: usize = 0;
    const CONST_TRACE: usize = 1;

    #[test]
    #[ignore = "SimdBackend `MIN_FFT_LOG_SIZE` is 5"]
    fn eq_constraints_with_4_variables() {
        const N_VARIABLES: usize = 4;
        let mut rng = SmallRng::seed_from_u64(0);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let base_trace = gen_evals_trace(&eval_point);
        let constants_trace = gen_constants_trace(N_VARIABLES);
        let traces = TreeVec::new(vec![base_trace, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);
        let mle_eval_point = MleEvalPoint::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            eval_eq_constraints(EVALS_TRACE, CONST_TRACE, &mut eval, mle_eval_point);
        });
    }

    #[test]
    fn eq_constraints_with_5_variables() {
        const N_VARIABLES: usize = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let base_trace = gen_evals_trace(&eval_point);
        let constants_trace = gen_constants_trace(N_VARIABLES);
        let traces = TreeVec::new(vec![base_trace, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);
        let mle_eval_point = MleEvalPoint::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            eval_eq_constraints(EVALS_TRACE, CONST_TRACE, &mut eval, mle_eval_point);
        });
    }

    #[test]
    fn eq_constraints_with_8_variables() {
        const N_VARIABLES: usize = 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let base_trace = gen_evals_trace(&eval_point);
        let constants_trace = gen_constants_trace(N_VARIABLES);
        let traces = TreeVec::new(vec![base_trace, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);
        let mle_eval_point = MleEvalPoint::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            eval_eq_constraints(EVALS_TRACE, CONST_TRACE, &mut eval, mle_eval_point);
        });
    }

    /// Generates a trace.
    ///
    /// Trace structure:
    ///
    /// ```text
    /// -----------------------------
    /// |          eq evals         |
    /// -----------------------------
    /// |  c0  |  c1  |  c2  |  c3  |
    /// -----------------------------
    /// ```
    pub fn gen_evals_trace(
        eval_point: &[SecureField],
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        // TODO(andrew): Consider storing eq evals as a SecureColumn.
        let eq_evals = SimdBackend::gen_eq_evals(eval_point, SecureField::one()).into_evals();
        let eq_evals_coordinate_columns = eq_evals.into_secure_column_by_coords().columns;

        let n_variables = eval_point.len();
        let domain = CanonicCoset::new(n_variables as u32).circle_domain();
        eq_evals_coordinate_columns
            .map(|col| CircleEvaluation::new(domain, col))
            .into()
    }

    pub fn gen_constants_trace(
        n_variables: usize,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let log_size = n_variables as u32;
        let mut constants_trace = Vec::new();
        constants_trace.push(gen_is_first(log_size));

        // TODO(andrew): Note the last selector column is not needed. The column for `is_first`
        // with an offset for each half coset midpoint can be used instead.
        for variable_i in 1..n_variables as u32 {
            let half_coset_log_step = variable_i;
            let half_coset_offset = (1 << (half_coset_log_step - 1)) - 1;

            let log_step = half_coset_log_step + 1;
            let offset = half_coset_offset * 2;

            constants_trace.push(gen_is_step_with_offset(log_size, log_step, offset))
        }

        constants_trace
    }
}
