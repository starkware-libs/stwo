use std::array;

use educe::Educe;
use num_traits::{One, Zero};

use crate::constraint_framework::EvalAtRow;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::eq;

#[derive(Educe)]
#[educe(Debug, Clone, Copy)]
pub struct EqEvalsMask<E: EvalAtRow, const N_VARIABLES: usize> {
    pub curr: E::EF,
    pub next_next: E::EF,
    pub is_first: E::F,
    pub is_second: E::F,
    pub is_half_coset_step_by_log_size: [Option<CircleDomainStepMask<E>>; N_VARIABLES],
}

impl<E: EvalAtRow, const N_VARIABLES: usize> EqEvalsMask<E, N_VARIABLES> {
    pub fn draw<const EQ_EVALS_TRACE: usize, const SELECTOR_TRACE: usize>(eval: &mut E) -> Self {
        let [is_first, is_second] = eval.next_interaction_mask(SELECTOR_TRACE, [0, -1]);
        let [curr, next_next] = eval.next_extension_interaction_mask(EQ_EVALS_TRACE, [0, 2]);

        let mut is_half_coset_step_by_log_size = [None; N_VARIABLES];

        for step in &mut is_half_coset_step_by_log_size[0..N_VARIABLES.saturating_sub(1)] {
            *step = Some(CircleDomainStepMask::draw::<SELECTOR_TRACE>(eval));
        }

        Self {
            curr,
            next_next,
            is_first,
            is_second,
            is_half_coset_step_by_log_size,
        }
    }

    /// Evaluates the constraints on the mask to enforce correct [`eq`] evals.
    ///
    /// See <https://eprint.iacr.org/2023/1284.pdf> (Section 5.1).
    ///
    /// The [`eq`] trace evals appear ordered on a `CircleDomain` rather than a `Coset`. This
    /// is why there are separate sets of constraints for each `CircleDomain` coset half.
    pub fn eval(self, eval: &mut E, mle_eval_point: MleEvalPoint<N_VARIABLES>) {
        let Self {
            curr,
            next_next,
            is_first,
            is_second,
            is_half_coset_step_by_log_size,
        } = self;

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
            let step = is_half_coset_step_by_log_size[variable_i].unwrap();
            let carry_quotient = mle_eval_point.eq_carry_quotients[variable_i];
            // Safe to combine these constraints as `is_step.half_coset0` and `is_step.half_coset1`
            // are never non-zero at the same time on the trace.
            let half_coset0_check = (curr - half_coset0_next * carry_quotient) * step.half_coset0;
            let half_coset1_check = (curr * carry_quotient - half_coset1_prev) * step.half_coset1;
            eval.add_constraint(half_coset0_check + half_coset1_check);
        }
    }
}

#[derive(Educe)]
#[educe(Debug, Clone, Copy)]
pub struct CircleDomainStepMask<E: EvalAtRow> {
    half_coset0: E::F,
    half_coset1: E::F,
}

impl<E: EvalAtRow> CircleDomainStepMask<E> {
    pub fn draw<const TRACE: usize>(eval: &mut E) -> Self {
        let [half_coset0, half_coset1] = eval.next_interaction_mask(TRACE, [0, -1]);
        Self {
            half_coset0,
            half_coset1,
        }
    }
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
    use crate::examples::xor::eq_eval_constraints::EqEvalsMask;

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
            let mask = EqEvalsMask::draw::<EVALS_TRACE, CONST_TRACE>(&mut eval);
            mask.eval(&mut eval, mle_eval_point);
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
            let mask = EqEvalsMask::draw::<EVALS_TRACE, CONST_TRACE>(&mut eval);
            mask.eval(&mut eval, mle_eval_point);
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
            let mask = EqEvalsMask::draw::<EVALS_TRACE, CONST_TRACE>(&mut eval);
            mask.eval(&mut eval, mle_eval_point);
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
