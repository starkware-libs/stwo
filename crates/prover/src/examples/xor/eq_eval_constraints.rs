use std::array;

use educe::Educe;
use itertools::izip;
use num_traits::{One, Zero};

use crate::constraint_framework::EvalAtRow;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::eq;

/// Constraints to enforce correct [`eq`] evals.
///
/// See <https://eprint.iacr.org/2023/1284.pdf> (Section 5.1).
///
/// The [`eq`] trace evals should appear ordered on a `CircleDomain` rather than a `Coset`. This
/// gives context for why there are separate sets of constraints for each `CircleDomain` coset half.
pub fn eq_evals_check<E: EvalAtRow, const N_VARIABLES: usize>(
    eval: &mut E,
    point_meta: PointMeta<N_VARIABLES>,
    at: &EqEvalsMaskAt<E, N_VARIABLES>,
    is: &EqEvalsMaskIs<E, N_VARIABLES>,
) where
    // Ensure the type exists.
    [(); N_VARIABLES + 1]:,
{
    eval.add_constraint((at.curr - point_meta.eq_0_p) * is.first);

    let mut at_steps = at.steps.into_iter();

    // Check last variable first due to ordering difference between `Coset` and `CircleDomain`.
    if let Some(at_step) = at_steps.next() {
        // Check eval on first point in half_coset0 with first point in half_coset1.
        let eq_0pi_div_eq_1pi = point_meta.eq_0pi_div_eq_1pi[N_VARIABLES - 1];
        eval.add_constraint((at.curr - at_step * eq_0pi_div_eq_1pi) * is.first);
    }

    // Check all other variables (all except last - see above).
    for (variable, (at_step, is_step)) in izip!(at_steps, is.step_by_log_size).enumerate() {
        let is_step = is_step.unwrap();
        let eq_0pi_div_eq_1pi = point_meta.eq_0pi_div_eq_1pi[variable];
        // Safe to combine these constraints as `is_step.half_coset0` and `is_step.half_coset1` are
        // never non-zero at the same time on the trace.
        let half_coset0_check = (at.curr - at_step * eq_0pi_div_eq_1pi) * is_step.half_coset0;
        let half_coset1_check = (at.curr * eq_0pi_div_eq_1pi - at_step) * is_step.half_coset1;
        eval.add_constraint(half_coset0_check + half_coset1_check);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EqEvalsMaskAt<E: EvalAtRow, const N_VARIABLES: usize> {
    pub curr: E::EF,
    pub steps: [E::EF; N_VARIABLES],
}

impl<E: EvalAtRow, const N_VARIABLES: usize> EqEvalsMaskAt<E, N_VARIABLES> {
    pub fn draw<const TRACE: usize>(eval: &mut E) -> Self
    where
        // Ensure the type exists.
        [(); N_VARIABLES + 1]:,
    {
        let current_offset = 0;
        let mut variable_step_offsets: [isize; N_VARIABLES] =
            array::from_fn(|variable| 1 << variable);

        // Swap first step due to ordering difference between `Coset` and `CircleDomain`.
        if let [first_step, _remaining_steps @ ..] = variable_step_offsets.as_mut_slice() {
            *first_step = -*first_step;
        }

        let mut mask_offsets = [0; N_VARIABLES + 1];
        mask_offsets[0] = current_offset;
        mask_offsets[1..].copy_from_slice(&variable_step_offsets);

        let mask_items = eval.next_extension_interaction_mask(TRACE, mask_offsets);

        Self {
            curr: mask_items[0],
            steps: mask_items[1..].try_into().unwrap(),
        }
    }
}

#[derive(Educe)]
#[educe(Debug, Clone, Copy)]
pub struct EqEvalsMaskIs<E: EvalAtRow, const N_VARIABLES: usize> {
    pub first: E::F,
    pub step_by_log_size: [Option<CircleDomainStepMask<E>>; N_VARIABLES],
}

impl<E: EvalAtRow, const N_VARIABLES: usize> EqEvalsMaskIs<E, N_VARIABLES> {
    pub fn draw<const TRACE: usize>(eval: &mut E) -> Self {
        let [first] = eval.next_interaction_mask(TRACE, [0]);
        Self::draw_steps::<TRACE>(eval, first)
    }

    pub fn draw_steps<const TRACE: usize>(eval: &mut E, first: E::F) -> Self {
        let mut step_by_log_size = [None; N_VARIABLES];

        for step in &mut step_by_log_size[0..N_VARIABLES.saturating_sub(1)] {
            *step = Some(CircleDomainStepMask::draw::<TRACE>(eval));
        }

        Self {
            first,
            step_by_log_size,
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
pub struct PointMeta<const N_VARIABLES: usize> {
    // Equals `eq({0}^|p|, p)`.
    eq_0_p: SecureField,
    // Stores all `eq(0, p_i) / eq(1, p_i)`.
    eq_0pi_div_eq_1pi: [SecureField; N_VARIABLES],
    // Point `p`.
    _p: [SecureField; N_VARIABLES],
}

impl<const N_VARIABLES: usize> PointMeta<N_VARIABLES> {
    /// Creates new metadata from point `p`.
    pub fn new(p: [SecureField; N_VARIABLES]) -> Self {
        let zero = SecureField::zero();
        let one = SecureField::one();

        Self {
            eq_0_p: eq(&[zero; N_VARIABLES], &p),
            eq_0pi_div_eq_1pi: array::from_fn(|i| eq(&[zero], &[p[i]]) / eq(&[one], &[p[i]])),
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

    use super::PointMeta;
    use crate::constraint_framework::assert_constraints;
    use crate::constraint_framework::constant_columns::{gen_is_first, gen_is_step_multiple};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::GkrOps;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::core::ColumnVec;
    use crate::examples::xor::eq_eval_constraints::{eq_evals_check, EqEvalsMaskAt, EqEvalsMaskIs};

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
        let point_meta = PointMeta::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let at_mask = EqEvalsMaskAt::draw::<EVALS_TRACE>(&mut eval);
            let is_mask = EqEvalsMaskIs::draw::<CONST_TRACE>(&mut eval);
            eq_evals_check(&mut eval, point_meta, &at_mask, &is_mask);
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
        let point_meta = PointMeta::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let at_mask = EqEvalsMaskAt::draw::<EVALS_TRACE>(&mut eval);
            let is_mask = EqEvalsMaskIs::draw::<CONST_TRACE>(&mut eval);
            eq_evals_check(&mut eval, point_meta, &at_mask, &is_mask);
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
        let point_meta = PointMeta::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let at_mask = EqEvalsMaskAt::draw::<EVALS_TRACE>(&mut eval);
            let is_mask = EqEvalsMaskIs::draw::<CONST_TRACE>(&mut eval);
            eq_evals_check(&mut eval, point_meta, &at_mask, &is_mask);
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
        let mut constants_trace = Vec::new();

        constants_trace.push(gen_is_first(n_variables as u32));

        for log_step in 1..n_variables as u32 {
            constants_trace.push(gen_is_step_multiple(n_variables as u32, log_step + 1))
        }

        constants_trace
    }
}
