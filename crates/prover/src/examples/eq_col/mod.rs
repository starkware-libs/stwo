#![allow(dead_code)]

use std::array;

use num_traits::{One, Zero};
use tracing::instrument;

use crate::constraint_framework::constant_cols::{gen_is_first, gen_is_step_multiple};
use crate::constraint_framework::EvalAtRow;
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::PackedM31;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Backend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::lookups::utils::eq;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

const BASE_TRACE: usize = 0;
const CONSTANTS_TRACE: usize = 2;

#[derive(Debug, Clone, Copy)]
struct PointMeta<const N_VARIABLES: usize> {
    // Point `p`.
    p: [SecureField; N_VARIABLES],
    // Stores all `eq(0, p_i) / eq(1, p_i)`.
    eq_0pi_div_eq_1pi: [SecureField; N_VARIABLES],
    // Equals `eq({0}^|p|, p)`.
    eq_0_p: SecureField,
}

impl<const N_VARIABLES: usize> PointMeta<N_VARIABLES> {
    /// Creates new metadata from point `p`.
    pub fn new(p: [SecureField; N_VARIABLES]) -> Self {
        let zero = SecureField::zero();
        let one = SecureField::one();

        Self {
            p,
            eq_0pi_div_eq_1pi: array::from_fn(|i| eq(&[zero], &[p[i]]) / eq(&[one], &[p[i]])),
            eq_0_p: eq(&[zero; N_VARIABLES], &p),
        }
    }
}

/// Constraints to enforce correct [`eq`] evals.
///
/// See <https://eprint.iacr.org/2023/1284.pdf> (Section 5.1).
///
/// The [`eq`] trace evals should appear ordered on a `CircleDomain` rather than a `Coset`. This
/// gives context for why there are separate sets of constraints for each `CircleDomain` coset half.
struct EqEvalsCheck<E: EvalAtRow, const N_VARIABLES: usize> {
    eval: E,
    point_meta: PointMeta<N_VARIABLES>,
}

impl<E: EvalAtRow, const N_VARIABLES: usize> EqEvalsCheck<E, N_VARIABLES> {
    fn eval(self) -> (E, EqEvalsCheckMask<E, N_VARIABLES>)
    where
        // Need this const generic to get all required mask items.
        [(); N_VARIABLES + 1]: Exists,
    {
        let Self {
            mut eval,
            point_meta,
        } = self;

        let eq_evals_mask = EqEvalsCheckMask::<E, N_VARIABLES>::new(&mut eval);
        let EqEvalsCheckMask { at_curr, at_steps } = eq_evals_mask;

        let [is_first, is_last] = eval.next_interaction_mask(CONSTANTS_TRACE, [0, 1]);
        eval.add_constraint((at_curr - point_meta.eq_0_p) * is_first);

        let mut at_steps = at_steps.into_iter();

        // Check last variable first due to ordering difference between `Coset` and `CircleDomain`.
        if let Some(at_step) = at_steps.next() {
            // Check eval on first point in half_coset0 with last point in half_coset1.
            let eq_0pi_div_eq_1pi = point_meta.eq_0pi_div_eq_1pi[N_VARIABLES - 1];
            // TODO: Can avoid taking `is_last` mask item by using is_first and setting first base
            // trace mask step to -1 (instead of 1). Constraint only changes slightly.
            eval.add_constraint((at_curr * eq_0pi_div_eq_1pi - at_step) * is_last);
        }

        // Check all other variables (all except last - see above).
        for (variable, at_step) in at_steps.enumerate() {
            // Consider adding `is_steps` to `EqEvalsCheckMask`.
            let [is_step_half_coset0, is_step_half_coset1] =
                eval.next_interaction_mask(CONSTANTS_TRACE, [0, -1]);
            let eq_0pi_div_eq_1pi = point_meta.eq_0pi_div_eq_1pi[variable];
            // TODO: Check if it's safe to combine these constraints
            eval.add_constraint((at_curr - at_step * eq_0pi_div_eq_1pi) * is_step_half_coset0);
            eval.add_constraint((at_curr * eq_0pi_div_eq_1pi - at_step) * is_step_half_coset1);
        }

        (eval, eq_evals_mask)
    }
}

#[derive(Debug, Clone, Copy)]
struct EqEvalsCheckMask<E: EvalAtRow, const N_VARIABLES: usize> {
    at_curr: E::EF,
    at_steps: [E::EF; N_VARIABLES],
}

impl<E: EvalAtRow, const N_VARIABLES: usize> EqEvalsCheckMask<E, N_VARIABLES> {
    pub fn new(eval: &mut E) -> Self
    where
        // Need this const generic to get all required mask items.
        [(); N_VARIABLES + 1]: Exists,
    {
        let mut mask_offsets = [0; N_VARIABLES + 1];
        // Current.
        mask_offsets[0] = 0;
        // Variable step offsets.
        mask_offsets[1..]
            .iter_mut()
            .enumerate()
            .for_each(|(variable, mask_offset)| {
                let variable_step = 1 << variable;
                *mask_offset = variable_step;
            });

        let mask_coord_cols = array::from_fn(|_| eval.next_interaction_mask(0, mask_offsets));

        let mask_items: [E::EF; N_VARIABLES + 1] =
            array::from_fn(|i| E::combine_ef(mask_coord_cols.map(|c| c[i])));

        Self {
            at_curr: mask_items[0],
            at_steps: mask_items[1..].try_into().unwrap(),
        }
    }
}

trait Exists {}

impl<T> Exists for T {}

#[instrument(skip_all)]
fn gen_base_trace<const N_VARIABLES: usize>(
    eval_point: [SecureField; N_VARIABLES],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let eq_evals = SimdBackend::gen_eq_evals(&eval_point, SecureField::one()).into_evals();

    // Currently have SecureField eq_evals.
    // Separate into SECURE_EXTENSION_DEGREE many BaseField columns.
    let mut eq_evals_cols: [Vec<PackedM31>; SECURE_EXTENSION_DEGREE] =
        array::from_fn(|_| Vec::new());

    for secure_vec in &eq_evals.data {
        let [v0, v1, v2, v3] = secure_vec.into_packed_m31s();
        eq_evals_cols[0].push(v0);
        eq_evals_cols[1].push(v1);
        eq_evals_cols[2].push(v2);
        eq_evals_cols[3].push(v3);
    }

    let domain = CanonicCoset::new(eval_point.len() as u32).circle_domain();
    let length = domain.size();
    eq_evals_cols
        .map(|col| BaseFieldVec { data: col, length })
        .map(|col| CircleEvaluation::new(domain, col))
        .into()
}

#[instrument]
fn gen_constants_trace<B: Backend, const N_VARIABLES: usize>(
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let mut constants_trace = Vec::new();

    let log_size = N_VARIABLES as u32;
    constants_trace.push(gen_is_first(log_size));

    // TODO: Last constant column actually equal to gen_is_first but makes the prototype easier.
    for log_step in 1..N_VARIABLES as u32 {
        constants_trace.push(gen_is_step_multiple(log_size, log_step + 1))
    }

    constants_trace
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use super::{EqEvalsCheck, PointMeta};
    use crate::constraint_framework::assert_constraints;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::CanonicCoset;
    use crate::examples::eq_col::{gen_base_trace, gen_constants_trace};

    #[test]
    #[ignore = "SimdBackend `MIN_FFT_LOG_SIZE` is 5"]
    fn test_eq_constraints_with_4_variables() {
        const N_VARIABLES: usize = 4;
        let mut rng = SmallRng::seed_from_u64(0);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let base_trace = gen_base_trace(eval_point);
        let constants_trace = gen_constants_trace::<SimdBackend, N_VARIABLES>();
        let traces = TreeVec::new(vec![base_trace, vec![], constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);
        let point_meta = PointMeta::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |eval| {
            EqEvalsCheck { eval, point_meta }.eval();
        });
    }

    #[test]
    fn test_eq_constraints_with_5_variables() {
        const N_VARIABLES: usize = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let base_trace = gen_base_trace(eval_point);
        let constants_trace = gen_constants_trace::<SimdBackend, N_VARIABLES>();
        let traces = TreeVec::new(vec![base_trace, vec![], constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);
        let point_meta = PointMeta::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |eval| {
            EqEvalsCheck { eval, point_meta }.eval();
        });
    }

    #[test]
    fn test_eq_constraints_with_8_variables() {
        const N_VARIABLES: usize = 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let base_trace = gen_base_trace(eval_point);
        let constants_trace = gen_constants_trace::<SimdBackend, N_VARIABLES>();
        let traces = TreeVec::new(vec![base_trace, vec![], constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);
        let point_meta = PointMeta::new(eval_point);

        assert_constraints(&trace_polys, trace_domain, |eval| {
            EqEvalsCheck { eval, point_meta }.eval();
        });
    }
}
