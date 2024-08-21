//! Multilinear extension (MLE) eval at point constraints.
// TODO(andrew): Remove in downstream PR.
#![allow(dead_code)]

use std::array;

use itertools::Itertools;
use num_traits::{One, Zero};

use crate::constraint_framework::EvalAtRow;
use crate::core::backend::simd::SimdBackend;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{coset_vanishing, point_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::{Field, FieldExpOps};
use crate::core::lookups::utils::eq;
use crate::core::poly::circle::{CanonicCoset, SecureEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};

/// Evaluates constraints that guarantee an MLE evaluates to a claim at a given point.
///
/// `mle_coeffs_col_eval` should be the evaluation of the column containing the coefficients of the
/// MLE in the multilinear Lagrange basis. `mle_claim_shift` should equal `claim / 2^N_VARIABLES`.
pub fn eval_mle_eval_constraints<E: EvalAtRow, const N_VARIABLES: usize>(
    mle_interaction: usize,
    const_interaction: usize,
    eval: &mut E,
    mle_coeffs_col_eval: E::EF,
    mle_eval_point: MleEvalPoint<N_VARIABLES>,
    mle_claim_shift: SecureField,
    carry_quotients_col_eval: E::EF,
) {
    let eq_col_eval = eval_eq_constraints(
        mle_interaction,
        const_interaction,
        eval,
        mle_eval_point,
        carry_quotients_col_eval,
    );
    let terms_col_eval = mle_coeffs_col_eval * eq_col_eval;
    eval_prefix_sum_constraints(mle_interaction, eval, terms_col_eval, mle_claim_shift)
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
    p: [SecureField; N_VARIABLES],
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
            p,
        }
    }
}

/// Evaluates EqEvals constraints on a column.
///
/// Returns the evaluation at offset 0 on the column.
///
/// Given a column `c(P)` defined on a circle domain `D`, and an MLE eval point `(r0, r1, ...)`
/// evaluates constraints that guarantee: `c(D[b0, b1, ...]) = eq((b0, b1, ...), (r0, r1, ...))`.
///
/// See <https://eprint.iacr.org/2023/1284.pdf> (Section 5.1).
fn eval_eq_constraints<E: EvalAtRow, const N_VARIABLES: usize>(
    eq_interaction: usize,
    const_interaction: usize,
    eval: &mut E,
    mle_eval_point: MleEvalPoint<N_VARIABLES>,
    carry_quotients_col_eval: E::EF,
) -> E::EF {
    let [curr, next_next] = eval.next_extension_interaction_mask(eq_interaction, [0, 2]);
    let [is_first, is_second] = eval.next_interaction_mask(const_interaction, [0, -1]);

    // Check the initial value on half_coset0 and final value on half_coset1.
    // Combining these constraints is safe because `is_first` and `is_second` are never
    // non-zero at the same time on the trace.
    let half_coset0_initial_check = (curr - mle_eval_point.eq_0_p) * is_first;
    let half_coset1_final_check = (curr - mle_eval_point.eq_1_p) * is_second;
    eval.add_constraint(half_coset0_initial_check + half_coset1_final_check);

    // Check all the steps.
    eval.add_constraint(curr - next_next * carry_quotients_col_eval);

    curr
}

/// Evaluates inclusive prefix sum constraints on a column.
///
/// Note the column values must be shifted by `cumulative_sum_shift` so the last value equals zero.
/// `cumulative_sum_shift` should equal `cumulative_sum / column_size`.
fn eval_prefix_sum_constraints<E: EvalAtRow>(
    interaction: usize,
    eval: &mut E,
    row_diff: E::EF,
    cumulative_sum_shift: SecureField,
) {
    let [curr, prev] = eval.next_extension_interaction_mask(interaction, [0, -1]);
    eval.add_constraint(curr - prev - row_diff + cumulative_sum_shift);
}

/// Returns succinct Eq carry quotients column.
///
/// Given column `c(P)` defined on a [`CircleDomain`] `D = +-C`, and an MLE eval point
/// `(r0, r1, ...)` let `c(D[b0, b1, ...]) = eq((b0, b1, ...), (r0, r1, ...))`. This function
/// returns column `q(P)` such that all `c(C[i]) = c(C[i + 1]) * q(C[i])` and
/// `c(-C[i]) = c(-C[i + 1]) * q(-C[i])`.
///
/// [`CircleDomain`]: crate::core::poly::circle::CircleDomain
fn gen_carry_quotient_col<const N_VARIABLES: usize>(
    eval_point: &MleEvalPoint<N_VARIABLES>,
) -> SecureEvaluation<SimdBackend, BitReversedOrder> {
    let (half_coset0_carry_quotients, half_coset1_carry_quotients) =
        gen_half_coset_carry_quotients(eval_point);

    let log_size = N_VARIABLES as u32;
    let size = 1 << log_size;
    let half_coset_size = size / 2;
    let mut col = SecureColumnByCoords::<SimdBackend>::zeros(size);

    // TODO(andrew): Optimize.
    for i in 0..half_coset_size {
        let half_coset0_index = coset_index_to_circle_domain_index(i * 2, log_size);
        let half_coset1_index = coset_index_to_circle_domain_index(i * 2 + 1, log_size);
        let half_coset0_index_bit_rev = bit_reverse_index(half_coset0_index, log_size);
        let half_coset1_index_bit_rev = bit_reverse_index(half_coset1_index, log_size);

        let n_trailing_ones = i.trailing_ones() as usize;
        let half_coset0_carry_quotient = half_coset0_carry_quotients[n_trailing_ones];
        let half_coset1_carry_quotient = half_coset1_carry_quotients[n_trailing_ones];

        col.set(half_coset0_index_bit_rev, half_coset0_carry_quotient);
        col.set(half_coset1_index_bit_rev, half_coset1_carry_quotient);
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    SecureEvaluation::new(domain, col)
}

/// Evaluates the succinct Eq carry quotients column at point `p`.
///
/// See [`gen_carry_quotient_col`].
// TODO(andrew): Optimize further. Inline `eval_step_selector` and get runtime down to
// O(N_VARIABLES) vs current O(N_VARIABLES^2). Can also use vanishing evals to compute
// half_coset0_last half_coset1_first.
fn eval_carry_quotient_col<const N_VARIABLES: usize>(
    eval_point: &MleEvalPoint<N_VARIABLES>,
    p: CirclePoint<SecureField>,
) -> SecureField {
    let log_size = N_VARIABLES as u32;
    let coset = CanonicCoset::new(log_size).coset();

    let (half_coset0_carry_quotients, half_coset1_carry_quotients) =
        gen_half_coset_carry_quotients(eval_point);

    let mut eval = SecureField::zero();

    for variable_i in 0..N_VARIABLES.saturating_sub(1) {
        let log_step = variable_i as u32 + 2;
        let offset = (1 << (log_step - 1)) - 2;
        let half_coset0_selector = eval_step_selector_with_offset(coset, offset, log_step, p);
        let half_coset1_selector = eval_step_selector_with_offset(coset, offset + 1, log_step, p);
        let half_coset0_carry_quotient = half_coset0_carry_quotients[variable_i];
        let half_coset1_carry_quotient = half_coset1_carry_quotients[variable_i];
        eval += half_coset0_selector * half_coset0_carry_quotient;
        eval += half_coset1_selector * half_coset1_carry_quotient;
    }

    let half_coset0_last = eval_is_first(coset, p + coset.step.double().into_ef());
    let half_coset1_first = eval_is_first(coset, p + coset.step.into_ef());
    eval += *half_coset0_carry_quotients.last().unwrap() * half_coset0_last;
    eval += *half_coset1_carry_quotients.last().unwrap() * half_coset1_first;

    eval
}

/// Evaluates a polynomial that's `1` every `2^log_step` coset points, shifted by an offset, and `0`
/// elsewhere on coset.
fn eval_step_selector_with_offset(
    coset: Coset,
    offset: usize,
    log_step: u32,
    p: CirclePoint<SecureField>,
) -> SecureField {
    let offset_step = coset.step.mul(offset as u128);
    eval_step_selector(coset, log_step, p - offset_step.into_ef())
}

/// Evaluates a polynomial that's `1` every `2^log_step` coset points and `0` elsewhere on coset.
fn eval_step_selector(coset: Coset, log_step: u32, p: CirclePoint<SecureField>) -> SecureField {
    if log_step == 0 {
        return SecureField::one();
    }

    // Rotate the coset to have points on the `x` axis.
    let p = p - coset.initial.into_ef();
    let mut vanish_at_log_step = (0..coset.log_size)
        .scan(p, |p, _| {
            let res = *p;
            *p = p.double();
            Some(res.y)
        })
        .collect_vec();
    vanish_at_log_step.reverse();
    // We only need the first `log_step` many values.
    vanish_at_log_step.truncate(log_step as usize);
    let mut vanish_at_log_step_inv = vec![SecureField::zero(); vanish_at_log_step.len()];
    SecureField::batch_inverse(&vanish_at_log_step, &mut vanish_at_log_step_inv);

    let half_coset_selector_dbl = (vanish_at_log_step[0] * vanish_at_log_step_inv[1]).square();
    let vanish_substep_inv_sum = vanish_at_log_step_inv[1..].iter().sum::<SecureField>();
    (half_coset_selector_dbl + vanish_at_log_step[0] * vanish_substep_inv_sum.double())
        / BaseField::from(1 << (log_step + 1))
}

fn eval_is_first(coset: Coset, p: CirclePoint<SecureField>) -> SecureField {
    coset_vanishing(coset, p)
        / (point_vanishing(coset.initial, p) * BaseField::from(1 << coset.log_size))
}

/// Output of the form: `(half_coset0_carry_quotients, half_coset1_carry_quotients)`.
fn gen_half_coset_carry_quotients<const N_VARIABLES: usize>(
    eval_point: &MleEvalPoint<N_VARIABLES>,
) -> ([SecureField; N_VARIABLES], [SecureField; N_VARIABLES]) {
    let last_variable = *eval_point.p.last().unwrap();
    let mut half_coset0_carry_quotients = eval_point.eq_carry_quotients;
    *half_coset0_carry_quotients.last_mut().unwrap() *=
        eq(&[SecureField::one()], &[last_variable]) / eq(&[SecureField::zero()], &[last_variable]);
    let half_coset1_carry_quotients = half_coset0_carry_quotients.map(|v| v.inverse());
    (half_coset0_carry_quotients, half_coset1_carry_quotients)
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::iter::{repeat, zip};

    use itertools::{chain, zip_eq, Itertools};
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{
        eval_carry_quotient_col, eval_eq_constraints, eval_mle_eval_constraints,
        eval_prefix_sum_constraints, gen_carry_quotient_col, MleEvalPoint,
    };
    use crate::constraint_framework::constant_columns::{gen_is_first, gen_is_step_with_offset};
    use crate::constraint_framework::{assert_constraints, EvalAtRow};
    use crate::core::backend::simd::column::SecureColumn;
    use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
    use crate::core::backend::simd::qm31::PackedSecureField;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Col, Column};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;
    use crate::core::lookups::gkr_prover::GkrOps;
    use crate::core::lookups::mle::Mle;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
    use crate::core::poly::BitReversedOrder;
    use crate::core::utils::{bit_reverse, coset_order_to_circle_domain_order};
    use crate::examples::xor::gkr_lookups::mle_eval::eval_step_selector_with_offset;

    #[test]
    fn test_mle_eval_constraints_with_log_size_5() {
        const N_VARIABLES: usize = 5;
        const EVAL_TRACE: usize = 0;
        const CARRY_QUOTIENTS_TRACE: usize = 1;
        const CONST_TRACE: usize = 2;
        let mut rng = SmallRng::seed_from_u64(0);
        let log_size = N_VARIABLES as u32;
        let size = 1 << log_size;
        let mle = Mle::new((0..size).map(|_| rng.gen::<SecureField>()).collect());
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let mle_eval_point = MleEvalPoint::new(eval_point);
        let base_trace = gen_base_trace(&mle, &eval_point);
        let claim = mle.eval_at_point(&eval_point);
        let claim_shift = claim / BaseField::from(size);
        let carry_quotients_col = gen_carry_quotient_col(&mle_eval_point)
            .into_coordinate_evals()
            .to_vec();
        let constants_trace = gen_constants_trace::<N_VARIABLES>();
        let traces = TreeVec::new(vec![base_trace, carry_quotients_col, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(log_size);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let [mle_coeff_col_eval] = eval.next_extension_interaction_mask(EVAL_TRACE, [0]);
            let [carry_quotients_col_eval] =
                eval.next_extension_interaction_mask(CARRY_QUOTIENTS_TRACE, [0]);
            eval_mle_eval_constraints(
                EVAL_TRACE,
                CONST_TRACE,
                &mut eval,
                mle_coeff_col_eval,
                mle_eval_point,
                claim_shift,
                carry_quotients_col_eval,
            )
        });
    }

    #[test]
    #[ignore = "SimdBackend `MIN_FFT_LOG_SIZE` is 5"]
    fn eq_constraints_with_4_variables() {
        const N_VARIABLES: usize = 4;
        const EVAL_TRACE: usize = 0;
        const CARRY_QUOTIENTS_TRACE: usize = 1;
        const CONST_TRACE: usize = 2;
        let mut rng = SmallRng::seed_from_u64(0);
        let mle = Mle::new(repeat(SecureField::one()).take(1 << N_VARIABLES).collect());
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let mle_eval_point = MleEvalPoint::new(eval_point);
        let base_trace = gen_base_trace(&mle, &eval_point);
        let carry_quotients_col = gen_carry_quotient_col(&mle_eval_point)
            .into_coordinate_evals()
            .to_vec();
        let constants_trace = gen_constants_trace::<N_VARIABLES>();
        let traces = TreeVec::new(vec![base_trace, carry_quotients_col, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let _mle_coeffs_col_eval = eval.next_extension_interaction_mask(EVAL_TRACE, [0]);
            let [carry_quotients_col_eval] =
                eval.next_extension_interaction_mask(CARRY_QUOTIENTS_TRACE, [0]);
            eval_eq_constraints(
                EVAL_TRACE,
                CONST_TRACE,
                &mut eval,
                mle_eval_point,
                carry_quotients_col_eval,
            );
        });
    }

    #[test]
    fn eq_constraints_with_5_variables() {
        const N_VARIABLES: usize = 5;
        const EVAL_TRACE: usize = 0;
        const CARRY_QUOTIENTS_TRACE: usize = 1;
        const CONST_TRACE: usize = 2;
        let mut rng = SmallRng::seed_from_u64(0);
        let mle = Mle::new(repeat(SecureField::one()).take(1 << N_VARIABLES).collect());
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let mle_eval_point = MleEvalPoint::new(eval_point);
        let base_trace = gen_base_trace(&mle, &eval_point);
        let carry_quotients_col = gen_carry_quotient_col(&mle_eval_point)
            .into_coordinate_evals()
            .to_vec();
        let constants_trace = gen_constants_trace::<N_VARIABLES>();
        let traces = TreeVec::new(vec![base_trace, carry_quotients_col, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let _mle_coeffs_col_eval = eval.next_extension_interaction_mask(EVAL_TRACE, [0]);
            let [carry_quotients_col_eval] =
                eval.next_extension_interaction_mask(CARRY_QUOTIENTS_TRACE, [0]);
            eval_eq_constraints(
                EVAL_TRACE,
                CONST_TRACE,
                &mut eval,
                mle_eval_point,
                carry_quotients_col_eval,
            );
        });
    }

    #[test]
    fn eq_constraints_with_8_variables() {
        const N_VARIABLES: usize = 8;
        const EVAL_TRACE: usize = 0;
        const CARRY_QUOTIENTS_TRACE: usize = 1;
        const CONST_TRACE: usize = 2;
        let mut rng = SmallRng::seed_from_u64(0);
        let mle = Mle::new(repeat(SecureField::one()).take(1 << N_VARIABLES).collect());
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let mle_eval_point = MleEvalPoint::new(eval_point);
        let base_trace = gen_base_trace(&mle, &eval_point);
        let carry_quotients_col = gen_carry_quotient_col(&mle_eval_point)
            .into_coordinate_evals()
            .to_vec();
        let constants_trace = gen_constants_trace::<N_VARIABLES>();
        let traces = TreeVec::new(vec![base_trace, carry_quotients_col, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(eval_point.len() as u32);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let _mle_coeffs_col_eval = eval.next_extension_interaction_mask(EVAL_TRACE, [0]);
            let [carry_quotients_col_eval] =
                eval.next_extension_interaction_mask(CARRY_QUOTIENTS_TRACE, [0]);
            eval_eq_constraints(
                EVAL_TRACE,
                CONST_TRACE,
                &mut eval,
                mle_eval_point,
                carry_quotients_col_eval,
            );
        });
    }

    #[test]
    fn inclusive_prefix_sum_constraints_with_log_size_5() {
        const LOG_SIZE: u32 = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let vals = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let cumulative_sum = vals.iter().sum::<SecureField>();
        let cumulative_sum_shift = cumulative_sum / BaseField::from(vals.len());
        let trace = TreeVec::new(vec![gen_prefix_sum_trace(vals)]);
        let trace_polys = trace.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(LOG_SIZE);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let [row_diff] = eval.next_extension_interaction_mask(0, [0]);
            eval_prefix_sum_constraints(0, &mut eval, row_diff, cumulative_sum_shift)
        });
    }

    #[test]
    fn eval_step_selector_with_offset_works() {
        const LOG_SIZE: u32 = 5;
        const OFFSET: usize = 1;
        const LOG_STEP: u32 = 2;
        let coset = CanonicCoset::new(LOG_SIZE).coset();
        let col_eval = gen_is_step_with_offset::<SimdBackend>(LOG_SIZE, LOG_STEP, OFFSET);
        let col_poly = col_eval.interpolate();
        let p = SECURE_FIELD_CIRCLE_GEN;

        let eval = eval_step_selector_with_offset(coset, OFFSET, LOG_STEP, p);

        assert_eq!(eval, col_poly.eval_at_point(p));
    }

    #[test]
    fn eval_carry_quotient_col_works() {
        const N_VARIABLES: usize = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let mle_eval_point = MleEvalPoint::<N_VARIABLES>::new(array::from_fn(|_| rng.gen()));
        let col_eval = gen_carry_quotient_col(&mle_eval_point);
        let twiddles = SimdBackend::precompute_twiddles(col_eval.domain.half_coset);
        let col_poly = col_eval.interpolate_with_twiddles(&twiddles);
        let p = SECURE_FIELD_CIRCLE_GEN;

        let eval = eval_carry_quotient_col(&mle_eval_point, p);

        assert_eq!(eval, col_poly.eval_at_point(p));
    }

    /// Generates a trace.
    ///
    /// Trace structure:
    ///
    /// ```text
    /// -------------------------------------------------------------------------------------
    /// |         MLE coeffs        |       EqEvals (basis)     |   MLE terms (prefix sum)  |
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

    /// Generates a trace.
    ///
    /// Trace structure:
    ///
    /// ```text
    /// ---------------------------------------------------------
    /// |           Values          |     Values prefix sum     |
    /// ---------------------------------------------------------
    /// |  c0  |  c1  |  c2  |  c3  |  c4  |  c5  |  c6  |  c7  |
    /// ---------------------------------------------------------
    /// ```
    fn gen_prefix_sum_trace(
        values: Vec<SecureField>,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        assert!(values.len().is_power_of_two());

        let vals_circle_domain_order = coset_order_to_circle_domain_order(&values);
        let mut vals_bit_rev_circle_domain_order = vals_circle_domain_order;
        bit_reverse(&mut vals_bit_rev_circle_domain_order);
        let vals_secure_col: SecureColumnByCoords<SimdBackend> =
            vals_bit_rev_circle_domain_order.into_iter().collect();
        let vals_cols = vals_secure_col.columns;

        let cumulative_sum = values.iter().sum::<SecureField>();
        let cumulative_sum_shift = cumulative_sum / BaseField::from(values.len());
        let packed_cumulative_sum_shift = PackedSecureField::broadcast(cumulative_sum_shift);
        let packed_shifts = packed_cumulative_sum_shift.into_packed_m31s();
        let mut shifted_cols = vals_cols.clone();
        zip(&mut shifted_cols, packed_shifts)
            .for_each(|(col, packed_shift)| col.data.iter_mut().for_each(|v| *v -= packed_shift));
        let shifted_prefix_sum_cols = shifted_cols.map(inclusive_prefix_sum);

        let log_size = values.len().ilog2();
        let trace_domain = CanonicCoset::new(log_size).circle_domain();

        chain![vals_cols, shifted_prefix_sum_cols]
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

    fn gen_constants_trace<const N_VARIABLES: usize>(
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let log_size = N_VARIABLES as u32;
        vec![gen_is_first(log_size)]
    }
}
