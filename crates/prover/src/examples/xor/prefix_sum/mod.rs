#![allow(dead_code)]

use std::iter::zip;
use std::ops::AddAssign;

use itertools::izip;
use num_traits::{One, Zero};

use crate::constraint_framework::EvalAtRow;
use crate::core::backend::simd::m31::{PackedBaseField, N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{
    bit_reverse, circle_domain_order_to_coset_order, coset_order_to_circle_domain_order,
};

/// Inclusive prefix sum constraint.
///
/// Note the prefix sum is index inclusive i.e. `prefix_sum([3, 4, 5, 1]) = [3, 7, 12, 13]`.
///
/// Checks:
/// - First row equals `claimed_row_diff`
/// - All rows (except first) equal `prev_sum + claimed_row_diff`
/// - Last row equals `claimed_sum`
pub fn prefix_sum_check<E: EvalAtRow>(
    eval: &mut E,
    row_diff: E::EF,
    final_sum: SecureField,
    at: &PrefixSumMaskAt<E>,
    is: &PrefixSumMaskIs<E>,
) {
    let is_not_first = E::F::from(BaseField::one()) - is.first;

    // TODO: Why won't this work?
    // `eval.add_constraint(at_curr - (at_prev * is_not_first + row_diff) +
    // (row_diff - final_sum) * is_last);`
    eval.add_constraint((at.curr - row_diff) * is.first);
    eval.add_constraint((at.curr - (at.prev + row_diff)) * is_not_first);
    eval.add_constraint((at.curr - final_sum) * is.last);
}

#[derive(Debug, Clone, Copy)]
pub struct PrefixSumMaskAt<E: EvalAtRow> {
    pub curr: E::EF,
    pub prev: E::EF,
}

impl<E: EvalAtRow> PrefixSumMaskAt<E> {
    pub fn draw<const TRACE: usize>(eval: &mut E) -> Self {
        let [curr, prev] = eval.next_extension_interaction_mask(TRACE, [0, -1]);
        Self { curr, prev }
    }
}

pub struct PrefixSumMaskIs<E: EvalAtRow> {
    pub first: E::F,
    pub last: E::F,
}

impl<E: EvalAtRow> PrefixSumMaskIs<E> {
    pub fn draw<const TRACE: usize>(eval: &mut E) -> Self {
        let [first, last] = eval.next_interaction_mask(TRACE, [0, 1]);
        Self { first, last }
    }
}

/// Returns the prefix sum.
///
/// The prefix sum is index inclusive i.e. `prefix_sum([3, 4, 5, 1]) = [3, 7, 12, 13]`.
pub fn inclusive_prefix_sum<F: Field>(v: &[F]) -> Vec<F> {
    v.iter()
        .scan(F::zero(), |acc, &v| {
            *acc += v;
            Some(*acc)
        })
        .collect()
}

/// Performs a exclusive prefix sum on values in `Coset` order when provided
/// with evaluations in `CircleDomain` order.
///
/// The final sum replaces the prefix sum at index `0` (i.e. essentially returns an
/// inclusive prefix sum rotated right by 1).
///
/// Based on parallel Blelloch prefix sum:
/// <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>
pub fn exclusive_prefix_sum_simd(
    eval: CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    if eval.len() < N_LANES * 4 {
        return exclusive_prefix_sum_slow(eval);
    }

    let domain = eval.domain;
    let mut res = eval.values;
    let packed_len = res.data.len();

    let (l_half, r_half) = res.data.split_at_mut(packed_len / 2);

    // Up Sweep
    // ========
    // Handle the first two up sweep rounds manually.
    // Required due different ordering of `CircleDomain` and `Coset`.
    // Evaluations are provided in bit-reversed `CircleDomain` order.
    for ([l0, l1], [r0, r1]) in izip!(l_half.array_chunks_mut(), r_half.array_chunks_mut().rev()) {
        let (half_coset0_lo, half_coset1_hi_rev) = l0.deinterleave(*l1);
        let mut half_coset1_hi = half_coset1_hi_rev.reverse();
        let (half_coset0_hi, half_coset1_lo_rev) = r0.deinterleave(*r1);
        let mut half_coset1_lo = half_coset1_lo_rev.reverse();
        up_sweep_val(half_coset0_lo, &mut half_coset1_lo);
        up_sweep_val(half_coset0_hi, &mut half_coset1_hi);
        *l0 = half_coset0_lo;
        *l1 = half_coset0_hi;
        *r0 = half_coset1_lo;
        *r1 = half_coset1_hi;
    }
    let half_coset1_sums: &mut [PackedBaseField] = r_half;
    for i in 0..half_coset1_sums.len() / 2 {
        let lo_index = half_coset1_sums.len() - 2 - i * 2;
        let hi_index = i * 2 + 1;
        up_sweep_val(half_coset1_sums[lo_index], &mut half_coset1_sums[hi_index])
    }
    // Handle remaining up sweep rounds.
    let mut chunk_size = half_coset1_sums.len() / 2;
    while chunk_size > 1 {
        let len = half_coset1_sums.len();
        let (lows, highs) = half_coset1_sums[len - chunk_size * 2..].split_at_mut(chunk_size);
        zip(lows.array_chunks(), highs.array_chunks_mut())
            .for_each(|([_, lo], [_, hi])| up_sweep_val(*lo, hi));
        chunk_size /= 2;
    }
    let (l_half, r_half) = res.data.split_at_mut(packed_len / 2);
    let half_coset1_sums: &mut [PackedBaseField] = r_half;
    // Up sweep the last SIMD vector.
    let mut last_vec = half_coset1_sums.last().unwrap().to_array();
    let mut chunk_size = last_vec.len() / 2;
    while chunk_size > 0 {
        let len = last_vec.len();
        let (lows, highs) = last_vec[len - chunk_size * 2..].split_at_mut(chunk_size);
        // let (highs, lows) = last_vec.split_at_mut(chunk_size);
        zip(lows, highs).for_each(|(lo, hi)| up_sweep_val(*lo, hi));
        chunk_size /= 2;
    }

    // Extract the final sum.
    let last = last_vec.last_mut().unwrap();
    let final_sum = *last;
    // Prepare for the down sweep.
    *last = BaseField::zero();

    // Down Sweep
    // ==========
    // Down sweep the last SIMD vector.
    let mut chunk_size = 1;
    while chunk_size < last_vec.len() {
        let len = last_vec.len();
        let (lows, highs) = last_vec[len - chunk_size * 2..].split_at_mut(chunk_size);
        zip(lows, highs).for_each(|(lo, hi)| down_sweep_val(lo, hi));
        chunk_size *= 2;
    }
    // Re-insert the SIMD vector.
    *half_coset1_sums.last_mut().unwrap() = last_vec.into();
    // Handle remaining down sweep rounds (except first two).
    let mut chunk_size = 2;
    while chunk_size < half_coset1_sums.len() {
        let len = half_coset1_sums.len();
        let (lows, highs) = half_coset1_sums[len - chunk_size * 2..].split_at_mut(chunk_size);
        zip(lows.array_chunks_mut(), highs.array_chunks_mut())
            .for_each(|([_, lo], [_, hi])| down_sweep_val(lo, hi));
        chunk_size *= 2;
    }
    // Handle last two down sweep rounds manually.
    // Required due different ordering of `CircleDomain` and `Coset`.
    // Evaluations must be returned in bit-reversed `CircleDomain` order.
    for i in 0..half_coset1_sums.len() / 2 {
        let lo_index = half_coset1_sums.len() - 2 - i * 2;
        let hi_index = i * 2 + 1;
        let (mut lo, mut hi) = (half_coset1_sums[lo_index], half_coset1_sums[hi_index]);
        down_sweep_val(&mut lo, &mut hi);
        (half_coset1_sums[lo_index], half_coset1_sums[hi_index]) = (lo, hi);
    }
    for ([l0, l1], [r0, r1]) in izip!(l_half.array_chunks_mut(), r_half.array_chunks_mut().rev()) {
        let mut half_coset0_lo = *l0;
        let mut half_coset1_lo = *r0;
        down_sweep_val(&mut half_coset0_lo, &mut half_coset1_lo);
        let mut half_coset0_hi = *l1;
        let mut half_coset1_hi = *r1;
        down_sweep_val(&mut half_coset0_hi, &mut half_coset1_hi);
        (*l0, *l1) = half_coset0_lo.interleave(half_coset1_hi.reverse());
        (*r0, *r1) = half_coset0_hi.interleave(half_coset1_lo.reverse());
    }

    // Re-insert the final sum in position 0.
    let mut first_vec = res.data.first().unwrap().to_array();
    first_vec[0] = final_sum;
    *res.data.first_mut().unwrap() = PackedBaseField::from_array(first_vec);

    CircleEvaluation::new(domain, res)
}

fn up_sweep_val<F: AddAssign + Copy>(lo: F, hi: &mut F) {
    *hi += lo;
}

fn down_sweep_val<F: AddAssign + Copy>(lo: &mut F, hi: &mut F) {
    let tmp = *hi;
    *hi += *lo;
    *lo = tmp;
}

fn exclusive_prefix_sum_slow(
    eval: CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let domain = eval.domain;

    // Obtain values in coset order.
    let mut coset_order_eval = eval.values.into_cpu_vec();
    bit_reverse(&mut coset_order_eval);
    coset_order_eval = circle_domain_order_to_coset_order(&coset_order_eval);

    let mut coset_order_prefix_sum = Vec::new();
    let mut sum = BaseField::zero();

    for v in coset_order_eval {
        coset_order_prefix_sum.push(sum);
        sum += v;
    }

    // Insert the final sum.
    coset_order_prefix_sum[0] = sum;

    let mut circle_domain_order_eval = coset_order_to_circle_domain_order(&coset_order_prefix_sum);
    bit_reverse(&mut circle_domain_order_eval);
    CircleEvaluation::new(domain, circle_domain_order_eval.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use super::{
        exclusive_prefix_sum_simd, inclusive_prefix_sum, prefix_sum_check, PrefixSumMaskAt,
        PrefixSumMaskIs,
    };
    use crate::constraint_framework::constant_cols::gen_is_first;
    use crate::constraint_framework::{assert_constraints, EvalAtRow};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::Column;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::core::utils::{
        bit_reverse, circle_domain_order_to_coset_order, coset_order_to_circle_domain_order,
    };

    const SUM_TRACE: usize = 0;
    const CONST_TRACE: usize = 1;

    #[test]
    fn test_prefix_sum_constraints_with_log_size_5() {
        const LOG_SIZE: u32 = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let vals = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let prefix_sum = inclusive_prefix_sum(&vals);
        let claim = vals.iter().sum();
        assert_eq!(prefix_sum.last(), Some(&claim));
        let base_trace = gen_base_trace(vals, prefix_sum);
        let constants_trace = gen_constants_trace(LOG_SIZE);
        let traces = TreeVec::new(vec![base_trace, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(LOG_SIZE);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let [row_diff] = eval.next_extension_interaction_mask(SUM_TRACE, [0]);
            let at_mask = PrefixSumMaskAt::draw::<SUM_TRACE>(&mut eval);
            let is_mask = PrefixSumMaskIs::draw::<CONST_TRACE>(&mut eval);
            prefix_sum_check(&mut eval, row_diff, claim, &at_mask, &is_mask);
        });
    }

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_3_works() {
        const LOG_N: u32 = 3;
        let mut rng = SmallRng::seed_from_u64(0);
        let domain = CanonicCoset::new(LOG_N).circle_domain();
        let evals = CircleEvaluation::new(domain, (0..1 << LOG_N).map(|_| rng.gen()).collect());
        let expected = exclusive_prefix_sum_ground_truth(&evals);

        let res = exclusive_prefix_sum_simd(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_6_works() {
        const LOG_N: u32 = 6;
        let mut rng = SmallRng::seed_from_u64(0);
        let domain = CanonicCoset::new(LOG_N).circle_domain();
        let evals = CircleEvaluation::new(domain, (0..1 << LOG_N).map(|_| rng.gen()).collect());
        let expected = exclusive_prefix_sum_ground_truth(&evals);

        let res = exclusive_prefix_sum_simd(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_8_works() {
        const LOG_N: u32 = 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let domain = CanonicCoset::new(LOG_N).circle_domain();
        let evals = CircleEvaluation::new(domain, (0..1 << LOG_N).map(|_| rng.gen()).collect());
        let expected = exclusive_prefix_sum_ground_truth(&evals);

        let res = exclusive_prefix_sum_simd(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }

    fn exclusive_prefix_sum_ground_truth(
        eval: &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        let mut circle_domain_order_evals = eval.to_cpu();
        bit_reverse(&mut circle_domain_order_evals);
        let coset_order_evals = circle_domain_order_to_coset_order(&circle_domain_order_evals);
        let mut coset_order_prefix_sum = inclusive_prefix_sum(&coset_order_evals);
        coset_order_prefix_sum.rotate_right(1);
        let mut circle_domain_order_sum =
            coset_order_to_circle_domain_order(&coset_order_prefix_sum);
        bit_reverse(&mut circle_domain_order_sum);
        CircleEvaluation::new(eval.domain, circle_domain_order_sum.into_iter().collect())
    }

    /// Generates a trace.
    ///
    /// Trace structure:
    ///
    /// ```text
    /// ---------------------------------------------------------
    /// |           Values          |     Values prefix sum     |
    /// ---------------------------------------------------------
    /// |  c0  |  c1  |  c2  |  c3  |  c0  |  c1  |  c2  |  c3  |
    /// ---------------------------------------------------------
    /// ```
    fn gen_base_trace(
        vals: Vec<SecureField>,
        prefix_sum: Vec<SecureField>,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        assert!(vals.len().is_power_of_two());
        assert_eq!(vals.len(), prefix_sum.len());

        let log_size = vals.len().ilog2();

        let vals_circle_domain_order = coset_order_to_circle_domain_order(&vals);
        let mut vals_bit_rev_circle_domain_order = vals_circle_domain_order;
        bit_reverse(&mut vals_bit_rev_circle_domain_order);

        let prefix_sum_circle_domain_order = coset_order_to_circle_domain_order(&prefix_sum);
        let mut prefix_sum_bit_rev_circle_domain_order = prefix_sum_circle_domain_order;
        bit_reverse(&mut prefix_sum_bit_rev_circle_domain_order);

        let mut vals_evals_col0 = Vec::new();
        let mut vals_evals_col1 = Vec::new();
        let mut vals_evals_col2 = Vec::new();
        let mut vals_evals_col3 = Vec::new();

        for v in vals_bit_rev_circle_domain_order {
            let [v0, v1, v2, v3] = v.to_m31_array();
            vals_evals_col0.push(v0);
            vals_evals_col1.push(v1);
            vals_evals_col2.push(v2);
            vals_evals_col3.push(v3);
        }

        let mut prefix_sum_col0 = Vec::new();
        let mut prefix_sum_col1 = Vec::new();
        let mut prefix_sum_col2 = Vec::new();
        let mut prefix_sum_col3 = Vec::new();

        for v in prefix_sum_bit_rev_circle_domain_order {
            let [v0, v1, v2, v3] = v.to_m31_array();
            prefix_sum_col0.push(v0);
            prefix_sum_col1.push(v1);
            prefix_sum_col2.push(v2);
            prefix_sum_col3.push(v3);
        }

        let trace_domain = CanonicCoset::new(log_size).circle_domain();

        vec![
            CircleEvaluation::new(trace_domain, vals_evals_col0.into_iter().collect()),
            CircleEvaluation::new(trace_domain, vals_evals_col1.into_iter().collect()),
            CircleEvaluation::new(trace_domain, vals_evals_col2.into_iter().collect()),
            CircleEvaluation::new(trace_domain, vals_evals_col3.into_iter().collect()),
            CircleEvaluation::new(trace_domain, prefix_sum_col0.into_iter().collect()),
            CircleEvaluation::new(trace_domain, prefix_sum_col1.into_iter().collect()),
            CircleEvaluation::new(trace_domain, prefix_sum_col2.into_iter().collect()),
            CircleEvaluation::new(trace_domain, prefix_sum_col3.into_iter().collect()),
        ]
    }

    /// Generates a single column trace. Column is `1` at the first trace point and `0` on all
    /// other trace points.
    fn gen_constants_trace(
        log_size: u32,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        vec![gen_is_first(log_size)]
    }
}
