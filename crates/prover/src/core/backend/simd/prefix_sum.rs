use std::iter::zip;
use std::ops::{AddAssign, Sub};

use itertools::{izip, Itertools};
use num_traits::Zero;

use crate::core::backend::cpu::bit_reverse;
use crate::core::backend::simd::m31::{PackedBaseField, N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::utils::{circle_domain_order_to_coset_order, coset_order_to_circle_domain_order};

/// Performs a inclusive prefix sum on values in `Coset` order when provided
/// with evaluations in bit-reversed `CircleDomain` order.
///
/// Based on parallel Blelloch prefix sum:
/// <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>
pub fn inclusive_prefix_sum(
    bit_rev_circle_domain_evals: Col<SimdBackend, BaseField>,
) -> Col<SimdBackend, BaseField> {
    if bit_rev_circle_domain_evals.len() < N_LANES * 4 {
        return inclusive_prefix_sum_slow(bit_rev_circle_domain_evals);
    }

    let mut res = bit_rev_circle_domain_evals;
    let packed_len = res.data.len();
    let (l_half, r_half) = res.data.split_at_mut(packed_len / 2);

    // Up Sweep
    // ========
    // Handle the first two up sweep rounds manually.
    // Required due different ordering of `CircleDomain` and `Coset`.
    // Evaluations are provided in bit-reversed `CircleDomain` order.
    for ([l0, l1], [r0, r1]) in izip!(l_half.array_chunks_mut(), r_half.array_chunks_mut().rev()) {
        let (mut half_coset0_lo, half_coset1_hi_rev) = l0.deinterleave(*l1);
        let half_coset1_hi = half_coset1_hi_rev.reverse();
        let (mut half_coset0_hi, half_coset1_lo_rev) = r0.deinterleave(*r1);
        let half_coset1_lo = half_coset1_lo_rev.reverse();
        up_sweep_val(&mut half_coset0_lo, half_coset1_lo);
        up_sweep_val(&mut half_coset0_hi, half_coset1_hi);
        *l0 = half_coset0_lo;
        *l1 = half_coset0_hi;
        *r0 = half_coset1_lo;
        *r1 = half_coset1_hi;
    }
    let half_coset0_sums: &mut [PackedBaseField] = l_half;
    for i in 0..half_coset0_sums.len() / 2 {
        let lo_index = i * 2;
        let hi_index = half_coset0_sums.len() - 1 - i * 2;
        let hi = half_coset0_sums[hi_index];
        up_sweep_val(&mut half_coset0_sums[lo_index], hi)
    }
    // Handle remaining up sweep rounds.
    let mut chunk_size = half_coset0_sums.len() / 2;
    while chunk_size > 1 {
        let (lows, highs) = half_coset0_sums.split_at_mut(chunk_size);
        zip(lows.array_chunks_mut(), highs.array_chunks())
            .for_each(|([lo, _], [hi, _])| up_sweep_val(lo, *hi));
        chunk_size /= 2;
    }
    // Up sweep the last SIMD vector.
    let mut first_vec = half_coset0_sums.first().unwrap().to_array();
    let mut chunk_size = first_vec.len() / 2;
    while chunk_size > 0 {
        let (lows, highs) = first_vec.split_at_mut(chunk_size);
        zip(lows, highs).for_each(|(lo, hi)| up_sweep_val(lo, *hi));
        chunk_size /= 2;
    }

    // Down Sweep
    // ==========
    // Down sweep the last SIMD vector.
    let mut chunk_size = 1;
    while chunk_size < first_vec.len() {
        let (lows, highs) = first_vec.split_at_mut(chunk_size);
        zip(lows, highs).for_each(|(lo, hi)| down_sweep_val(lo, hi));
        chunk_size *= 2;
    }
    // Re-insert the SIMD vector.
    *half_coset0_sums.first_mut().unwrap() = first_vec.into();
    // Handle remaining down sweep rounds (except first two).
    let mut chunk_size = 2;
    while chunk_size < half_coset0_sums.len() {
        let (lows, highs) = half_coset0_sums.split_at_mut(chunk_size);
        zip(lows.array_chunks_mut(), highs.array_chunks_mut())
            .for_each(|([lo, _], [hi, _])| down_sweep_val(lo, hi));
        chunk_size *= 2;
    }
    // Handle last two down sweep rounds manually.
    // Required due different ordering of `CircleDomain` and `Coset`.
    // Evaluations must be returned in bit-reversed `CircleDomain` order.
    for i in 0..half_coset0_sums.len() / 2 {
        let lo_index = i * 2;
        let hi_index = half_coset0_sums.len() - 1 - i * 2;
        let (mut lo, mut hi) = (half_coset0_sums[lo_index], half_coset0_sums[hi_index]);
        down_sweep_val(&mut lo, &mut hi);
        (half_coset0_sums[lo_index], half_coset0_sums[hi_index]) = (lo, hi);
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

    res
}

fn up_sweep_val<F: AddAssign + Copy>(lo: &mut F, hi: F) {
    *lo += hi;
}

fn down_sweep_val<F: Sub<Output = F> + Copy>(lo: &mut F, hi: &mut F) {
    (*lo, *hi) = (*lo - *hi, *lo)
}

fn inclusive_prefix_sum_slow(
    bit_rev_circle_domain_evals: Col<SimdBackend, BaseField>,
) -> Col<SimdBackend, BaseField> {
    // Obtain values in coset order.
    let mut coset_order_eval = bit_rev_circle_domain_evals.into_cpu_vec();
    bit_reverse(&mut coset_order_eval);
    coset_order_eval = circle_domain_order_to_coset_order(&coset_order_eval);
    let coset_order_prefix_sum = coset_order_eval
        .into_iter()
        .scan(BaseField::zero(), |acc, v| {
            *acc += v;
            Some(*acc)
        })
        .collect_vec();
    let mut circle_domain_order_eval = coset_order_to_circle_domain_order(&coset_order_prefix_sum);
    bit_reverse(&mut circle_domain_order_eval);
    circle_domain_order_eval.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use super::inclusive_prefix_sum;
    use crate::core::backend::simd::column::BaseColumn;
    use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum_slow;
    use crate::core::backend::Column;

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_3_works() {
        const LOG_N: u32 = 3;
        let mut rng = SmallRng::seed_from_u64(0);
        let evals: BaseColumn = (0..1 << LOG_N).map(|_| rng.gen()).collect();
        let expected = inclusive_prefix_sum_slow(evals.clone());

        let res = inclusive_prefix_sum(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_6_works() {
        const LOG_N: u32 = 6;
        let mut rng = SmallRng::seed_from_u64(0);
        let evals: BaseColumn = (0..1 << LOG_N).map(|_| rng.gen()).collect();
        let expected = inclusive_prefix_sum_slow(evals.clone());

        let res = inclusive_prefix_sum(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_8_works() {
        const LOG_N: u32 = 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let evals: BaseColumn = (0..1 << LOG_N).map(|_| rng.gen()).collect();
        let expected = inclusive_prefix_sum_slow(evals.clone());

        let res = inclusive_prefix_sum(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }
}
