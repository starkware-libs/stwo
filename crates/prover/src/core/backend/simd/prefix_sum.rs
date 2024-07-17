use std::iter::zip;
use std::ops::AddAssign;

use itertools::izip;
use num_traits::Zero;

use crate::core::backend::simd::m31::{PackedBaseField, N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::utils::{
    bit_reverse, circle_domain_order_to_coset_order, coset_order_to_circle_domain_order,
};

/// Performs a exclusive prefix sum on values in `Coset` order when provided
/// with evaluations in bit-reversed `CircleDomain` order.
///
/// The final sum replaces the prefix sum at index `0` (i.e. essentially returns an
/// inclusive prefix sum rotated right by 1).
///
/// Based on parallel Blelloch prefix sum:
/// <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>
pub fn exclusive_prefix_sum_simd(
    bit_rev_circle_domain_evals: Col<SimdBackend, BaseField>,
) -> Col<SimdBackend, BaseField> {
    if bit_rev_circle_domain_evals.len() < N_LANES * 4 {
        return exclusive_prefix_sum_slow(bit_rev_circle_domain_evals);
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

    res
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
    bit_rev_circle_domain_evals: Col<SimdBackend, BaseField>,
) -> Col<SimdBackend, BaseField> {
    // Obtain values in coset order.
    let mut coset_order_eval = bit_rev_circle_domain_evals.into_cpu_vec();
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
    circle_domain_order_eval.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use super::exclusive_prefix_sum_simd;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Col, Column};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::Field;
    use crate::core::utils::{
        bit_reverse, circle_domain_order_to_coset_order, coset_order_to_circle_domain_order,
    };

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_3_works() {
        const LOG_N: u32 = 3;
        let mut rng = SmallRng::seed_from_u64(0);
        let evals = (0..1 << LOG_N).map(|_| rng.gen()).collect();
        let expected = exclusive_prefix_sum_ground_truth(&evals);

        let res = exclusive_prefix_sum_simd(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_6_works() {
        const LOG_N: u32 = 6;
        let mut rng = SmallRng::seed_from_u64(0);
        let evals = (0..1 << LOG_N).map(|_| rng.gen()).collect();
        let expected = exclusive_prefix_sum_ground_truth(&evals);

        let res = exclusive_prefix_sum_simd(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }

    #[test]
    fn exclusive_prefix_sum_simd_with_log_size_8_works() {
        const LOG_N: u32 = 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let evals = (0..1 << LOG_N).map(|_| rng.gen()).collect();
        let expected = exclusive_prefix_sum_ground_truth(&evals);

        let res = exclusive_prefix_sum_simd(evals);

        assert_eq!(res.to_cpu(), expected.to_cpu());
    }

    fn exclusive_prefix_sum_ground_truth(
        bit_rev_circle_domain_evals: &Col<SimdBackend, BaseField>,
    ) -> Col<SimdBackend, BaseField> {
        let mut circle_domain_order_evals = bit_rev_circle_domain_evals.to_cpu();
        bit_reverse(&mut circle_domain_order_evals);
        let coset_order_evals = circle_domain_order_to_coset_order(&circle_domain_order_evals);
        let mut coset_order_prefix_sum = inclusive_prefix_sum(&coset_order_evals);
        coset_order_prefix_sum.rotate_right(1);
        let mut circle_domain_order_sum =
            coset_order_to_circle_domain_order(&coset_order_prefix_sum);
        bit_reverse(&mut circle_domain_order_sum);
        circle_domain_order_sum.into_iter().collect()
    }

    fn inclusive_prefix_sum<F: Field>(v: &[F]) -> Vec<F> {
        v.iter()
            .scan(F::zero(), |acc, &v| {
                *acc += v;
                Some(*acc)
            })
            .collect()
    }
}
