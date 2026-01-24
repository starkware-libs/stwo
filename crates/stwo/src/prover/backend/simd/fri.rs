use std::array;
use std::simd::{u32x16, u32x8};

use num_traits::Zero;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

use super::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use super::SimdBackend;
use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::line::LineDomain;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::prover::backend::cpu::{fold_circle_into_line_cpu, fold_line_cpu};
use crate::prover::backend::simd::fft::compute_first_twiddles;
use crate::prover::backend::simd::fft::ifft::simd_ibutterfly;
use crate::prover::backend::simd::qm31::PackedSecureField;
use crate::prover::backend::Column;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

pub const FOLD_LINE_CHUNK_SIZE: usize = 1 << 7;
pub const FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE: usize = 1 << 7;
impl FriOps for SimdBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let log_size = eval.len().ilog2();
        if log_size <= LOG_N_LANES {
            let eval = fold_line_cpu(&eval.to_cpu(), alpha);
            return LineEvaluation::new(eval.domain(), eval.values.into_iter().collect());
        }

        let domain = eval.domain();
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        let mut folded_values = unsafe { SecureColumnByCoords::uninitialized(1 << (log_size - 1)) };

        folded_values
            .par_chunks_mut(FOLD_LINE_CHUNK_SIZE)
            .zip_eq(eval.values.par_chunks(2 * FOLD_LINE_CHUNK_SIZE))
            .zip_eq(itwiddles.par_chunks(16 * FOLD_LINE_CHUNK_SIZE))
            .for_each(|((mut dst_chunk, src_chunk), itwiddles_chunk)| {
                for i in 0..dst_chunk.len() {
                    let value = unsafe {
                        // The 16 twiddles of the circle domain can be derived from the 8 twiddles
                        // of the next line domain. See `compute_first_twiddles()`.
                        let twiddle_dbl = u32x16::from_array(array::from_fn(|j| {
                            *itwiddles_chunk.get_unchecked(i * 16 + j)
                        }));
                        let val0 = src_chunk.packed_at(2 * i).into_packed_m31s();
                        let val1 = src_chunk.packed_at(2 * i + 1).into_packed_m31s();
                        let pairs: [_; 4] = array::from_fn(|j| {
                            let (a, b) = val0[j].deinterleave(val1[j]);
                            simd_ibutterfly(a, b, twiddle_dbl)
                        });
                        let val0 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|j| pairs[j].0));
                        let val1 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|j| pairs[j].1));
                        val0 + PackedSecureField::broadcast(alpha) * val1
                    };

                    unsafe {
                        dst_chunk.set_packed(i, value);
                    }
                }
            });

        LineEvaluation::new(domain.double(), folded_values)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let log_size = src.len().ilog2();
        if log_size <= LOG_N_LANES {
            // Fall back to CPU implementation.
            let mut cpu_dst = dst.to_cpu();
            fold_circle_into_line_cpu(&mut cpu_dst, &src.to_cpu(), alpha);
            *dst = LineEvaluation::new(
                cpu_dst.domain(),
                SecureColumnByCoords::from_cpu(cpu_dst.values),
            );
            return;
        }

        let domain = src.domain;
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        dst.values
            .par_chunks_mut(FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE)
            .zip_eq(src.values.par_chunks(2 * FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE))
            .zip_eq(itwiddles.par_chunks(8 * FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE))
            .for_each(|((mut dst_chunk, src_chunk), itwiddles_chunk)| {
                for i in 0..dst_chunk.len() {
                    let value = unsafe {
                        // The 16 twiddles of the circle domain can be derived from the 8 twiddles
                        // of the next line domain. See `compute_first_twiddles()`.
                        let twiddle_dbl = u32x8::from_array(array::from_fn(|j| {
                            *itwiddles_chunk.get_unchecked(i * 8 + j)
                        }));
                        let (t0, _) = compute_first_twiddles(twiddle_dbl);
                        let val0 = src_chunk.packed_at(2 * i).into_packed_m31s();
                        let val1 = src_chunk.packed_at(2 * i + 1).into_packed_m31s();
                        let pairs: [_; 4] = array::from_fn(|j| {
                            let (a, b) = val0[j].deinterleave(val1[j]);
                            simd_ibutterfly(a, b, t0)
                        });
                        let val0 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|j| pairs[j].0));
                        let val1 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|j| pairs[j].1));
                        val0 + PackedSecureField::broadcast(alpha) * val1
                    };

                    unsafe {
                        dst_chunk.set_packed(i, value);
                    }
                }
            });
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        let lambda = decomposition_coefficient(eval);
        let broadcasted_lambda = PackedSecureField::broadcast(lambda);
        let mut g_values = SecureColumnByCoords::<Self>::zeros(eval.len());

        let range = eval.len().div_ceil(N_LANES);
        let half_range = range / 2;
        for i in 0..half_range {
            let val = unsafe { eval.packed_at(i) } - broadcasted_lambda;
            unsafe { g_values.set_packed(i, val) }
        }
        for i in half_range..range {
            let val = unsafe { eval.packed_at(i) } + broadcasted_lambda;
            unsafe { g_values.set_packed(i, val) }
        }

        let g = SecureEvaluation::new(eval.domain, g_values);
        (g, lambda)
    }
}

/// Similar to [`crate::prover::fri::FriOps::fold_circle_into_line`], but optimized for folding a
/// BaseField circle evaluation directly into a line evaluation, without going through
/// SecureEvaluation.
pub fn fold_circle_evaluation_into_line(
    eval: &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
    alpha: SecureField,
    twiddles: &TwiddleTree<SimdBackend>,
) -> LineEvaluation<SimdBackend> {
    let log_size = eval.domain.log_size();
    let line_domain = LineDomain::new(Coset::half_odds(log_size - 1));
    let uninit_values = unsafe { SecureColumnByCoords::uninitialized(1 << (log_size - 1)) };
    let mut line_evaluation = LineEvaluation::new(line_domain, uninit_values);

    if log_size <= LOG_N_LANES {
        // Fall back to CPU implementation.
        let mut cpu_dst = line_evaluation.to_cpu();
        let secure_evaluation = SecureEvaluation::new(
            eval.domain,
            SecureColumnByCoords::from_base_field_col(&eval.values.to_cpu()),
        );
        fold_circle_into_line_cpu(&mut cpu_dst, &secure_evaluation, alpha);
        return LineEvaluation::new(
            cpu_dst.domain(),
            SecureColumnByCoords::from_cpu(cpu_dst.values),
        );
    }

    let domain = eval.domain;
    let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

    line_evaluation
        .values
        .par_chunks_mut(FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE)
        .zip_eq(
            eval.values
                .data
                .par_chunks(2 * FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE),
        )
        .zip_eq(itwiddles.par_chunks(8 * FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE))
        .for_each(|((mut dst_chunk, src_chunk), itwiddles_chunk)| {
            for i in 0..dst_chunk.len() {
                let value = unsafe {
                    // The 16 twiddles of the circle domain can be derived from the 8 twiddles of
                    // the next line domain. See `compute_first_twiddles()`.
                    let twiddle_dbl = u32x8::from_array(array::from_fn(|j| {
                        *itwiddles_chunk.get_unchecked(i * 8 + j)
                    }));
                    let (t0, _) = compute_first_twiddles(twiddle_dbl);
                    let val0 = src_chunk[2 * i];
                    let val1 = src_chunk[2 * i + 1];
                    let pairs = {
                        let (a, b) = val0.deinterleave(val1);
                        simd_ibutterfly(a, b, t0)
                    };
                    let [alpha_1, alpha_2, alpha_3, alpha_4] = alpha.to_m31_array();
                    PackedSecureField::from_packed_m31s([
                        pairs.0 + PackedBaseField::broadcast(alpha_1) * pairs.1,
                        PackedBaseField::broadcast(alpha_2) * pairs.1,
                        PackedBaseField::broadcast(alpha_3) * pairs.1,
                        PackedBaseField::broadcast(alpha_4) * pairs.1,
                    ])
                };

                unsafe {
                    dst_chunk.set_packed(i, value);
                }
            }
        });

    line_evaluation
}

/// Chunk size for parallel `fold_line_4x` operations.
pub const FOLD_LINE_4X_CHUNK_SIZE: usize = 1 << 7;

/// Batched fold that combines 4 fold_line steps into one operation.
/// This reduces memory allocations and improves cache efficiency by processing
/// 16 input values into 1 output value in a single pass.
///
/// # Arguments
/// * `eval` - The line evaluation to fold (must have at least 16 elements)
/// * `alphas` - Array of 4 folding alphas, in order from first to last fold step
/// * `twiddles` - Precomputed twiddles
///
/// # Returns
/// A line evaluation with 1/16th the size of the input
pub fn fold_line_4x(
    eval: &LineEvaluation<SimdBackend>,
    alphas: [SecureField; 4],
    twiddles: &TwiddleTree<SimdBackend>,
) -> LineEvaluation<SimdBackend> {
    let log_size = eval.len().ilog2();
    assert!(log_size >= 4, "fold_line_4x requires at least 16 elements");

    // Fall back to regular fold_line for small sizes
    if log_size <= LOG_N_LANES + 3 {
        let mut result = SimdBackend::fold_line(eval, alphas[0], twiddles);
        result = SimdBackend::fold_line(&result, alphas[1], twiddles);
        result = SimdBackend::fold_line(&result, alphas[2], twiddles);
        result = SimdBackend::fold_line(&result, alphas[3], twiddles);
        return result;
    }

    let domain = eval.domain();

    // Get twiddles for all 4 levels
    let all_itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles);

    let mut folded_values = unsafe { SecureColumnByCoords::uninitialized(1 << (log_size - 4)) };

    // Broadcast alphas for SIMD operations
    let alpha0 = PackedSecureField::broadcast(alphas[0]);
    let alpha1 = PackedSecureField::broadcast(alphas[1]);
    let alpha2 = PackedSecureField::broadcast(alphas[2]);
    let alpha3 = PackedSecureField::broadcast(alphas[3]);

    folded_values
        .par_chunks_mut(FOLD_LINE_4X_CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, mut dst_chunk)| {
            let base_idx = chunk_idx * FOLD_LINE_4X_CHUNK_SIZE;

            for i in 0..dst_chunk.len() {
                let output_idx = base_idx + i;

                // Read 16 consecutive packed values from the source
                // Each packed value contains N_LANES (16) SecureField elements
                let src_base = output_idx * 16;

                let value = unsafe {
                    // Level 0: 16 -> 8 values
                    // Twiddles for level 0 (largest, stride 16 per output element)
                    let itwiddles_l0 = all_itwiddles[0];
                    let mut vals_l1: [PackedSecureField; 8] = std::mem::zeroed();

                    for j in 0..8 {
                        let idx0 = src_base + 2 * j;
                        let idx1 = src_base + 2 * j + 1;

                        let twiddle_dbl = u32x16::from_array(array::from_fn(|k| {
                            *itwiddles_l0.get_unchecked((idx0 / 2) * 16 + k)
                        }));

                        let val0 = eval.values.packed_at(idx0).into_packed_m31s();
                        let val1 = eval.values.packed_at(idx1).into_packed_m31s();

                        let pairs: [_; 4] = array::from_fn(|c| {
                            let (a, b) = val0[c].deinterleave(val1[c]);
                            simd_ibutterfly(a, b, twiddle_dbl)
                        });

                        let r0 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].0));
                        let r1 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].1));

                        vals_l1[j] = r0 + alpha0 * r1;
                    }

                    // Level 1: 8 -> 4 values
                    let itwiddles_l1 = all_itwiddles[1];
                    let mut vals_l2: [PackedSecureField; 4] = std::mem::zeroed();

                    for j in 0..4 {
                        let logical_idx = (output_idx * 8 + 2 * j) / 2;

                        let twiddle_dbl = u32x16::from_array(array::from_fn(|k| {
                            *itwiddles_l1.get_unchecked(logical_idx * 16 + k)
                        }));

                        let val0 = vals_l1[2 * j].into_packed_m31s();
                        let val1 = vals_l1[2 * j + 1].into_packed_m31s();

                        let pairs: [_; 4] = array::from_fn(|c| {
                            let (a, b) = val0[c].deinterleave(val1[c]);
                            simd_ibutterfly(a, b, twiddle_dbl)
                        });

                        let r0 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].0));
                        let r1 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].1));

                        vals_l2[j] = r0 + alpha1 * r1;
                    }

                    // Level 2: 4 -> 2 values
                    let itwiddles_l2 = all_itwiddles[2];
                    let mut vals_l3: [PackedSecureField; 2] = std::mem::zeroed();

                    for j in 0..2 {
                        let logical_idx = (output_idx * 4 + 2 * j) / 2;

                        let twiddle_dbl = u32x16::from_array(array::from_fn(|k| {
                            *itwiddles_l2.get_unchecked(logical_idx * 16 + k)
                        }));

                        let val0 = vals_l2[2 * j].into_packed_m31s();
                        let val1 = vals_l2[2 * j + 1].into_packed_m31s();

                        let pairs: [_; 4] = array::from_fn(|c| {
                            let (a, b) = val0[c].deinterleave(val1[c]);
                            simd_ibutterfly(a, b, twiddle_dbl)
                        });

                        let r0 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].0));
                        let r1 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].1));

                        vals_l3[j] = r0 + alpha2 * r1;
                    }

                    // Level 3: 2 -> 1 value
                    let itwiddles_l3 = all_itwiddles[3];
                    let logical_idx = output_idx;

                    let twiddle_dbl = u32x16::from_array(array::from_fn(|k| {
                        *itwiddles_l3.get_unchecked(logical_idx * 16 + k)
                    }));

                    let val0 = vals_l3[0].into_packed_m31s();
                    let val1 = vals_l3[1].into_packed_m31s();

                    let pairs: [_; 4] = array::from_fn(|c| {
                        let (a, b) = val0[c].deinterleave(val1[c]);
                        simd_ibutterfly(a, b, twiddle_dbl)
                    });

                    let r0 = PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].0));
                    let r1 = PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].1));

                    r0 + alpha3 * r1
                };

                unsafe {
                    dst_chunk.set_packed(i, value);
                }
            }
        });

    // The domain is doubled 4 times
    let final_domain = domain.double().double().double().double();
    LineEvaluation::new(final_domain, folded_values)
}

/// See [`decomposition_coefficient`].
///
/// [`decomposition_coefficient`]: crate::prover::backend::cpu::CpuBackend::decomposition_coefficient
fn decomposition_coefficient(
    eval: &SecureEvaluation<SimdBackend, BitReversedOrder>,
) -> SecureField {
    let cols = &eval.values.columns;
    let [mut x_sum, mut y_sum, mut z_sum, mut w_sum] = [PackedBaseField::zero(); 4];

    let range = cols[0].len() / N_LANES;
    let (half_a, half_b) = (range / 2, range);

    for i in 0..half_a {
        x_sum += cols[0].data[i];
        y_sum += cols[1].data[i];
        z_sum += cols[2].data[i];
        w_sum += cols[3].data[i];
    }
    for i in half_a..half_b {
        x_sum -= cols[0].data[i];
        y_sum -= cols[1].data[i];
        z_sum -= cols[2].data[i];
        w_sum -= cols[3].data[i];
    }

    let x = x_sum.pointwise_sum();
    let y = y_sum.pointwise_sum();
    let z = z_sum.pointwise_sum();
    let w = w_sum.pointwise_sum();

    SecureField::from_m31(x, y, z, w) / BaseField::from_u32_unchecked(1 << eval.domain.log_size())
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::poly::line::LineDomain;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::simd::fri::fold_circle_evaluation_into_line;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::fri::FriOps;
    use crate::prover::line::LineEvaluation;
    use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps, SecureEvaluation};
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::{m31, qm31};

    #[test]
    fn test_fold_line() {
        const LOG_SIZE: u32 = 7;
        let mut rng = SmallRng::seed_from_u64(0);
        let values = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let alpha = qm31!(1, 3, 5, 7);
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
        let cpu_fold = CpuBackend::fold_line(
            &LineEvaluation::new(domain, values.iter().copied().collect()),
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
        );

        let avx_fold = SimdBackend::fold_line(
            &LineEvaluation::new(domain, values.iter().copied().collect()),
            alpha,
            &SimdBackend::precompute_twiddles(domain.coset()),
        );

        assert_eq!(cpu_fold.values.to_vec(), avx_fold.values.to_vec());
    }

    #[test]
    fn test_fold_circle_into_line() {
        const LOG_SIZE: u32 = 7;
        let values: Vec<SecureField> = (0..(1 << LOG_SIZE))
            .map(|i| qm31!(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
            .collect();
        let alpha = qm31!(1, 3, 5, 7);
        let circle_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let line_domain = LineDomain::new(circle_domain.half_coset);
        let mut cpu_fold = LineEvaluation::new(
            line_domain,
            SecureColumnByCoords::zeros(1 << (LOG_SIZE - 1)),
        );
        CpuBackend::fold_circle_into_line(
            &mut cpu_fold,
            &SecureEvaluation::new(circle_domain, values.iter().copied().collect()),
            alpha,
            &CpuBackend::precompute_twiddles(line_domain.coset()),
        );

        let mut simd_fold = LineEvaluation::new(
            line_domain,
            SecureColumnByCoords::zeros(1 << (LOG_SIZE - 1)),
        );
        SimdBackend::fold_circle_into_line(
            &mut simd_fold,
            &SecureEvaluation::new(circle_domain, values.iter().copied().collect()),
            alpha,
            &SimdBackend::precompute_twiddles(line_domain.coset()),
        );

        assert_eq!(cpu_fold.values.to_vec(), simd_fold.values.to_vec());
    }

    #[test]
    fn decomposition_test() {
        const DOMAIN_LOG_SIZE: u32 = 5;
        const DOMAIN_LOG_HALF_SIZE: u32 = DOMAIN_LOG_SIZE - 1;
        let s = CanonicCoset::new(DOMAIN_LOG_SIZE);
        let domain = s.circle_domain();
        let mut coeffs = BaseColumn::zeros(1 << DOMAIN_LOG_SIZE);
        // Polynomial is out of FFT space.
        coeffs.as_mut_slice()[1 << DOMAIN_LOG_HALF_SIZE] = BaseField::one();
        let poly = CircleCoefficients::<SimdBackend>::new(coeffs);
        let values = poly.evaluate(domain);
        let avx_column = SecureColumnByCoords::<SimdBackend> {
            columns: [
                values.values.clone(),
                values.values.clone(),
                values.values.clone(),
                values.values.clone(),
            ],
        };
        let avx_eval = SecureEvaluation::new(domain, avx_column.clone());
        let cpu_eval =
            SecureEvaluation::<CpuBackend, BitReversedOrder>::new(domain, avx_eval.values.to_cpu());
        let (cpu_g, cpu_lambda) = CpuBackend::decompose(&cpu_eval);
        let (avx_g, avx_lambda) = SimdBackend::decompose(&avx_eval);

        assert_eq!(avx_lambda, cpu_lambda);
        for i in 0..1 << DOMAIN_LOG_SIZE {
            assert_eq!(avx_g.values.at(i), cpu_g.values.at(i));
        }
    }

    #[test]
    fn test_fold_circle_into_line_v2() {
        const LOG_SIZE: u32 = 25;
        let values: Vec<BaseField> = (0..(1 << LOG_SIZE))
            .map(|i| m31!(4 * i))
            .collect();
        let alpha = qm31!(1, 3, 5, 7);
        let circle_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let line_domain = LineDomain::new(circle_domain.half_coset);

        for _ in 0..50 {

        fold_circle_evaluation_into_line(
            &CircleEvaluation::new(circle_domain, values.iter().copied().collect()),
            alpha,
            &SimdBackend::precompute_twiddles(line_domain.coset()),
        );
        }
    }

    #[test]
    fn test_fold_line_4x() {
        use super::fold_line_4x;

        const LOG_SIZE: u32 = 12;
        let mut rng = SmallRng::seed_from_u64(0);
        let values: Vec<SecureField> = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect();
        let alphas = [
            qm31!(1, 3, 5, 7),
            qm31!(2, 4, 6, 8),
            qm31!(9, 11, 13, 15),
            qm31!(10, 12, 14, 16),
        ];
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
        let twiddles = SimdBackend::precompute_twiddles(domain.coset());

        // Compute using 4 sequential fold_line calls
        let eval = LineEvaluation::new(domain, values.iter().copied().collect());
        let mut sequential_result = SimdBackend::fold_line(&eval, alphas[0], &twiddles);
        sequential_result = SimdBackend::fold_line(&sequential_result, alphas[1], &twiddles);
        sequential_result = SimdBackend::fold_line(&sequential_result, alphas[2], &twiddles);
        sequential_result = SimdBackend::fold_line(&sequential_result, alphas[3], &twiddles);

        // Compute using batched fold_line_4x
        let eval = LineEvaluation::new(domain, values.iter().copied().collect());
        let batched_result = fold_line_4x(&eval, alphas, &twiddles);

        assert_eq!(
            sequential_result.values.to_vec(),
            batched_result.values.to_vec(),
            "fold_line_4x should produce the same result as 4 sequential fold_line calls"
        );
    }

    #[test]
    fn test_fold_line_4x_large() {
        use super::fold_line_4x;

        const LOG_SIZE: u32 = 16;
        let mut rng = SmallRng::seed_from_u64(42);
        let values: Vec<SecureField> = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect();
        let alphas = [
            qm31!(17, 23, 31, 47),
            qm31!(53, 59, 67, 71),
            qm31!(73, 79, 83, 89),
            qm31!(97, 101, 103, 107),
        ];
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
        let twiddles = SimdBackend::precompute_twiddles(domain.coset());

        // Compute using 4 sequential fold_line calls
        let eval = LineEvaluation::new(domain, values.iter().copied().collect());
        let mut sequential_result = SimdBackend::fold_line(&eval, alphas[0], &twiddles);
        sequential_result = SimdBackend::fold_line(&sequential_result, alphas[1], &twiddles);
        sequential_result = SimdBackend::fold_line(&sequential_result, alphas[2], &twiddles);
        sequential_result = SimdBackend::fold_line(&sequential_result, alphas[3], &twiddles);

        // Compute using batched fold_line_4x
        let eval = LineEvaluation::new(domain, values.iter().copied().collect());
        let batched_result = fold_line_4x(&eval, alphas, &twiddles);

        assert_eq!(
            sequential_result.values.to_vec(),
            batched_result.values.to_vec(),
            "fold_line_4x should produce the same result as 4 sequential fold_line calls (large)"
        );
    }
}
