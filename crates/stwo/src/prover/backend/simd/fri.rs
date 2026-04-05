#[cfg(target_feature = "avx512f")]
use std::arch::x86_64::{__m512i, _mm512_stream_si512};
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
use std::arch::x86_64::{__m256i, _mm256_stream_si256};
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
            .for_each(|((dst_chunk, src_chunk), itwiddles_chunk)| {
                let mut dst_chunk = dst_chunk;
                for i in 0..dst_chunk.len() {
                    let value = unsafe {
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
                        let res = val0 + PackedSecureField::broadcast(alpha) * val1;
                        res.into_packed_m31s()
                    };

                    unsafe {
                        dst_chunk.set_packed(i, PackedSecureField::from_packed_m31s(value));
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
            .for_each(|((dst_chunk, src_chunk), itwiddles_chunk)| {
                let mut dst_chunk = dst_chunk;
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

    if log_size <= LOG_N_LANES {
        // Fall back to CPU implementation.
        let uninit_values =
            unsafe { SecureColumnByCoords::<SimdBackend>::uninitialized(1 << (log_size - 1)) };
        let mut cpu_dst = LineEvaluation::new(line_domain, uninit_values).to_cpu();
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
    let uninit_values =
        unsafe { SecureColumnByCoords::<SimdBackend>::uninitialized(1 << (log_size - 1)) };
    let mut line_evaluation = LineEvaluation::new(line_domain, uninit_values);

    // Precompute alpha components outside the loop
    let [alpha_1, alpha_2, alpha_3, alpha_4] = alpha.to_m31_array();
    let alpha_1_packed = PackedBaseField::broadcast(alpha_1);
    let alpha_2_packed = PackedBaseField::broadcast(alpha_2);
    let alpha_3_packed = PackedBaseField::broadcast(alpha_3);
    let alpha_4_packed = PackedBaseField::broadcast(alpha_4);

    line_evaluation
        .values
        .par_chunks_mut(FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE)
        .zip_eq(
            eval.values
                .data
                .par_chunks(2 * FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE),
        )
        .zip_eq(itwiddles.par_chunks(8 * FOLD_CIRCLE_INTO_LINE_CHUNK_SIZE))
        .for_each(|((_dst_chunk, src_chunk), itwiddles_chunk)| {
            #[allow(unused_mut)]
            let mut dst_chunk = _dst_chunk;
            for i in 0..dst_chunk.len() {
                let value = unsafe {
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
                    [
                        pairs.0 + alpha_1_packed * pairs.1,
                        alpha_2_packed * pairs.1,
                        alpha_3_packed * pairs.1,
                        alpha_4_packed * pairs.1,
                    ]
                };

                // Use streaming stores to bypass cache (write-allocate).
                #[cfg(target_feature = "avx512f")]
                unsafe {
                    _mm512_stream_si512(
                        dst_chunk.0[0].0.as_mut_ptr().add(i) as *mut __m512i,
                        std::mem::transmute(value[0].into_simd()),
                    );
                    _mm512_stream_si512(
                        dst_chunk.0[1].0.as_mut_ptr().add(i) as *mut __m512i,
                        std::mem::transmute(value[1].into_simd()),
                    );
                    _mm512_stream_si512(
                        dst_chunk.0[2].0.as_mut_ptr().add(i) as *mut __m512i,
                        std::mem::transmute(value[2].into_simd()),
                    );
                    _mm512_stream_si512(
                        dst_chunk.0[3].0.as_mut_ptr().add(i) as *mut __m512i,
                        std::mem::transmute(value[3].into_simd()),
                    );
                }

                // Use AVX2 streaming stores when AVX-512 is not available.
                #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
                unsafe {
                    for col_idx in 0..4 {
                        let simd = value[col_idx].into_simd();
                        let ptr = dst_chunk.0[col_idx].0.as_mut_ptr().add(i) as *mut __m256i;
                        // Store low 256 bits
                        _mm256_stream_si256(
                            ptr,
                            std::mem::transmute::<[u32; 8], __m256i>(
                                simd.to_array()[0..8].try_into().unwrap(),
                            ),
                        );
                        // Store high 256 bits
                        _mm256_stream_si256(
                            ptr.add(1),
                            std::mem::transmute::<[u32; 8], __m256i>(
                                simd.to_array()[8..16].try_into().unwrap(),
                            ),
                        );
                    }
                }

                #[cfg(not(any(target_feature = "avx512f", target_arch = "x86_64")))]
                unsafe {
                    dst_chunk.set_packed(i, PackedSecureField::from_packed_m31s(value));
                }
            }
        });

    line_evaluation
}

/// Folds a line evaluation 4 times in a single pass, keeping intermediate results in registers.
/// This reduces memory traffic by reading 16 packed elements and writing 1 packed element,
/// instead of 4 separate read-write passes.
///
/// Requires that the input has at least 2^(LOG_N_LANES + 4) scalar elements.
pub fn fold_line_4x(
    eval: &LineEvaluation<SimdBackend>,
    alphas: [SecureField; 4],
    twiddles: &TwiddleTree<SimdBackend>,
) -> LineEvaluation<SimdBackend> {
    let log_size = eval.len().ilog2();
    assert!(
        log_size >= LOG_N_LANES + 4,
        "fold_line_4x requires at least {} elements, got {}",
        1 << (LOG_N_LANES + 4),
        eval.len()
    );

    let domain = eval.domain();
    // Get twiddles for all 4 layers
    let all_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles);
    let itwiddles_0 = all_twiddles[0];
    let itwiddles_1 = all_twiddles[1];
    let itwiddles_2 = all_twiddles[2];
    let itwiddles_3 = all_twiddles[3];

    let output_size = 1 << (log_size - 4);
    let mut folded_values = unsafe { SecureColumnByCoords::uninitialized(output_size) };

    folded_values
        .par_chunks_mut(FOLD_LINE_CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, mut dst_chunk)| {
            let chunk_start = chunk_idx * FOLD_LINE_CHUNK_SIZE;

            for local_i in 0..dst_chunk.len() {
                let i = chunk_start + local_i;

                // Read 16 input packed elements
                let input_base = i * 16;
                let values: [[PackedBaseField; 4]; 16] = unsafe {
                    array::from_fn(|j| {
                        eval.values.packed_at(input_base + j).into_packed_m31s()
                    })
                };

                // Layer 1: 16 -> 8 elements
                let layer1: [[PackedBaseField; 4]; 8] = unsafe {
                    array::from_fn(|j| {
                        let twiddle_dbl = u32x16::from_array(array::from_fn(|k| {
                            *itwiddles_0.get_unchecked((i * 8 + j) * 16 + k)
                        }));
                        let val0 = values[2 * j];
                        let val1 = values[2 * j + 1];
                        let pairs: [_; 4] = array::from_fn(|c| {
                            let (a, b) = val0[c].deinterleave(val1[c]);
                            simd_ibutterfly(a, b, twiddle_dbl)
                        });
                        let v0 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].0));
                        let v1 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].1));
                        (v0 + PackedSecureField::broadcast(alphas[0]) * v1).into_packed_m31s()
                    })
                };

                // Layer 2: 8 -> 4 elements
                let layer2: [[PackedBaseField; 4]; 4] = unsafe {
                    array::from_fn(|j| {
                        let twiddle_dbl = u32x16::from_array(array::from_fn(|k| {
                            *itwiddles_1.get_unchecked((i * 4 + j) * 16 + k)
                        }));
                        let val0 = layer1[2 * j];
                        let val1 = layer1[2 * j + 1];
                        let pairs: [_; 4] = array::from_fn(|c| {
                            let (a, b) = val0[c].deinterleave(val1[c]);
                            simd_ibutterfly(a, b, twiddle_dbl)
                        });
                        let v0 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].0));
                        let v1 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].1));
                        (v0 + PackedSecureField::broadcast(alphas[1]) * v1).into_packed_m31s()
                    })
                };

                // Layer 3: 4 -> 2 elements
                let layer3: [[PackedBaseField; 4]; 2] = unsafe {
                    array::from_fn(|j| {
                        let twiddle_dbl = u32x16::from_array(array::from_fn(|k| {
                            *itwiddles_2.get_unchecked((i * 2 + j) * 16 + k)
                        }));
                        let val0 = layer2[2 * j];
                        let val1 = layer2[2 * j + 1];
                        let pairs: [_; 4] = array::from_fn(|c| {
                            let (a, b) = val0[c].deinterleave(val1[c]);
                            simd_ibutterfly(a, b, twiddle_dbl)
                        });
                        let v0 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].0));
                        let v1 =
                            PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].1));
                        (v0 + PackedSecureField::broadcast(alphas[2]) * v1).into_packed_m31s()
                    })
                };

                // Layer 4: 2 -> 1 element
                let result: [PackedBaseField; 4] = unsafe {
                    let twiddle_dbl = u32x16::from_array(array::from_fn(|k| {
                        *itwiddles_3.get_unchecked(i * 16 + k)
                    }));
                    let val0 = layer3[0];
                    let val1 = layer3[1];
                    let pairs: [_; 4] = array::from_fn(|c| {
                        let (a, b) = val0[c].deinterleave(val1[c]);
                        simd_ibutterfly(a, b, twiddle_dbl)
                    });
                    let v0 = PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].0));
                    let v1 = PackedSecureField::from_packed_m31s(array::from_fn(|c| pairs[c].1));
                    (v0 + PackedSecureField::broadcast(alphas[3]) * v1).into_packed_m31s()
                };

                // Use non-temporal stores to bypass cache when AVX-512 is available.
                #[cfg(target_feature = "avx512f")]
                unsafe {
                    _mm512_stream_si512(
                        dst_chunk.0[0].0.as_mut_ptr().add(local_i) as *mut __m512i,
                        std::mem::transmute(result[0].into_simd()),
                    );
                    _mm512_stream_si512(
                        dst_chunk.0[1].0.as_mut_ptr().add(local_i) as *mut __m512i,
                        std::mem::transmute(result[1].into_simd()),
                    );
                    _mm512_stream_si512(
                        dst_chunk.0[2].0.as_mut_ptr().add(local_i) as *mut __m512i,
                        std::mem::transmute(result[2].into_simd()),
                    );
                    _mm512_stream_si512(
                        dst_chunk.0[3].0.as_mut_ptr().add(local_i) as *mut __m512i,
                        std::mem::transmute(result[3].into_simd()),
                    );
                }

                #[cfg(not(target_feature = "avx512f"))]
                unsafe {
                    dst_chunk.set_packed(local_i, PackedSecureField::from_packed_m31s(result));
                }
            }
        });

    // Domain doubles 4 times
    let new_domain = domain.double().double().double().double();
    LineEvaluation::new(new_domain, folded_values)
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
    use crate::prover::backend::simd::fri::fold_line_4x;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::fri::FriOps;
    use crate::prover::line::LineEvaluation;
    use crate::prover::poly::circle::{CircleCoefficients, PolyOps, SecureEvaluation};
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::qm31;

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
    fn test_fold_line_4x_correctness() {
        // Test that fold_line_4x produces the same result as 4 sequential fold_line calls
        const LOG_SIZE: u32 = 12; // Must be >= LOG_N_LANES + 4 = 4 + 4 = 8
        let mut rng = SmallRng::seed_from_u64(42);
        let values: Vec<SecureField> = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let alphas = [
            qm31!(1, 3, 5, 7),
            qm31!(2, 4, 6, 8),
            qm31!(9, 11, 13, 15),
            qm31!(10, 12, 14, 16),
        ];
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
        let twiddles = SimdBackend::precompute_twiddles(domain.coset());

        // Method 1: Use fold_line_4x
        let eval = LineEvaluation::new(domain, values.iter().copied().collect());
        let result_4x = fold_line_4x(&eval, alphas, &twiddles);

        // Method 2: Use 4 sequential fold_line calls
        let mut result_sequential = LineEvaluation::new(domain, values.iter().copied().collect());
        for alpha in alphas.iter() {
            result_sequential = SimdBackend::fold_line(&result_sequential, *alpha, &twiddles);
        }

        // Compare results
        assert_eq!(
            result_4x.values.to_vec(),
            result_sequential.values.to_vec(),
            "fold_line_4x should produce same result as 4 sequential fold_line calls"
        );
        assert_eq!(
            result_4x.domain().log_size(),
            result_sequential.domain().log_size(),
            "domain log sizes should match"
        );
    }
}
