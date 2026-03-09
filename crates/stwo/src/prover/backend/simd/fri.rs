use std::array;
use std::simd::{u32x16, u32x8};

use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use super::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use super::SimdBackend;
use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::line::LineDomain;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::core::utils::uninit_vec;
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

const FOLD_LINE_CHUNK_SIZE: usize = 128;

// TODO(andrew) Is this optimized?
impl FriOps for SimdBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
        fold_step: u32,
    ) -> LineEvaluation<Self> {
        assert!(fold_step >= 1, "fold_step must be positive.");

        let log_size = eval.len().ilog2();
        // Fallback to cpu if the log size is too small.
        if log_size < LOG_N_LANES + fold_step {
            let mut folding_alpha = alpha;
            let mut eval = fold_line_cpu(&eval.to_cpu(), folding_alpha);
            for _ in 0..fold_step - 1 {
                folding_alpha = folding_alpha * folding_alpha;
                eval = fold_line_cpu(&eval, folding_alpha)
            }
            return LineEvaluation::new(eval.domain(), eval.values.into_iter().collect());
        }
        let mut alphas = vec![];
        let mut folding_alpha = alpha;
        for _ in 0..fold_step {
            alphas.push(folding_alpha);
            folding_alpha = folding_alpha * folding_alpha;
        }

        let domain = eval.domain();
        let all_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles);
        let mut folded_values =
            unsafe { SecureColumnByCoords::uninitialized(1 << (log_size - fold_step)) };

        #[cfg(not(feature = "parallel"))]
        let folded_values_iter = folded_values.chunks_mut(FOLD_LINE_CHUNK_SIZE);
        #[cfg(feature = "parallel")]
        let folded_values_iter = folded_values.par_chunks_mut(FOLD_LINE_CHUNK_SIZE);

        folded_values_iter
            .enumerate()
            .for_each(|(chunk_idx, mut dst_chunk)| {
                let chunk_start = chunk_idx * FOLD_LINE_CHUNK_SIZE;
                let mut layer_values: Vec<[PackedBaseField; 4]> =
                    unsafe { uninit_vec(1 << fold_step) };
                let packed_chunk_len = dst_chunk.0[0].0.len();

                for local_i in 0..packed_chunk_len {
                    let i = chunk_start + local_i;
                    // Read the packed inputs needed for a full fold.
                    let input_base = i << fold_step;
                    unsafe {
                        for (j, val) in layer_values.iter_mut().enumerate() {
                            *val = eval.values.packed_at(input_base + j).into_packed_m31s();
                        }
                    }
                    let mut next_layer_size = 1 << (fold_step - 1);
                    for layer in 0..fold_step as usize {
                        let itwiddles = all_twiddles[layer];
                        let alpha = alphas[layer];
                        unsafe {
                            for j in 0..next_layer_size {
                                let packed_itwiddles = u32x16::from_array(array::from_fn(|k| {
                                    *itwiddles.get_unchecked((i * next_layer_size + j) * 16 + k)
                                }));
                                let val0 = layer_values[2 * j];
                                let val1 = layer_values[2 * j + 1];
                                let pairs: [_; 4] = array::from_fn(|c| {
                                    let (a, b) = val0[c].deinterleave(val1[c]);
                                    simd_ibutterfly(a, b, packed_itwiddles)
                                });
                                let v0 = PackedSecureField::from_packed_m31s(array::from_fn(|c| {
                                    pairs[c].0
                                }));
                                let v1 = PackedSecureField::from_packed_m31s(array::from_fn(|c| {
                                    pairs[c].1
                                }));
                                layer_values[j] = (v0 + PackedSecureField::broadcast(alpha) * v1)
                                    .into_packed_m31s();
                            }
                        }
                        next_layer_size >>= 1;
                    }
                    let result = layer_values[0];

                    unsafe {
                        dst_chunk.set_packed(local_i, PackedSecureField::from_packed_m31s(result));
                    }
                }
            });

        let new_domain = domain.repeated_double(fold_step);
        LineEvaluation::new(new_domain, folded_values)
    }

    fn fold_circle_into_line(
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let log_size = src.len().ilog2();
        if log_size <= LOG_N_LANES {
            // Fall back to CPU implementation.
            let cpu_fold = fold_circle_into_line_cpu(&src.to_cpu(), alpha);
            return LineEvaluation::new(
                cpu_fold.domain(),
                SecureColumnByCoords::from_cpu(cpu_fold.values),
            );
        }

        // Create the destination buffer.
        let line_log_size = src.domain.log_size() - 1;
        let dst_domain = LineDomain::new(Coset::half_odds(line_log_size));
        let values = unsafe { SecureColumnByCoords::uninitialized(1 << line_log_size) };
        let mut dst = LineEvaluation::new(dst_domain, values);
        let itwiddles = domain_line_twiddles_from_tree(src.domain, &twiddles.itwiddles)[0];

        for vec_index in 0..(1 << (log_size - 1 - LOG_N_LANES)) {
            let value = unsafe {
                // The 16 twiddles of the circle domain can be derived from the 8 twiddles of the
                // next line domain. See `compute_first_twiddles()`.
                let twiddle_dbl = u32x8::from_array(array::from_fn(|i| {
                    *itwiddles.get_unchecked(vec_index * 8 + i)
                }));
                let (t0, _) = compute_first_twiddles(twiddle_dbl);
                let val0 = src.values.packed_at(vec_index * 2).into_packed_m31s();
                let val1 = src.values.packed_at(vec_index * 2 + 1).into_packed_m31s();
                let pairs: [_; 4] = array::from_fn(|i| {
                    let (a, b) = val0[i].deinterleave(val1[i]);
                    simd_ibutterfly(a, b, t0)
                });
                let val0 = PackedSecureField::from_packed_m31s(array::from_fn(|i| pairs[i].0));
                let val1 = PackedSecureField::from_packed_m31s(array::from_fn(|i| pairs[i].1));
                val0 + PackedSecureField::broadcast(alpha) * val1
            };
            unsafe { dst.values.set_packed(vec_index, value) };
        }
        dst
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
    let mut line_evaluation = LineEvaluation::new_zero(line_domain);

    if log_size <= LOG_N_LANES {
        // Fall back to CPU implementation.
        let secure_evaluation = SecureEvaluation::new(
            eval.domain,
            SecureColumnByCoords::from_base_field_col(&eval.values.to_cpu()),
        );
        let cpu_fold = fold_circle_into_line_cpu(&secure_evaluation, alpha);
        return LineEvaluation::new(
            cpu_fold.domain(),
            SecureColumnByCoords::from_cpu(cpu_fold.values),
        );
    }

    let itwiddles = domain_line_twiddles_from_tree(line_domain, &twiddles.itwiddles)[0];

    for vec_index in 0..(1 << (log_size - 1 - LOG_N_LANES)) {
        let value = {
            // The 16 twiddles of the circle domain can be derived from the 8 twiddles of the
            // next line domain. See `compute_first_twiddles()`.
            let twiddle_dbl = u32x8::from_array(array::from_fn(|i| unsafe {
                *itwiddles.get_unchecked(vec_index * 8 + i)
            }));
            let (t0, _) = compute_first_twiddles(twiddle_dbl);
            let val0 = eval.values.data[vec_index * 2];
            let val1 = eval.values.data[vec_index * 2 + 1];
            let pairs = {
                let (a, b) = val0.deinterleave(val1);
                simd_ibutterfly(a, b, t0)
            };
            let val0 = PackedSecureField::from_packed_m31s(array::from_fn(|i| {
                if i == 0 {
                    pairs.0
                } else {
                    PackedBaseField::zero()
                }
            }));
            let val1 = PackedSecureField::from_packed_m31s(array::from_fn(|i| {
                if i == 0 {
                    pairs.1
                } else {
                    PackedBaseField::zero()
                }
            }));
            val0 + PackedSecureField::broadcast(alpha) * val1
        };

        unsafe { line_evaluation.values.set_packed(vec_index, value) };
    }

    line_evaluation
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
            1,
        );

        let avx_fold = SimdBackend::fold_line(
            &LineEvaluation::new(domain, values.iter().copied().collect()),
            alpha,
            &SimdBackend::precompute_twiddles(domain.coset()),
            1,
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
        let cpu_fold = CpuBackend::fold_circle_into_line(
            &SecureEvaluation::new(circle_domain, values.iter().copied().collect()),
            alpha,
            &CpuBackend::precompute_twiddles(line_domain.coset()),
        );

        let simd_fold = SimdBackend::fold_circle_into_line(
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
}
