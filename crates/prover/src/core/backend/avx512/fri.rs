use num_traits::Zero;

use super::{AVX512Backend, PackedBaseField, K_BLOCK_SIZE};
use crate::core::backend::avx512::fft::compute_first_twiddles;
use crate::core::backend::avx512::fft::ifft::avx_ibutterfly;
use crate::core::backend::avx512::qm31::PackedSecureField;
use crate::core::backend::avx512::VECS_LOG_SIZE;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fri::{self, FriOps};
use crate::core::poly::circle::SecureEvaluation;
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::domain_line_twiddles_from_tree;

impl FriOps for AVX512Backend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let log_size = eval.len().ilog2();
        if log_size <= VECS_LOG_SIZE as u32 {
            let eval = fri::fold_line(&eval.to_cpu(), alpha);
            return LineEvaluation::new(eval.domain(), eval.values.into_iter().collect());
        }

        let domain = eval.domain();
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        let mut folded_values = SecureColumn::zeros(1 << (log_size - 1));

        for vec_index in 0..(1 << (log_size - 1 - VECS_LOG_SIZE as u32)) {
            let value = unsafe {
                let twiddle_dbl: [i32; 16] =
                    std::array::from_fn(|i| *itwiddles.get_unchecked(vec_index * 16 + i));
                let val0 = eval.values.packed_at(vec_index * 2).to_packed_m31s();
                let val1 = eval.values.packed_at(vec_index * 2 + 1).to_packed_m31s();
                let pairs: [_; 4] = std::array::from_fn(|i| {
                    let (a, b) = val0[i].deinterleave_with(val1[i]);
                    avx_ibutterfly(a, b, std::mem::transmute(twiddle_dbl))
                });
                let val0 = PackedSecureField::from_packed_m31s(std::array::from_fn(|i| pairs[i].0));
                let val1 = PackedSecureField::from_packed_m31s(std::array::from_fn(|i| pairs[i].1));
                val0 + PackedSecureField::broadcast(alpha) * val1
            };
            unsafe { folded_values.set_packed(vec_index, value) };
        }

        LineEvaluation::new(domain.double(), folded_values)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let log_size = src.len().ilog2();
        assert!(log_size > VECS_LOG_SIZE as u32, "Evaluation too small");

        let domain = src.domain;
        let alpha_sq = alpha * alpha;
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        for vec_index in 0..(1 << (log_size - 1 - VECS_LOG_SIZE as u32)) {
            let value = unsafe {
                // The 16 twiddles of the circle domain can be derived from the 8 twiddles of the
                // next line domain. See `compute_first_twiddles()`.
                let twiddle_dbl: [i32; 8] =
                    std::array::from_fn(|i| *itwiddles.get_unchecked(vec_index * 8 + i));
                let (t0, _) = compute_first_twiddles(twiddle_dbl);
                let val0 = src.values.packed_at(vec_index * 2).to_packed_m31s();
                let val1 = src.values.packed_at(vec_index * 2 + 1).to_packed_m31s();
                let pairs: [_; 4] = std::array::from_fn(|i| {
                    let (a, b) = val0[i].deinterleave_with(val1[i]);
                    avx_ibutterfly(a, b, t0)
                });
                let val0 = PackedSecureField::from_packed_m31s(std::array::from_fn(|i| pairs[i].0));
                let val1 = PackedSecureField::from_packed_m31s(std::array::from_fn(|i| pairs[i].1));
                val0 + PackedSecureField::broadcast(alpha) * val1
            };
            unsafe {
                dst.values.set_packed(
                    vec_index,
                    dst.values.packed_at(vec_index) * PackedSecureField::broadcast(alpha_sq)
                        + value,
                )
            };
        }
    }

    fn decompose(mut eval: SecureEvaluation<Self>) -> (SecureEvaluation<Self>, SecureField) {
        let domain_half_size = 1 << (eval.domain.log_size() - 1);
        let lambda = Self::decomposition_coefficient(&eval);
        let broadcasted_lambda = PackedSecureField::broadcast(lambda);
        let col = &mut eval.values;

        (0..col.len().div_ceil(K_BLOCK_SIZE)).for_each(|i| {
            if i < domain_half_size / K_BLOCK_SIZE {
                unsafe { col.set_packed(i, col.packed_at(i) - broadcasted_lambda) }
            } else {
                unsafe { col.set_packed(i, col.packed_at(i) + broadcasted_lambda) }
            }
        });

        (eval, lambda)
    }
}

impl AVX512Backend {
    /// See [`decomposition_coefficient`].
    ///
    /// [`decomposition_coefficient`]: crate::core::backend::cpu::CPUBackend::decomposition_coefficient
    // TODO(Ohad): remove pub.
    pub fn decomposition_coefficient(eval: &SecureEvaluation<Self>) -> SecureField {
        let cols = &eval.values.columns;
        let [mut x_sum, mut y_sum, mut z_sum, mut w_sum] = [PackedBaseField::zero(); 4];

        let range = cols[0].len() / K_BLOCK_SIZE;
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

        SecureField::from_m31(x, y, z, w)
            / BaseField::from_u32_unchecked(1 << eval.domain.log_size())
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec};
    use crate::core::backend::{CPUBackend, Column};
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumn;
    use crate::core::fri::FriOps;
    use crate::core::poly::circle::{CanonicCoset, CirclePoly, PolyOps, SecureEvaluation};
    use crate::core::poly::line::{LineDomain, LineEvaluation};
    use crate::{m31, qm31};

    #[test]
    fn test_fold_line() {
        const LOG_SIZE: u32 = 7;
        let values: Vec<SecureField> = (0..(1 << LOG_SIZE))
            .map(|i| qm31!(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
            .collect();
        let alpha = qm31!(1, 3, 5, 7);
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
        let cpu_fold = CPUBackend::fold_line(
            &LineEvaluation::new(domain, values.iter().copied().collect()),
            alpha,
            &CPUBackend::precompute_twiddles(domain.coset()),
        );

        let avx_fold = AVX512Backend::fold_line(
            &LineEvaluation::new(domain, values.iter().copied().collect()),
            alpha,
            &AVX512Backend::precompute_twiddles(domain.coset()),
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

        let mut cpu_fold =
            LineEvaluation::new(line_domain, SecureColumn::zeros(1 << (LOG_SIZE - 1)));
        CPUBackend::fold_circle_into_line(
            &mut cpu_fold,
            &SecureEvaluation {
                domain: circle_domain,
                values: values.iter().copied().collect(),
            },
            alpha,
            &CPUBackend::precompute_twiddles(line_domain.coset()),
        );

        let mut avx_fold =
            LineEvaluation::new(line_domain, SecureColumn::zeros(1 << (LOG_SIZE - 1)));
        AVX512Backend::fold_circle_into_line(
            &mut avx_fold,
            &SecureEvaluation {
                domain: circle_domain,
                values: values.iter().copied().collect(),
            },
            alpha,
            &AVX512Backend::precompute_twiddles(line_domain.coset()),
        );

        assert_eq!(cpu_fold.values.to_vec(), avx_fold.values.to_vec());
    }

    #[test]
    fn decomposition_test() {
        const DOMAIN_LOG_SIZE: u32 = 5;
        const DOMAIN_LOG_HALF_SIZE: u32 = DOMAIN_LOG_SIZE - 1;
        let s = CanonicCoset::new(DOMAIN_LOG_SIZE);
        let domain = s.circle_domain();

        let mut coeffs = BaseFieldVec::zeros(1 << DOMAIN_LOG_SIZE);

        // Polynomial is out of FFT space.
        coeffs.as_mut_slice()[1 << DOMAIN_LOG_HALF_SIZE] = m31!(1);
        let poly = CirclePoly::<AVX512Backend>::new(coeffs);
        let values = poly.evaluate(domain);

        let avx_column = SecureColumn::<AVX512Backend> {
            columns: [
                values.values.clone(),
                values.values.clone(),
                values.values.clone(),
                values.values.clone(),
            ],
        };
        let avx_eval = SecureEvaluation {
            domain,
            values: avx_column.clone(),
        };
        let cpu_eval = SecureEvaluation::<CPUBackend> {
            domain,
            values: avx_eval.to_cpu(),
        };
        let (cpu_g, cpu_lambda) = CPUBackend::decompose(cpu_eval);

        let (avx_g, avx_lambda) = AVX512Backend::decompose(avx_eval);

        assert_eq!(avx_lambda, cpu_lambda);

        for i in 0..(1 << DOMAIN_LOG_SIZE) {
            assert_eq!(avx_g.values.at(i), cpu_g.values.at(i));
        }
    }
}
