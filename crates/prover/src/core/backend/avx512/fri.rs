use super::{AVX512Backend, PackedBaseField};
use crate::core::backend::avx512::fft::compute_first_twiddles;
use crate::core::backend::avx512::fft::ifft::avx_ibutterfly;
use crate::core::backend::avx512::qm31::PackedSecureField;
use crate::core::backend::avx512::VECS_LOG_SIZE;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fri::{self, FriOps};
use crate::core::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::core::poly::NaturalOrder;

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

    fn coset_diff(eval: CircleEvaluation<Self, BaseField, NaturalOrder>) -> BaseField {
        let half_domain_size = 1 << (eval.domain.log_size() - 1);
        let (a_values, b_values) = eval.values.data.split_at(half_domain_size / 16);
        let a_sum = a_values
            .iter()
            .copied()
            .sum::<PackedBaseField>()
            .pointwise_sum();
        let b_sum = b_values
            .iter()
            .copied()
            .sum::<PackedBaseField>()
            .pointwise_sum();
        a_sum - b_sum
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec, PackedBaseField};
    use crate::core::backend::CPUBackend;
    use crate::core::fields::m31::BaseField;
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
    fn coset_diff_out_fft_space_test() {
        const DOMAIN_LOG_SIZE: u32 = 5;
        const DOMAIN_SIZE: u32 = 1 << DOMAIN_LOG_SIZE;
        let evaluation_domain = CanonicCoset::new(DOMAIN_LOG_SIZE).circle_domain();

        // [0, 1, 2, ..., DOMAIN_SIZE - 2, 0]
        let coeffs_in_fft = (0..DOMAIN_SIZE)
            .map(|i| BaseField::from_u32_unchecked(i % (DOMAIN_SIZE - 1)))
            .collect();

        // [0, 0, 0, ..., 0, DOMAIN_SIZE - 1]
        let mut coeffs_out_fft: BaseFieldVec = [0; DOMAIN_SIZE as usize]
            .into_iter()
            .map(BaseField::from_u32_unchecked)
            .collect();
        let mut last_coeff = [m31!(0); 16];
        last_coeff[15] = m31!(31);

        let data_len = coeffs_out_fft.data.len();
        coeffs_out_fft.data[data_len - 1] = PackedBaseField::from_array(last_coeff);

        // [0, 1, 2, ... , DOMAIN_SIZE - 1]
        let combined_poly_coeffs = (0..DOMAIN_SIZE)
            .map(BaseField::from_u32_unchecked)
            .collect();

        let in_fft_poly_eval = CirclePoly::<AVX512Backend>::new(coeffs_in_fft)
            .evaluate(evaluation_domain)
            .bit_reverse();
        let out_fft_poly_eval = CirclePoly::<AVX512Backend>::new(coeffs_out_fft)
            .evaluate(evaluation_domain)
            .bit_reverse();
        let combined_poly_eval = CirclePoly::<AVX512Backend>::new(combined_poly_coeffs)
            .evaluate(evaluation_domain)
            .bit_reverse();

        let in_lambda = AVX512Backend::coset_diff(in_fft_poly_eval);
        let out_lambda = AVX512Backend::coset_diff(out_fft_poly_eval);
        let combined_lambda = AVX512Backend::coset_diff(combined_poly_eval);

        assert_eq!(in_lambda, BaseField::zero());
        assert_eq!(out_lambda, combined_lambda);
    }
}
