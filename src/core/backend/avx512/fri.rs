use super::AVX512Backend;
use crate::core::backend::avx512::fft::compute_first_twiddles;
use crate::core::backend::avx512::fft::ifft::avx_ibutterfly;
use crate::core::backend::avx512::qm31::PackedQM31;
use crate::core::backend::avx512::VECS_LOG_SIZE;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fri::FriOps;
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
        assert!(log_size > VECS_LOG_SIZE as u32, "Evaluation too small");

        let domain = eval.domain();
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        let mut folded_values = SecureColumn::zeros(1 << (log_size - 1));

        for vec_index in 0..(1 << (log_size - 1 - VECS_LOG_SIZE as u32)) {
            let value = unsafe {
                let twiddle_dbl: [i32; 16] =
                    std::array::from_fn(|i| *itwiddles.get_unchecked(vec_index * 16 + i));
                let val0 = eval.values.packed_at(vec_index * 2).to_packed_m31s();
                let val1 = eval.values.packed_at(vec_index * 2 + 1).to_packed_m31s();
                let els: [_; 4] = std::array::from_fn(|i| {
                    avx_ibutterfly(val0[i], val1[i], std::mem::transmute(twiddle_dbl))
                });
                let val0 = PackedQM31::from_packed_m31s(std::array::from_fn(|i| els[i].0));
                let val1 = PackedQM31::from_packed_m31s(std::array::from_fn(|i| els[i].1));
                val0 + PackedQM31::broadcast(alpha) * val1
            };
            folded_values.set_packed(vec_index, value);
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
                let twiddle_dbl: [i32; 8] =
                    std::array::from_fn(|i| *itwiddles.get_unchecked(vec_index * 8 + i));
                let (t0, _) = compute_first_twiddles(twiddle_dbl);
                let val0 = src.values.packed_at(vec_index * 2).to_packed_m31s();
                let val1 = src.values.packed_at(vec_index * 2 + 1).to_packed_m31s();
                let els: [_; 4] = std::array::from_fn(|i| avx_ibutterfly(val0[i], val1[i], t0));
                let val0 = PackedQM31::from_packed_m31s(std::array::from_fn(|i| els[i].0));
                let val1 = PackedQM31::from_packed_m31s(std::array::from_fn(|i| els[i].1));
                val0 + PackedQM31::broadcast(alpha) * val1
            };
            dst.values.set_packed(
                vec_index,
                dst.values.packed_at(vec_index) * PackedQM31::broadcast(alpha_sq) + value,
            );
        }
    }
}
