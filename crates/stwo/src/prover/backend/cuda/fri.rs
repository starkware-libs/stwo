use interface::bindings;
use interface::bindings::CudaSecureField;

use super::CudaBackend;
use crate::core::circle::Coset;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::FOLD_STEP as CIRCLE_TO_LINE_FOLD_STEP;
use crate::core::poly::line::LineDomain;
use crate::prover::backend::cuda::secure_column::CudaSecureColumn;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::BitReversedOrder;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::stwo_cuda as interface;

/// Single fold step (NitrooZK's original `fold_line`, fold_step = 1): folds a degree-d line
/// polynomial into degree d/2 using one `alpha`. The 74951f79 `FriOps::fold_line` (below) loops
/// this over `alphas` — k sequential single-steps == its batched contract (the SIMD backend just
/// fuses them for speed; the result is identical).
fn fold_line_single(
    eval: &LineEvaluation<CudaBackend>,
    alpha: SecureField,
    twiddles: &TwiddleTree<CudaBackend>,
) -> LineEvaluation<CudaBackend> {
    let n = eval.len();
    assert!(n >= 2, "Evaluation too small");

    let twiddles_size = twiddles.itwiddles.size;
    let remaining_folds = n.ilog2();
    let twiddle_offset: usize = twiddles_size - (1 << remaining_folds);

    unsafe {
        let gpu_domain = twiddles.itwiddles.device_ptr;
        let folded_values = CudaSecureColumn::new_with_size(n >> 1);

        bindings::fold_line(
            gpu_domain,
            twiddle_offset,
            n,
            CudaSecureColumn::from(&eval.values).device_ptr(),
            CudaSecureField::from(alpha),
            CudaSecureColumn::from(&folded_values).device_ptr(),
        );

        LineEvaluation::new(eval.domain().double(), folded_values)
    }
}

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alphas: &[SecureField],
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        assert!(!alphas.is_empty(), "fold_line: alphas must be non-empty");
        let mut cur = fold_line_single(eval, alphas[0], twiddles);
        for &alpha in &alphas[1..] {
            cur = fold_line_single(&cur, alpha, twiddles);
        }
        cur
    }

    fn fold_circle_into_line(
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let n = src.len();
        // 74951f79 returns a FRESH f' (the `dst = dst*alpha^2 + f'` accumulation moved to callers).
        // We allocate a ZERO dst so NitrooZK's accumulating kernel computes 0*alpha^2 + f' = f'.
        let line_log_size = src.domain.log_size() - 1;
        let dst_domain = LineDomain::new(Coset::half_odds(line_log_size));
        let dst_values = SecureColumnByCoords::<Self>::zeros(1 << line_log_size);
        let dst = LineEvaluation::new(dst_domain, dst_values);
        assert_eq!(n >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

        unsafe {
            let gpu_domain = twiddles.itwiddles.device_ptr;
            let twiddle_offset = twiddles.root_coset.size() - dst.domain().size();
            bindings::fold_circle_into_line(
                gpu_domain,
                twiddle_offset,
                n,
                CudaSecureColumn::from(&src.values).device_ptr(),
                CudaSecureField::from(alpha),
                CudaSecureColumn::from(&dst.values).device_ptr(),
            );
        }
        dst
    }

    fn decompose(
        _eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // This method will be deprecated and is no longer used in stwo. In stwo, every polynomial
        // that goes into FRI is in the FFT space already and there's no need to decompose
        // it.
        todo!()
    }
}
