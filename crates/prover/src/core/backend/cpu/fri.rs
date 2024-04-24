use stwo_verifier::core::fields::qm31::SecureField;

use super::CPUBackend;
use crate::core::fri::{fold_circle_into_line, fold_line, FriOps};
use crate::core::poly::circle::SecureEvaluation;
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::twiddles::TwiddleTree;

// TODO(spapini): Optimized these functions as well.
impl FriOps for CPUBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        fold_line(eval, alpha)
    }
    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) {
        fold_circle_into_line(dst, src, alpha)
    }
}
