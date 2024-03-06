use std::iter::zip;

use super::CPUBackend;
use crate::core::fft::ibutterfly;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, Field, FieldExpOps};
use crate::core::fri::{FriOps, CIRCLE_TO_LINE_FOLD_STEP, FOLD_STEP};
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;

impl FriOps for CPUBackend {
    fn fold_line(
        eval: &LineEvaluation<Self, SecureField, BitReversedOrder>,
        alpha: SecureField,
    ) -> LineEvaluation<Self, SecureField, BitReversedOrder> {
        let n = eval.len();
        assert!(n >= 2, "Evaluation too small");

        let domain = eval.domain();

        let folded_values = eval
            .values
            .array_chunks()
            .enumerate()
            .map(|(i, &[f_x, f_neg_x])| {
                // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
                let x = domain.at(bit_reverse_index(i << FOLD_STEP, domain.log_size()));

                let (mut f0, mut f1) = (f_x, f_neg_x);
                ibutterfly(&mut f0, &mut f1, x.inverse());
                f0 + alpha * f1
            })
            .collect();

        LineEvaluation::new(domain.double(), folded_values)
    }
    fn fold_circle_into_line<F: Field>(
        dst: &mut LineEvaluation<Self, SecureField, BitReversedOrder>,
        src: &CircleEvaluation<Self, F, BitReversedOrder>,
        alpha: SecureField,
    ) where
        F: ExtensionOf<BaseField>,
        SecureField: ExtensionOf<F> + Field,
    {
        assert_eq!(src.len() >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

        let domain = src.domain;
        let alpha_sq = alpha * alpha;

        zip(&mut dst.values, src.array_chunks())
            .enumerate()
            .for_each(|(i, (dst, &[f_p, f_neg_p]))| {
                // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
                let p = domain.at(bit_reverse_index(
                    i << CIRCLE_TO_LINE_FOLD_STEP,
                    domain.log_size(),
                ));

                // Calculate `f0(px)` and `f1(px)` such that `2f(p) = f0(px) + py * f1(px)`.
                let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
                ibutterfly(&mut f0_px, &mut f1_px, p.y.inverse());
                let f_prime = alpha * f1_px + f0_px;

                *dst = *dst * alpha_sq + f_prime;
            });
    }
}
