use super::CpuBackend;
use crate::core::fft::ibutterfly;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::{CIRCLE_TO_LINE_FOLD_STEP, FOLD_STEP};
use crate::core::secure_column::SecureColumnByCoords;
use crate::core::utils::bit_reverse_index;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

impl FriOps for CpuBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        fold_line_cpu(eval, alpha)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) {
        fold_circle_into_line_cpu(dst, src, alpha)
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        let lambda = Self::decomposition_coefficient(eval);
        let mut g_values = unsafe { SecureColumnByCoords::<Self>::uninitialized(eval.len()) };

        let domain_size = eval.len();
        let half_domain_size = domain_size / 2;

        for i in 0..half_domain_size {
            let x = eval.values.at(i);
            let val = x - lambda;
            g_values.set(i, val);
        }
        for i in half_domain_size..domain_size {
            let x = eval.values.at(i);
            let val = x + lambda;
            g_values.set(i, val);
        }

        let g = SecureEvaluation::new(eval.domain, g_values);
        (g, lambda)
    }
}

/// TODO: Almost duplicate code of [`crate::core::fri::fold_line`]. Consider refactor.
pub fn fold_line_cpu(
    eval: &LineEvaluation<CpuBackend>,
    alpha: SecureField,
) -> LineEvaluation<CpuBackend> {
    let n = eval.len();
    assert!(n >= 2, "Evaluation too small");

    let domain = eval.domain();

    let folded_values = eval
        .values
        .into_iter()
        .array_chunks()
        .enumerate()
        .map(|(i, [f_x, f_neg_x])| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let x = domain.at(bit_reverse_index(i << FOLD_STEP, domain.log_size()));

            let (mut f0, mut f1) = (f_x, f_neg_x);
            ibutterfly(&mut f0, &mut f1, x.inverse());
            f0 + alpha * f1
        })
        .collect();

    LineEvaluation::new(domain.double(), folded_values)
}

/// TODO: Almost duplicate code of [`crate::core::fri::fold_circle_into_line`]. Consider refactor.
pub fn fold_circle_into_line_cpu(
    dst: &mut LineEvaluation<CpuBackend>,
    src: &SecureEvaluation<CpuBackend, BitReversedOrder>,
    alpha: SecureField,
) {
    assert_eq!(src.len() >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

    let domain = src.domain;
    let alpha_sq = alpha * alpha;

    src.values
        .into_iter()
        .array_chunks()
        .enumerate()
        .for_each(|(i, [f_p, f_neg_p])| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let p = domain.at(bit_reverse_index(
                i << CIRCLE_TO_LINE_FOLD_STEP,
                domain.log_size(),
            ));

            // Calculate `f0(px)` and `f1(px)` such that `2f(p) = f0(px) + py * f1(px)`.
            let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
            ibutterfly(&mut f0_px, &mut f1_px, p.y.inverse());
            let f_prime = alpha * f1_px + f0_px;

            dst.values.set(i, dst.values.at(i) * alpha_sq + f_prime)
        });
}

impl CpuBackend {
    /// Used to decompose a general polynomial to a polynomial inside the fft-space, and
    /// the remainder terms.
    /// A coset-diff on a [`CirclePoly`] that is in the FFT space will return zero.
    ///
    /// Let N be the domain size, Let h be a coset size N/2. Using lemma #7 from the CircleStark
    /// paper, <f,V_h> = lambda<V_h,V_h> = lambda\*N => lambda = f(0)\*V_h(0) + f(1)*V_h(1) + .. +
    /// f(N-1)\*V_h(N-1). The Vanishing polynomial of a cannonic coset sized half the circle
    /// domain,evaluated on the circle domain, is [(1, -1, -1, 1)] repeating. This becomes
    /// alternating [+-1] in our NaturalOrder, and [(+, +, +, ... , -, -)] in bit reverse.
    /// Explicitly, lambda\*N = sum(+f(0..N/2)) + sum(-f(N/2..)).
    ///
    /// # Warning
    /// This function assumes the blowupfactor is 2
    ///
    /// [`CirclePoly`]: crate::core::poly::circle::CirclePoly
    fn decomposition_coefficient(eval: &SecureEvaluation<Self, BitReversedOrder>) -> SecureField {
        let domain_size = 1 << eval.domain.log_size();
        let half_domain_size = domain_size / 2;

        // eval is in bit-reverse, hence all the positive factors are in the first half, opposite to
        // the latter.
        let a_sum = (0..half_domain_size)
            .map(|i| eval.values.at(i))
            .sum::<SecureField>();
        let b_sum = (half_domain_size..domain_size)
            .map(|i| eval.values.at(i))
            .sum::<SecureField>();

        // lambda = sum(+-f(p)) / 2N.
        (a_sum - b_sum) / BaseField::from_u32_unchecked(domain_size as u32)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::secure_column::SecureColumnByCoords;
    use crate::m31;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::CpuBackend;
    use crate::prover::fri::FriOps;
    use crate::prover::poly::circle::SecureEvaluation;
    use crate::prover::poly::BitReversedOrder;

    #[test]
    fn decompose_coeff_out_fft_space_test() {
        for domain_log_size in 5..12 {
            let domain_log_half_size = domain_log_size - 1;
            let s = CanonicCoset::new(domain_log_size);
            let domain = s.circle_domain();

            let mut coeffs = vec![BaseField::zero(); 1 << domain_log_size];

            // Polynomial is out of FFT space.
            coeffs[1 << domain_log_half_size] = m31!(1);
            assert!(!CpuCirclePoly::new(coeffs.clone()).is_in_fft_space(domain_log_half_size));

            let poly = CpuCirclePoly::new(coeffs);
            let values = poly.evaluate(domain);
            let secure_column = SecureColumnByCoords {
                columns: [
                    values.values.clone(),
                    values.values.clone(),
                    values.values.clone(),
                    values.values.clone(),
                ],
            };
            let secure_eval = SecureEvaluation::<CpuBackend, BitReversedOrder>::new(
                domain,
                secure_column.clone(),
            );

            let (g, lambda) = CpuBackend::decompose(&secure_eval);

            // Sanity check.
            assert_ne!(lambda, SecureField::zero());

            // Assert the new polynomial is in the FFT space.
            for i in 0..4 {
                let basefield_column = g.columns[i].clone();
                let eval = CpuCircleEvaluation::new(domain, basefield_column);
                let coeffs = eval.interpolate().coeffs;
                assert!(CpuCirclePoly::new(coeffs).is_in_fft_space(domain_log_half_size));
            }
        }
    }
}
