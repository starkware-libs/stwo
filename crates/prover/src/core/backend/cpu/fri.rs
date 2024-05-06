use super::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
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

impl CPUBackend {
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
    // TODO(Ohad): remove pub.
    pub fn decomposition_coefficient(eval: &SecureEvaluation<Self>) -> SecureField {
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

    use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
    use crate::core::backend::CPUBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::secure_column::SecureColumn;
    use crate::core::poly::circle::{CanonicCoset, SecureEvaluation};
    use crate::m31;

    #[test]
    fn decompose_coeff_out_fft_space_test() {
        for domain_log_size in 5..12 {
            let domain_log_half_size = domain_log_size - 1;
            let s = CanonicCoset::new(domain_log_size);
            let domain = s.circle_domain();

            let mut coeffs = vec![BaseField::zero(); 1 << domain_log_size];

            // Polynomial is out of FFT space.
            coeffs[1 << domain_log_half_size] = m31!(1);
            assert!(!CPUCirclePoly::new(coeffs.clone()).is_in_fft_space(domain_log_half_size));

            let poly = CPUCirclePoly::new(coeffs);
            let values = poly.evaluate(domain);
            let secure_column = SecureColumn {
                columns: [
                    values.values.clone(),
                    values.values.clone(),
                    values.values.clone(),
                    values.values.clone(),
                ],
            };
            let secure_eval = SecureEvaluation::<CPUBackend> {
                domain,
                values: secure_column.clone(),
            };

            let lambda = CPUBackend::decomposition_coefficient(&secure_eval.clone());

            // isolate the polynomial that is inside of the fft-space.
            // TODO(Ohad): this should be a function in itself.
            let q_x_values: SecureColumn<CPUBackend> = secure_column
                .into_iter()
                .enumerate()
                .map(|(i, x)| {
                    if i < (1 << domain_log_half_size) {
                        x - lambda
                    } else {
                        x + lambda
                    }
                })
                .collect();

            // Assert the new polynomial is in the FFT space.
            for i in 0..4 {
                let basefield_column = q_x_values.columns[i].clone();
                let eval = CPUCircleEvaluation::new(domain, basefield_column);
                let coeffs = eval.interpolate().coeffs;
                assert!(CPUCirclePoly::new(coeffs).is_in_fft_space(domain_log_half_size));
            }
        }
    }
}
