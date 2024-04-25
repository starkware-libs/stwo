use super::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
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

    fn coset_diff(eval: &SecureEvaluation<Self>) -> SecureField {
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

    fn decompose(eval: &SecureEvaluation<Self>) -> (SecureEvaluation<Self>, SecureField) {
        let domain_half_size = 1 << (eval.domain.log_size() - 1);
        let lambda = Self::coset_diff(eval);

        // g = f -+ lambda.
        let g_values: SecureColumn<CPUBackend> = eval
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                if i < domain_half_size {
                    x - lambda
                } else {
                    x + lambda
                }
            })
            .collect();
        let g = SecureEvaluation {
            domain: eval.domain,
            values: g_values,
        };

        (g, lambda)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
    use crate::core::backend::CPUBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumn;
    use crate::core::fri::FriOps;
    use crate::core::poly::circle::{CanonicCoset, SecureEvaluation};
    use crate::m31;

    #[test]
    fn decomposition_test() {
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

            let (g, lambda) = CPUBackend::decompose(&secure_eval.clone());

            // Sanity check.
            assert_ne!(lambda, SecureField::zero());

            // Assert the new polynomial is in the FFT space.
            for i in 0..4 {
                let basefield_column = g.columns[i].clone();
                let eval = CPUCircleEvaluation::new(domain, basefield_column);
                let coeffs = eval.interpolate().coeffs;
                assert!(CPUCirclePoly::new(coeffs).is_in_fft_space(domain_log_half_size));
            }
        }
    }
}
