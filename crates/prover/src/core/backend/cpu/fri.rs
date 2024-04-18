use super::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::{fold_circle_into_line, fold_line, FriOps};
use crate::core::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::NaturalOrder;

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

    fn coset_diff(eval: CircleEvaluation<Self, BaseField, NaturalOrder>) -> BaseField {
        let half_domain_size = 1 << (eval.domain.log_size() - 1);
        let (a_values, b_values) = eval.values.split_at(half_domain_size);
        let a_sum: BaseField = a_values.iter().sum();
        let b_sum: BaseField = b_values.iter().sum();
        a_sum - b_sum
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::backend::cpu::CPUCirclePoly;
    use crate::core::backend::CPUBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fri::FriOps;
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn coset_diff_out_fft_space_test() {
        let domain_log_size = 3;
        let evaluation_domain = CanonicCoset::new(domain_log_size).circle_domain();
        let coeffs_in_fft = [0, 1, 2, 3, 4, 5, 6, 0]
            .into_iter()
            .map(BaseField::from_u32_unchecked)
            .collect();
        let coeffs_out_fft = [0, 0, 0, 0, 0, 0, 0, 7]
            .into_iter()
            .map(BaseField::from_u32_unchecked)
            .collect();
        let combined_poly_coeffs = [0, 1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .map(BaseField::from_u32_unchecked)
            .collect();

        let in_fft_poly = CPUCirclePoly::new(coeffs_in_fft);
        let out_fft_poly = CPUCirclePoly::new(coeffs_out_fft);
        let combined_poly = CPUCirclePoly::new(combined_poly_coeffs);

        let in_lambda =
            CPUBackend::coset_diff(in_fft_poly.evaluate(evaluation_domain).bit_reverse());
        let out_lambda =
            CPUBackend::coset_diff(out_fft_poly.evaluate(evaluation_domain).bit_reverse());
        let combined_lambda =
            CPUBackend::coset_diff(combined_poly.evaluate(evaluation_domain).bit_reverse());

        assert_eq!(in_lambda, BaseField::zero());
        assert_eq!(out_lambda, combined_lambda);
    }
}
