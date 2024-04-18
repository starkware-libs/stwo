use num_traits::Zero;

use super::circle::CircleEvaluation;
use super::line::LineDomain;
use super::BitReversedOrder;
use crate::core::backend::{Backend, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, Field};
use crate::core::utils::bit_reverse_index;

/// Folds values recursively in `O(n)` by a hierarchical application of folding factors.
///
/// i.e. folding `n = 8` values with `folding_factors = [x, y, z]`:
///
/// ```text
///               n2=n1+x*n2
///           /               \
///     n1=n3+y*n4          n2=n5+y*n6
///      /      \            /      \
/// n3=a+z*b  n4=c+z*d  n5=e+z*f  n6=g+z*h
///   /  \      /  \      /  \      /  \
///  a    b    c    d    e    f    g    h
/// ```
///
/// # Panics
///
/// Panics if the number of values is not a power of two or if an incorrect number of of folding
/// factors is provided.
// TODO(Andrew): Can be made to run >10x faster by unrolling lower layers of recursion
pub fn fold<F: Field, E: ExtensionOf<F>>(values: &[F], folding_factors: &[E]) -> E {
    let n = values.len();
    assert_eq!(n, 1 << folding_factors.len());
    if n == 1 {
        return values[0].into();
    }
    let (lhs_values, rhs_values) = values.split_at(n / 2);
    let (&folding_factor, folding_factors) = folding_factors.split_first().unwrap();
    let lhs_val = fold(lhs_values, folding_factors);
    let rhs_val = fold(rhs_values, folding_factors);
    lhs_val + rhs_val * folding_factor
}

/// Repeats each value sequentially `duplicity` many times.
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(repeat_value(&[1, 2, 3], 2), vec![1, 1, 2, 2, 3, 3]);
/// ```
pub fn repeat_value<T: Copy>(values: &[T], duplicity: usize) -> Vec<T> {
    let n = values.len();
    let mut res: Vec<T> = Vec::with_capacity(n * duplicity);

    // Fill each chunk with its corresponding value.
    for &v in values {
        for _ in 0..duplicity {
            res.push(v)
        }
    }

    res
}

/// Computes the difference in evaluations on the  [`CircleDomain`]'s half-coset and it's
/// conjugate. Used to decompose a general polynomial to a polynomial inside the fft-space, and
/// the remainder terms.
/// A coset-diff on a [`CirclePoly`] that is in the FFT space will return zero.
///
/// [`CircleDomain`]: super::circle::CircleDomain
/// [`CirclePoly`]: super::circle::CirclePoly
pub fn coset_diff<B: Backend>(eval: CircleEvaluation<B, BaseField, BitReversedOrder>) -> BaseField {
    let domain_log_size = eval.domain.log_size();
    eval.values
        .to_cpu()
        .iter()
        .enumerate()
        .fold(BaseField::zero(), |acc, (i, &x)| {
            // The wanted result is the sum of the evaluations at the first half of the domain
            // minus the second. The evaluations are bit reversed, so we need to
            // compare the un-bit-reversed indices against the domain size.
            let idx = bit_reverse_index(i, domain_log_size);
            if idx < (1 << (domain_log_size - 1)) {
                acc + x
            } else {
                acc - x
            }
        })
}

/// Computes the line twiddles for a [`CircleDomain`] or a [`LineDomain`] from the precomputed
/// twiddles tree.
///
/// [`CircleDomain`]: super::circle::CircleDomain
pub fn domain_line_twiddles_from_tree<T>(
    domain: impl Into<LineDomain>,
    twiddle_buffer: &[T],
) -> Vec<&[T]> {
    (0..domain.into().coset().log_size())
        .map(|i| {
            let len = 1 << i;
            &twiddle_buffer[twiddle_buffer.len() - len * 2..twiddle_buffer.len() - len]
        })
        .rev()
        .collect()
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::repeat_value;
    use crate::core::backend::cpu::CPUCirclePoly;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::poly::utils::coset_diff;

    #[test]
    fn repeat_value_0_times_works() {
        assert!(repeat_value(&[1, 2, 3], 0).is_empty());
    }

    #[test]
    fn repeat_value_2_times_works() {
        assert_eq!(repeat_value(&[1, 2, 3], 2), vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn repeat_value_3_times_works() {
        assert_eq!(repeat_value(&[1, 2], 3), vec![1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn coset_diff_out_fft_space_test() {
        let domain_log_size = 3;
        let evaluation_domain = CanonicCoset::new(domain_log_size).circle_domain();

        // f(x,y) = in(x,y) + out(x,y).
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

        let in_lambda = coset_diff(in_fft_poly.evaluate(evaluation_domain));
        let out_lambda = coset_diff(out_fft_poly.evaluate(evaluation_domain));
        let combined_lambda = coset_diff(combined_poly.evaluate(evaluation_domain));

        assert_eq!(in_lambda, BaseField::zero());
        assert_eq!(out_lambda, combined_lambda);
    }
}
