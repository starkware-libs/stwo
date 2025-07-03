use itertools::Itertools;

use crate::core::fft::ibutterfly;
use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;
use crate::core::poly::line::{LineDomain, LinePoly};
use crate::prover::backend::{ColumnOps, CpuBackend};
use crate::prover::secure_column::SecureColumnByCoords;

// Evaluations of a univariate polynomial on a [LineDomain].
// TODO(andrew): Remove EvalOrder. Bit-reversed evals are only necessary since LineEvaluation is
// only used by FRI where evaluations are in bit-reversed order.
// TODO(andrew): Remove pub.
#[derive(Clone, Debug)]
pub struct LineEvaluation<B: ColumnOps<BaseField>> {
    /// Evaluations of a univariate polynomial on `domain`.
    pub values: SecureColumnByCoords<B>,
    domain: LineDomain,
}

impl<B: ColumnOps<BaseField>> LineEvaluation<B> {
    /// Creates new [LineEvaluation] from a set of polynomial evaluations over a [LineDomain].
    ///
    /// # Panics
    ///
    /// Panics if the number of evaluations does not match the size of the domain.
    pub fn new(domain: LineDomain, values: SecureColumnByCoords<B>) -> Self {
        assert_eq!(values.len(), domain.size());
        Self { values, domain }
    }

    pub fn new_zero(domain: LineDomain) -> Self {
        Self::new(domain, SecureColumnByCoords::zeros(domain.size()))
    }

    /// Returns the number of evaluations.
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        1 << self.domain.log_size()
    }

    pub const fn domain(&self) -> LineDomain {
        self.domain
    }

    /// Clones the values into a new line evaluation in the CPU.
    pub fn to_cpu(&self) -> LineEvaluation<CpuBackend> {
        LineEvaluation::new(self.domain, self.values.to_cpu())
    }
}

impl LineEvaluation<CpuBackend> {
    /// Interpolates the polynomial as evaluations on `domain`.
    pub fn interpolate(self) -> LinePoly {
        let mut values = self.values.into_iter().collect_vec();
        CpuBackend::bit_reverse_column(&mut values);
        line_ifft(&mut values, self.domain);
        // Normalize the coefficients.
        let len_inv = BaseField::from(values.len()).inverse();
        values.iter_mut().for_each(|v| *v *= len_inv);
        LinePoly::new(values)
    }
}

/// Performs a univariate IFFT on a polynomial's evaluation over a [LineDomain].
///
/// This is not the standard univariate IFFT, because [LineDomain] is not a cyclic group.
///
/// The transform happens in-place. `values` should be the evaluations of a polynomial over `domain`
/// in their natural order. After the transformation `values` becomes the coefficients of the
/// polynomial stored in bit-reversed order.
///
/// For performance reasons and flexibility the normalization of the coefficients is omitted. The
/// normalized coefficients can be obtained by scaling all coefficients by `1 / len(values)`.
///
/// This algorithm does not return coefficients in the standard monomial basis but rather returns
/// coefficients in a basis relating to the circle's x-coordinate doubling map `pi(x) = 2x^2 - 1`
/// i.e.
///
/// ```text
/// B = { 1 } * { x } * { pi(x) } * { pi(pi(x)) } * ...
///   = { 1, x, pi(x), pi(x) * x, pi(pi(x)), pi(pi(x)) * x, pi(pi(x)) * pi(x), ... }
/// ```
///
/// # Panics
///
/// Panics if the number of values doesn't match the size of the domain.
fn line_ifft<F: ExtensionOf<BaseField> + Copy>(values: &mut [F], mut domain: LineDomain) {
    assert_eq!(values.len(), domain.size());
    while domain.size() > 1 {
        for chunk in values.chunks_exact_mut(domain.size()) {
            let (l, r) = chunk.split_at_mut(domain.size() / 2);
            for (i, x) in domain.iter().take(domain.size() / 2).enumerate() {
                ibutterfly(&mut l[i], &mut r[i], x.inverse());
            }
        }
        domain = domain.double();
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::circle::{CirclePoint, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::line::LineDomain;
    use crate::core::utils::bit_reverse_index;
    use crate::prover::backend::{ColumnOps, CpuBackend};
    use crate::prover::line::{LineEvaluation, LinePoly};

    #[test]
    fn line_evaluation_interpolation() {
        let coeffs = vec![
            BaseField::from(7).into(), // 7 * 1
            BaseField::from(9).into(), // 9 * pi(x)
            BaseField::from(5).into(), // 5 * x
            BaseField::from(3).into(), // 3 * pi(x)*x
        ];
        let poly = LinePoly::new(coeffs.clone());
        let coset = Coset::half_odds(poly.len().ilog2());
        let domain = LineDomain::new(coset);
        let mut values = domain
            .iter()
            .map(|x| {
                let pi_x = CirclePoint::double_x(x);
                coeffs[0] + coeffs[1] * pi_x + coeffs[2] * x + coeffs[3] * pi_x * x
            })
            .collect_vec();
        CpuBackend::bit_reverse_column(&mut values);
        let evals = LineEvaluation::<CpuBackend>::new(domain, values.into_iter().collect());

        let interpolated_poly = evals.interpolate();
        let mut coeffs = interpolated_poly.into_ordered_coefficients();
        CpuBackend::bit_reverse_column(&mut coeffs);

        assert_eq!(coeffs, coeffs);
    }

    #[test]
    fn line_polynomial_eval_at_point() {
        const LOG_SIZE: u32 = 2;
        let coset = Coset::half_odds(LOG_SIZE);
        let domain = LineDomain::new(coset);
        let evals = LineEvaluation::<CpuBackend>::new(
            domain,
            (0..1 << LOG_SIZE)
                .map(BaseField::from)
                .map(|x| x.into())
                .collect(),
        );
        let poly = evals.clone().interpolate();

        for (i, x) in domain.iter().enumerate() {
            assert_eq!(
                poly.eval_at_point(x.into()),
                evals.values.at(bit_reverse_index(i, domain.log_size())),
                "mismatch at {i}"
            );
        }
    }
}
