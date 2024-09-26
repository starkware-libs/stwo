use num_traits::One;

use super::circle::{CirclePoint, Coset};
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::ExtensionOf;
use super::pcs::quotients::PointSample;
use crate::core::fields::ComplexConjugate;

/// Evaluates a vanishing polynomial of the coset at a point.
pub fn coset_vanishing<F: ExtensionOf<BaseField>>(coset: Coset, mut p: CirclePoint<F>) -> F {
    // Doubling a point `log_order - 1` times and taking the x coordinate is
    // essentially evaluating a polynomial in x of degree `2^(log_order - 1)`. If
    // the entire `2^log_order` points of the coset are roots (i.e. yield 0), then
    // this is a vanishing polynomial of these points.

    // Rotating the coset -coset.initial + step / 2 yields a canonic coset:
    // `step/2 + <step>.`
    // Doubling this coset log_order - 1 times yields the coset +-G_4.
    // The polynomial x vanishes on these points.
    // ```text
    //   X
    // .   .
    //   X
    // ```
    p = p - coset.initial.into_ef() + coset.step_size.half().to_point().into_ef();
    let mut x = p.x;

    // The formula for the x coordinate of the double of a point.
    for _ in 1..coset.log_size {
        x = CirclePoint::double_x(x);
    }
    x
}

/// Evaluates the polynomial that is used to exclude the excluded point at point
/// p. Note that this polynomial has a zero of multiplicity 2 at the excluded
/// point.
pub fn point_excluder<F: ExtensionOf<BaseField>>(
    excluded: CirclePoint<BaseField>,
    p: CirclePoint<F>,
) -> F {
    (p - excluded.into_ef()).x - BaseField::one()
}

// A vanishing polynomial on 2 circle points.
pub fn pair_vanishing<F: ExtensionOf<BaseField>>(
    excluded0: CirclePoint<F>,
    excluded1: CirclePoint<F>,
    p: CirclePoint<F>,
) -> F {
    // The algorithm check computes the area of the triangle formed by the
    // 3 points. This is done using the determinant of:
    // | p.x  p.y  1 |
    // | e0.x e0.y 1 |
    // | e1.x e1.y 1 |
    // This is a polynomial of degree 1 in p.x and p.y, and thus it is a line.
    // It vanishes at e0 and e1.
    (excluded0.y - excluded1.y) * p.x
        + (excluded1.x - excluded0.x) * p.y
        + (excluded0.x * excluded1.y - excluded0.y * excluded1.x)
}

/// Evaluates a vanishing polynomial of the vanish_point at a point.
/// Note that this function has a pole on the antipode of the vanish_point.
pub fn point_vanishing<F: ExtensionOf<BaseField>, EF: ExtensionOf<F>>(
    vanish_point: CirclePoint<F>,
    p: CirclePoint<EF>,
) -> EF {
    let h = p - vanish_point.into_ef();
    h.y / (EF::one() + h.x)
}

/// Evaluates a point on a line between a point and its complex conjugate.
/// Relies on the fact that every polynomial F over the base field holds:
/// F(p*) == F(p)* (* being the complex conjugate).
pub fn complex_conjugate_line(
    point: CirclePoint<SecureField>,
    value: SecureField,
    p: CirclePoint<BaseField>,
) -> SecureField {
    // TODO(AlonH): This assertion will fail at a probability of 1 to 2^62. Use a better solution.
    assert_ne!(
        point.y,
        point.y.complex_conjugate(),
        "Cannot evaluate a line with a single point ({point:?})."
    );
    value
        + (value.complex_conjugate() - value) * (-point.y + p.y)
            / (point.complex_conjugate().y - point.y)
}

/// Evaluates the coefficients of a line between a point and its complex conjugate.
///
/// Specifically, `a, b, and c, s.t. a*x + b -c*y = 0` for (x,y) being (sample.y, sample.value) and
/// (conj(sample.y), conj(sample.value)). Relies on the fact that every polynomial F over the base
/// field holds: `F(p*) == F(p)*` (`*` being the complex conjugate).
pub fn complex_conjugate_line_coeffs(
    sample: &PointSample,
    alpha: SecureField,
) -> (SecureField, SecureField, SecureField) {
    // TODO(AlonH): This assertion will fail at a probability of 1 to 2^62. Use a better solution.
    assert_ne!(
        sample.point.y,
        sample.point.y.complex_conjugate(),
        "Cannot evaluate a line with a single point ({:?}).",
        sample.point
    );
    let a = sample.value.complex_conjugate() - sample.value;
    let c = sample.point.complex_conjugate().y - sample.point.y;
    let b = sample.value * c - a * sample.point.y;
    (alpha * a, alpha * b, alpha * c)
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::{coset_vanishing, point_excluder, point_vanishing};
    use crate::core::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
    use crate::core::constraints::{complex_conjugate_line, pair_vanishing};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::{ComplexConjugate, FieldExpOps};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::poly::NaturalOrder;
    use crate::core::test_utils::secure_eval_to_base_eval;
    use crate::m31;

    #[test]
    fn test_coset_vanishing() {
        let cosets = [
            Coset::half_odds(5),
            Coset::odds(5),
            Coset::new(CirclePointIndex::zero(), 5),
            Coset::half_odds(5).conjugate(),
        ];
        for c0 in cosets.iter() {
            for el in c0.iter() {
                assert_eq!(coset_vanishing(*c0, el), BaseField::zero());
                for c1 in cosets.iter() {
                    if c0 == c1 {
                        continue;
                    }
                    assert_ne!(coset_vanishing(*c1, el), BaseField::zero());
                }
            }
        }
    }

    #[test]
    fn test_point_excluder() {
        let excluded = Coset::half_odds(5).at(10);
        let point = (CirclePointIndex::generator() * 4).to_point();

        let num = point_excluder(excluded, point) * point_excluder(excluded.conjugate(), point);
        let denom = (point.x - excluded.x).pow(2);

        assert_eq!(num, denom);
    }

    #[test]
    fn test_pair_excluder() {
        let excluded0 = Coset::half_odds(5).at(10);
        let excluded1 = Coset::half_odds(5).at(13);
        let point = (CirclePointIndex::generator() * 4).to_point();

        assert_ne!(pair_vanishing(excluded0, excluded1, point), M31::zero());
        assert_eq!(pair_vanishing(excluded0, excluded1, excluded0), M31::zero());
        assert_eq!(pair_vanishing(excluded0, excluded1, excluded1), M31::zero());
    }

    #[test]
    fn test_point_vanishing_success() {
        let coset = Coset::odds(5);
        let vanish_point = coset.at(2);
        for el in coset.iter() {
            if el == vanish_point {
                assert_eq!(point_vanishing(vanish_point, el), BaseField::zero());
                continue;
            }
            if el == vanish_point.antipode() {
                continue;
            }
            assert_ne!(point_vanishing(vanish_point, el), BaseField::zero());
        }
    }

    #[test]
    #[should_panic(expected = "0 has no inverse")]
    fn test_point_vanishing_failure() {
        let coset = Coset::half_odds(6);
        let point = coset.at(4);
        point_vanishing(point, point.antipode());
    }

    #[test]
    fn test_complex_conjugate_symmetry() {
        // Create a polynomial over a base circle domain.
        let polynomial = CpuCirclePoly::new((0..1 << 7).map(|i| m31!(i)).collect());
        let oods_point = CirclePoint::get_point(9834759221);

        // Assert that the base field polynomial is complex conjugate symmetric.
        assert_eq!(
            polynomial.eval_at_point(oods_point.complex_conjugate()),
            polynomial.eval_at_point(oods_point).complex_conjugate()
        );
    }

    #[test]
    fn test_point_vanishing_degree() {
        // Create a polynomial over a circle domain.
        let log_domain_size = 7;
        let domain_size = 1 << log_domain_size;
        let polynomial = CpuCirclePoly::new((0..domain_size).map(|i| m31!(i)).collect());

        // Create a larger domain.
        let log_large_domain_size = log_domain_size + 1;
        let large_domain_size = 1 << log_large_domain_size;
        let large_domain = CanonicCoset::new(log_large_domain_size).circle_domain();

        // Create a vanish point that is not in the large domain.
        let vanish_point = CirclePoint::get_point(97);
        let vanish_point_value = polynomial.eval_at_point(vanish_point);

        // Compute the quotient polynomial.
        let mut quotient_polynomial_values = Vec::with_capacity(large_domain_size as usize);
        for point in large_domain.iter() {
            let line = complex_conjugate_line(vanish_point, vanish_point_value, point);
            let mut value = polynomial.eval_at_point(point.into_ef()) - line;
            value /= pair_vanishing(
                vanish_point,
                vanish_point.complex_conjugate(),
                point.into_ef(),
            );
            quotient_polynomial_values.push(value);
        }
        let quotient_evaluation = CpuCircleEvaluation::<SecureField, NaturalOrder>::new(
            large_domain,
            quotient_polynomial_values,
        );
        let quotient_polynomial = secure_eval_to_base_eval(&quotient_evaluation)
            .bit_reverse()
            .interpolate();

        // Check that the quotient polynomial is indeed in the wanted fft space.
        assert!(quotient_polynomial.is_in_fft_space(log_domain_size));
    }
}
