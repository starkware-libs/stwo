use std::cmp::Ordering;
use std::ops::Deref;

use num_traits::Zero;

use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;
use crate::core::fields::Field;

/// Domain comprising of the x-coordinates of points in a [Coset].
///
/// For use with univariate polynomials.
#[derive(Copy, Clone, Debug)]
pub struct LineDomain {
    coset: Coset,
}

impl LineDomain {
    /// Returns a domain comprising of the x-coordinates of points in a coset.
    ///
    /// # Panics
    ///
    /// Panics if the coset items don't have unique x-coordinates.
    pub fn new(coset: Coset) -> Self {
        match coset.size().cmp(&2) {
            Ordering::Less => {}
            Ordering::Equal => {
                // If the coset with two points contains (0, y) then the coset is {(0, y), (0, -y)}.
                assert!(!coset.initial.x.is_zero(), "coset x-coordinates not unique");
            }
            Ordering::Greater => {
                // Let our coset be `E = c + <G>` with `|E| > 2` then:
                // 1. if `ord(c) <= ord(G)` the coset contains two points at x=0
                // 2. if `ord(c) = 2 * ord(G)` then `c` and `-c` are in our coset
                assert!(
                    coset.initial.order_bits() >= coset.step.order_bits() + 2,
                    "coset x-coordinates not unique"
                );
            }
        }
        Self { coset }
    }

    /// Returns the `i`th domain element.
    pub fn at(&self, i: usize) -> BaseField {
        self.coset.at(i).x
    }

    /// Returns the size of the domain.
    pub fn size(&self) -> usize {
        self.coset.size()
    }

    /// Returns an iterator over elements in the domain.
    pub fn iter(&self) -> impl Iterator<Item = BaseField> {
        self.coset.iter().map(|p| p.x)
    }

    /// Returns a new domain comprising of all points in current domain doubled.
    pub fn double(&self) -> Self {
        Self {
            coset: self.coset.double(),
        }
    }

    /// Returns the domain's underlying coset.
    pub fn coset(&self) -> Coset {
        self.coset
    }
}

/// A univariate polynomial defined on a [LineDomain].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LinePoly<F> {
    /// Coefficients of the polynomial in the IFFT algorithm's basis.
    ///
    /// The coefficients are stored in bit-reversed order.
    coeffs: Vec<F>,
    /// The degree bound of the polynomial stored as `log2(degree_bound)`.
    ///
    /// The number of `coeffs` will always equal `2^bound_bits`.
    bound_bits: u32,
}

impl<F: Field> LinePoly<F> {
    /// Creates a new line polynomial from bit reversed coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients is not a power of two.
    pub fn new(coeffs: Vec<F>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        let bound_bits = coeffs.len().ilog2();
        Self { coeffs, bound_bits }
    }

    /// Evaluates the polynomial at a single point.
    pub fn eval_at_point(&self, _x: F) -> F {
        todo!()
    }

    /// Evaluates the polynomial at all points in the domain.
    pub fn evaluate(self, _domain: LineDomain) -> LineEvaluation<F> {
        todo!()
    }

    /// Returns the number of evaluations.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.coeffs.len(), 1 << self.bound_bits);
        1 << self.bound_bits
    }
}

impl<F: Field> Deref for LinePoly<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Vec<F> {
        &self.coeffs
    }
}

/// Evaluations of a univariate polynomial on a [LineDomain].
pub struct LineEvaluation<F> {
    _evals: Vec<F>,
    /// The degree bound of the polynomial stored as `log2(degree_bound)`.
    ///
    /// The number of `evals` will always equal `2^bound_bits`.
    _bound_bits: u32,
}

impl<F: Field> LineEvaluation<F> {
    /// Creates new [LineEvaluation] from a set of polynomial evaluations over a [LineDomain].
    ///
    /// # Panics
    ///
    /// Panics if the number of evaluations is not a power of two.
    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        Self {
            _bound_bits: evals.len().ilog2(),
            _evals: evals,
        }
    }

    /// Interpolates the polynomial as evaluations on `domain`.
    pub fn interpolate(self, _domain: LineDomain) -> LinePoly<F> {
        todo!()
    }

    /// Returns the number of evaluations.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self._evals.len(), 1 << self._bound_bits);
        1 << self._bound_bits
    }
}

impl<F: Field> Deref for LineEvaluation<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Vec<F> {
        &self._evals
    }
}

#[cfg(test)]
mod tests {
    use super::LineDomain;
    use crate::core::circle::{CirclePoint, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::line::{LineEvaluation, LinePoly};

    #[test]
    #[should_panic]
    fn bad_line_domain() {
        // This coset doesn't have points with unique x-coordinates.
        let coset = Coset::odds(2);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_of_size_two_works() {
        const LOG_N: usize = 1;
        let coset = Coset::subgroup(LOG_N);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_of_size_one_works() {
        const LOG_N: usize = 0;
        let coset = Coset::subgroup(LOG_N);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_size_is_correct() {
        const LOG_N: usize = 8;
        let coset = Coset::half_odds(LOG_N);
        let domain = LineDomain::new(coset);

        let size = domain.size();

        assert_eq!(size, 1 << LOG_N);
    }

    #[test]
    fn line_domain_coset_returns_the_coset() {
        let coset = Coset::half_odds(5);
        let domain = LineDomain::new(coset);

        assert_eq!(domain.coset(), coset);
    }

    #[test]
    fn line_domain_double_works() {
        const LOG_N: usize = 8;
        let coset = Coset::half_odds(LOG_N);
        let domain = LineDomain::new(coset);

        let doubled_domain = domain.double();

        assert_eq!(doubled_domain.size(), 1 << (LOG_N - 1));
        assert_eq!(doubled_domain.at(0), CirclePoint::double_x(domain.at(0)));
        assert_eq!(doubled_domain.at(1), CirclePoint::double_x(domain.at(1)));
    }

    #[test]
    fn line_domain_iter_works() {
        const LOG_N: usize = 8;
        let coset = Coset::half_odds(LOG_N);
        let domain = LineDomain::new(coset);

        let elements = domain.iter().collect::<Vec<BaseField>>();

        assert_eq!(elements.len(), domain.size());
        for (i, element) in elements.into_iter().enumerate() {
            assert_eq!(element, domain.at(i), "mismatch at {i}");
        }
    }

    #[test]
    #[ignore = "not implemented"]
    fn line_polynomial_evaluation() {
        let poly = LinePoly::new(vec![
            BaseField::from(7), // 7 * 1
            BaseField::from(9), // 9 * Φ(x)
            BaseField::from(5), // 5 * x
            BaseField::from(3), // 3 * Φ(x)*x
        ]);
        let coset = Coset::half_odds(poly.len().ilog2() as usize);
        let domain = LineDomain::new(coset);
        let expected_evals = domain
            .iter()
            .map(|x| {
                let phi_x = CirclePoint::double_x(x);
                poly[0] + poly[1] * phi_x + poly[2] * x + poly[3] * phi_x * x
            })
            .collect::<Vec<BaseField>>();

        let actual_evals = poly.evaluate(domain);

        assert_eq!(*actual_evals, expected_evals);
    }

    #[test]
    #[ignore = "not implemented"]
    fn line_evaluation_interpolation() {
        let poly = LinePoly::new(vec![
            BaseField::from(7), // 7 * 1
            BaseField::from(9), // 9 * Φ(x)
            BaseField::from(5), // 5 * x
            BaseField::from(3), // 3 * Φ(x)*x
        ]);
        let coset = Coset::half_odds(poly.len().ilog2() as usize);
        let domain = LineDomain::new(coset);
        let evals = LineEvaluation::new(
            domain
                .iter()
                .map(|x| {
                    let phi_x = CirclePoint::double_x(x);
                    poly[0] + poly[1] * phi_x + poly[2] * x + poly[3] * phi_x * x
                })
                .collect::<Vec<BaseField>>(),
        );

        let interpolated_poly = evals.interpolate(domain);

        assert_eq!(interpolated_poly, poly);
    }
}
