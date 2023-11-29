use std::cmp::Ordering;
use std::ops::Deref;

use num_traits::Zero;

use super::utils::fold;
use crate::core::circle::{CirclePoint, Coset};
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
        match coset.len().cmp(&2) {
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
    ///
    /// # Panics
    ///
    /// Panics if the index exceeds the size of the domain.
    pub fn at(&self, i: usize) -> BaseField {
        let n = self.size();
        assert!(i < n, "the size of the domain is {n} but index is {i}");
        self.coset.at(i).x
    }

    /// Returns the number of elements in the domain.
    // TODO(Andrew): Rename len() on cosets and domains to size() and remove is_empty() since you
    // shouldn't be able to create an empty coset.
    pub fn size(&self) -> usize {
        self.coset.len()
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
}

/// A univariate polynomial defined on a [LineDomain].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LinePoly<F> {
    /// Coefficients of the polynomial in the IFFT algorithm's basis.
    ///
    /// These are not coefficients in the standard monomial basis but rather the tensor product of
    /// the twiddle factors i.e `{1} ⊗ {x} ⊗ {Φ(x)} ⊗ {Φ^2(x)} ⊗ ... ⊗ {Φ^{log(n)-2}(x)}`.
    ///
    /// The coefficients are stored in bit-reversed order.
    coeffs: Vec<F>,
}

impl<F: Field> LinePoly<F> {
    /// Creates a new line polynomial from bit reversed coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients isn't a power of two.
    pub fn new(coeffs: Vec<F>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        Self { coeffs }
    }

    /// Evaluates the polynomial at a single point.
    pub fn eval_at_point(&self, mut x: F) -> F {
        // TODO(Andrew): Allocation here expensive for small polynomials.
        let mut twiddle_factors = vec![x];
        for _ in 2..self.coeffs.len().ilog2() {
            x = CirclePoint::double_x(x);
            twiddle_factors.push(x);
        }
        fold(&self.coeffs, &twiddle_factors)
    }

    /// Evaluates the polynomial at all points in the domain.
    pub fn evaluate(self, _domain: LineDomain) -> LineEvaluation<F> {
        todo!()
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
}

impl<F: Field> LineEvaluation<F> {
    // TODO: docs
    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        Self { _evals: evals }
    }

    pub fn interpolate(&self, _domain: LineDomain) -> LinePoly<F> {
        todo!()
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

        assert_eq!(elements.len(), 1 << LOG_N);
        for (i, element) in elements.into_iter().enumerate() {
            assert_eq!(element, domain.at(i), "mismatch at {i}");
        }
    }

    #[test]
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

    #[test]
    fn line_polynomial_eval_at_point() {
        const LOG_N: usize = 8;
        let coset = Coset::half_odds(LOG_N);
        let evals = LineEvaluation::new((0..1 << LOG_N).map(BaseField::from).collect());
        let domain = LineDomain::new(coset);
        let poly = evals.interpolate(domain);

        for (i, x) in domain.iter().enumerate() {
            assert_eq!(poly.eval_at_point(x), evals[i], "mismatch at {i}");
        }
    }
}
