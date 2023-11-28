use std::cmp::Ordering;

use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;

/// Domain comprising of the x-coordinates of points in a [Coset].
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
                assert_ne!(
                    coset.at(0).x,
                    coset.at(1).x,
                    "coset x-coordinates are not unique"
                );
            }
            Ordering::Greater => {
                assert!(
                    coset.initial.order_bits() >= coset.step.order_bits() + 2,
                    "coset x-coordinates are not unique"
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
    // TODO: rename len() on cosets and domains to size() and remove is_empty() since you shouldn't
    // be able to create an empty coset
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

/// A polynomial defined on a [LineDomain].
#[derive(Clone, Debug)]
pub struct LinePoly {
    /// Coefficients of the polynomial in the IFFT algorithm's basis.
    ///
    /// These are not coefficients in the standard monomial basis but rather the tensor product of
    /// the twiddle factors i.e `{1} ⊗ {x} ⊗ {Φ(x)} ⊗ {Φ^2(x)} ⊗ ... ⊗ {Φ^{log(n)-2}(x)}`.
    ///
    /// The coefficients are stored in bit-reversed order.
    coeffs: Vec<BaseField>,
}

impl LinePoly {
    /// Creates a new line polynomial from a set of bit reversed coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients isn't a power of two.
    pub fn new(coeffs: Vec<BaseField>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        Self { coeffs }
    }

    /// Evaluates the polynomial at a point
    pub fn eval_at_point(&self, _x: BaseField) -> BaseField {
        todo!()
    }

    /// Evaluates the polynomial at all points in the domain.
    pub fn evaluate(self, _domain: LineDomain) -> LineEvaluation {
        todo!()
    }
}

/// Evaluations of a polynomial on a [LineDomain].
pub struct LineEvaluation {
    evals: Vec<BaseField>,
}

#[cfg(test)]
mod tests {
    use super::LineDomain;
    use crate::core::circle::Coset;

    #[test]
    #[should_panic]
    fn bad_line_domain() {
        // this coset doesn't have points with unique x-coordinates
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
        let line_domain = LineDomain::new(coset);

        let line_domain_len = line_domain.size();

        assert_eq!(line_domain_len, 1 << LOG_N);
    }
}
