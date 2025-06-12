use std::cmp::Ordering;
use std::fmt::Debug;
use std::iter::Map;
use std::ops::{Deref, DerefMut};

use num_traits::Zero;
use serde::{Deserialize, Serialize};

use super::circle::CircleDomain;
use crate::core::circle::{CirclePoint, Coset, CosetIterator};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::utils::fold;
use crate::prover::backend::cpu::bit_reverse;

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
                    coset.initial.log_order() >= coset.step.log_order() + 2,
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
    pub const fn size(&self) -> usize {
        self.coset.size()
    }

    /// Returns the log size of the domain.
    pub const fn log_size(&self) -> u32 {
        self.coset.log_size()
    }

    /// Returns an iterator over elements in the domain.
    pub fn iter(&self) -> LineDomainIterator {
        self.coset.iter().map(|p| p.x)
    }

    /// Returns a new domain comprising of all points in current domain doubled.
    pub fn double(&self) -> Self {
        Self {
            coset: self.coset.double(),
        }
    }

    /// Returns the domain's underlying coset.
    pub const fn coset(&self) -> Coset {
        self.coset
    }
}

impl IntoIterator for LineDomain {
    type Item = BaseField;
    type IntoIter = LineDomainIterator;

    /// Returns an iterator over elements in the domain.
    fn into_iter(self) -> LineDomainIterator {
        self.iter()
    }
}

impl From<CircleDomain> for LineDomain {
    fn from(domain: CircleDomain) -> Self {
        Self {
            coset: domain.half_coset,
        }
    }
}

/// An iterator over the x-coordinates of points in a coset.
type LineDomainIterator =
    Map<CosetIterator<CirclePoint<BaseField>>, fn(CirclePoint<BaseField>) -> BaseField>;

/// A univariate polynomial defined on a [LineDomain].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub struct LinePoly {
    /// Coefficients of the polynomial in [`line_ifft`] algorithm's basis.
    ///
    /// The coefficients are stored in bit-reversed order.
    #[allow(rustdoc::private_intra_doc_links)]
    coeffs: Vec<SecureField>,
    /// The number of coefficients stored as `log2(len(coeffs))`.
    log_size: u32,
}

impl LinePoly {
    /// Creates a new line polynomial from bit reversed coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients is not a power of two.
    pub fn new(coeffs: Vec<SecureField>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        let log_size = coeffs.len().ilog2();
        Self { coeffs, log_size }
    }

    /// Evaluates the polynomial at a single point.
    pub fn eval_at_point(&self, mut x: SecureField) -> SecureField {
        let mut doublings = Vec::new();
        for _ in 0..self.log_size {
            doublings.push(x);
            x = CirclePoint::double_x(x);
        }
        fold(&self.coeffs, &doublings)
    }

    /// Returns the number of coefficients.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        // `.len().ilog2()` is a common operation. By returning the length like so the compiler
        // optimizes `.len().ilog2()` to a load of `log_size` instead of a branch and a bit count.
        debug_assert_eq!(self.coeffs.len(), 1 << self.log_size);
        1 << self.log_size
    }

    /// Returns the polynomial's coefficients in their natural order.
    pub fn into_ordered_coefficients(mut self) -> Vec<SecureField> {
        bit_reverse(&mut self.coeffs);
        self.coeffs
    }

    /// Creates a new line polynomial from coefficients in their natural order.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients is not a power of two.
    pub fn from_ordered_coefficients(mut coeffs: Vec<SecureField>) -> Self {
        bit_reverse(&mut coeffs);
        Self::new(coeffs)
    }
}

impl Deref for LinePoly {
    type Target = [SecureField];

    fn deref(&self) -> &[SecureField] {
        &self.coeffs
    }
}

impl DerefMut for LinePoly {
    fn deref_mut(&mut self) -> &mut [SecureField] {
        &mut self.coeffs
    }
}
#[cfg(test)]
mod tests {

    use super::LineDomain;
    use crate::core::circle::{CirclePoint, Coset};
    use crate::core::fields::m31::BaseField;

    #[test]
    #[should_panic]
    fn bad_line_domain() {
        // This coset doesn't have points with unique x-coordinates.
        let coset = Coset::odds(2);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_of_size_two_works() {
        const LOG_SIZE: u32 = 1;
        let coset = Coset::subgroup(LOG_SIZE);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_of_size_one_works() {
        const LOG_SIZE: u32 = 0;
        let coset = Coset::subgroup(LOG_SIZE);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_size_is_correct() {
        const LOG_SIZE: u32 = 8;
        let coset = Coset::half_odds(LOG_SIZE);
        let domain = LineDomain::new(coset);

        let size = domain.size();

        assert_eq!(size, 1 << LOG_SIZE);
    }

    #[test]
    fn line_domain_coset_returns_the_coset() {
        let coset = Coset::half_odds(5);
        let domain = LineDomain::new(coset);

        assert_eq!(domain.coset(), coset);
    }

    #[test]
    fn line_domain_double_works() {
        const LOG_SIZE: u32 = 8;
        let coset = Coset::half_odds(LOG_SIZE);
        let domain = LineDomain::new(coset);

        let doubled_domain = domain.double();

        assert_eq!(doubled_domain.size(), 1 << (LOG_SIZE - 1));
        assert_eq!(doubled_domain.at(0), CirclePoint::double_x(domain.at(0)));
        assert_eq!(doubled_domain.at(1), CirclePoint::double_x(domain.at(1)));
    }

    #[test]
    fn line_domain_iter_works() {
        const LOG_SIZE: u32 = 8;
        let coset = Coset::half_odds(LOG_SIZE);
        let domain = LineDomain::new(coset);

        let elements = domain.iter().collect::<Vec<BaseField>>();

        assert_eq!(elements.len(), domain.size());
        for (i, element) in elements.into_iter().enumerate() {
            assert_eq!(element, domain.at(i), "mismatch at {i}");
        }
    }
}
