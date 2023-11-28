use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;

/// Domain comprising of the x-coordinates of points in a [Coset]
#[derive(Copy, Clone, Debug)]
pub struct LineDomain {
    coset: Coset,
}

impl LineDomain {
    /// Returns a domain comprising of the x-coordinates of points in a coset
    ///
    /// # Panics
    ///
    /// * if the coset items don't all have unique x-coordinates
    /// * if the coset lies entirely on the x-axis.
    pub fn new(coset: Coset) -> Self {
        assert!(coset.initial().order_bits() >= coset.step.order_bits() + 2);
        Self { coset }
    }

    /// Returns the `i`th domain element
    ///
    /// # Panics
    ///
    /// Panics if the index exceeds the size of the domain
    pub fn at(&self, i: usize) -> BaseField {
        let n = self.len();
        assert!(i < n, "the size of the domain is {n} but index is {i}");
        self.coset.at(i).x
    }

    /// Returns the number of elements in the domain
    // TODO: size might be a more appropriate name
    // a line domain can never be empty
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.coset.len()
    }

    /// Returns an iterator over elements in the domain
    pub fn iter(&self) -> impl Iterator<Item = BaseField> {
        self.coset.iter().map(|p| p.x)
    }
}

#[cfg(test)]
mod tests {
    use super::LineDomain;
    use crate::core::circle::Coset;

    #[test]
    #[should_panic]
    fn bad_line_domain() {
        // the coset points don't have unique x-coordinates
        let coset = Coset::odds(8);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_has_correct_size() {
        const N: usize = 8;
        let coset = Coset::half_odds(N);

        let line_domain = LineDomain::new(coset);

        assert_eq!(line_domain.len(), N);
    }
}
