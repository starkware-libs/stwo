use crate::core::circle::Coset;

/// Domain defined as the x-coordinates of all points in a [CircleDomain]
#[derive(Copy, Clone, Debug)]
pub struct LineDomain {
    coset: Coset,
}

impl LineDomain {
    pub fn new(coset: Coset) {
        assert!(n.is_power_of_two());
        assert!(n as u32 <= CircleGroup::ORDER / 4);
        let g = CircleGroup::point_with_order(4 * n as u32);
        let coset = Coset::new(g.double().double(), g);
        Self::from_coset(coset)
    }

    /// Returns the `i`th domain element
    // TODO: could use Index trait
    pub fn at(&self, i: usize) -> BaseField {
        let n = self.len().get();
        assert!(i < n, "the len is {n} but index is {i}");
        self.coset.at(i).x
    }

    /// Returns the number of elements in the domain
    // TODO: size might be a better name
    pub fn len(&self) -> usize {
        // half the size because a [CircleDomain] is symmetric across the x-axis
        // there's always at least one item in the domain
        NonZeroUsize::new(self.0.len() / 2).unwrap()
    }

    /// Returns an iterator over elements in the domain
    pub fn iter(&self) -> impl Iterator<Item = BaseField> {
        self.0.iter().take(self.len().get()).map(|p| p.x)
    }
}
