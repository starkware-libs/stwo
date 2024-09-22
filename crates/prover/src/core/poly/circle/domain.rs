use std::iter::Chain;

use itertools::Itertools;

use crate::core::circle::{
    CirclePoint, CirclePointIndex, Coset, CosetIterator, M31_CIRCLE_LOG_ORDER,
};
use crate::core::fields::m31::BaseField;

pub const MAX_CIRCLE_DOMAIN_LOG_SIZE: u32 = M31_CIRCLE_LOG_ORDER - 1;

/// A valid domain for circle polynomial interpolation and evaluation.
///
/// Valid domains are a disjoint union of two conjugate cosets: `+-C + <G_n>`.
/// The ordering defined on this domain is `C + iG_n`, and then `-C - iG_n`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CircleDomain {
    pub half_coset: Coset,
}

impl CircleDomain {
    /// Given a coset C + <G_n>, constructs the circle domain +-C + <G_n> (i.e.,
    /// this coset and its conjugate).
    pub const fn new(half_coset: Coset) -> Self {
        Self { half_coset }
    }

    pub fn iter(&self) -> CircleDomainIterator {
        self.half_coset
            .iter()
            .chain(self.half_coset.conjugate().iter())
    }

    /// Iterates over point indices.
    pub fn iter_indices(&self) -> CircleDomainIndexIterator {
        self.half_coset
            .iter_indices()
            .chain(self.half_coset.conjugate().iter_indices())
    }

    /// Returns the size of the domain.
    pub const fn size(&self) -> usize {
        1 << self.log_size()
    }

    /// Returns the log size of the domain.
    pub const fn log_size(&self) -> u32 {
        self.half_coset.log_size + 1
    }

    /// Returns the `i` th domain element.
    pub fn at(&self, i: usize) -> CirclePoint<BaseField> {
        self.index_at(i).to_point()
    }

    /// Returns the [CirclePointIndex] of the `i`th domain element.
    pub fn index_at(&self, i: usize) -> CirclePointIndex {
        if i < self.half_coset.size() {
            self.half_coset.index_at(i)
        } else {
            -self.half_coset.index_at(i - self.half_coset.size())
        }
    }

    pub fn find(&self, i: CirclePointIndex) -> Option<usize> {
        if let Some(d) = self.half_coset.find(i) {
            return Some(d);
        }
        if let Some(d) = self.half_coset.conjugate().find(i) {
            return Some(self.half_coset.size() + d);
        }
        None
    }

    /// Returns true if the domain is canonic.
    ///
    /// Canonic domains are domains with elements that are the entire set of points defined by
    /// `G_2n + <G_n>` where `G_n` and `G_2n` are obtained by repeatedly doubling
    /// [crate::core::circle::M31_CIRCLE_GEN].
    pub fn is_canonic(&self) -> bool {
        self.half_coset.initial_index * 4 == self.half_coset.step_size
    }

    /// Splits a circle domain into a smaller [CircleDomain]s, shifted by offsets.
    pub fn split(&self, log_parts: u32) -> (CircleDomain, Vec<CirclePointIndex>) {
        assert!(log_parts <= self.half_coset.log_size);
        let subdomain = CircleDomain::new(Coset::new(
            self.half_coset.initial_index,
            self.half_coset.log_size - log_parts,
        ));
        let shifts = (0..1 << log_parts)
            .map(|i| self.half_coset.step_size * i)
            .collect_vec();
        (subdomain, shifts)
    }

    pub fn shift(&self, shift: CirclePointIndex) -> CircleDomain {
        CircleDomain::new(self.half_coset.shift(shift))
    }
}

impl IntoIterator for CircleDomain {
    type Item = CirclePoint<BaseField>;
    type IntoIter = CircleDomainIterator;

    /// Iterates over the points in the domain.
    fn into_iter(self) -> CircleDomainIterator {
        self.iter()
    }
}

/// An iterator over points in a circle domain.
///
/// Let the domain be `+-c + <G>`. The first iterated points are `c + <G>`, then `-c + <-G>`.
pub type CircleDomainIterator =
    Chain<CosetIterator<CirclePoint<BaseField>>, CosetIterator<CirclePoint<BaseField>>>;

/// Like [CircleDomainIterator] but returns corresponding [CirclePointIndex]s.
type CircleDomainIndexIterator =
    Chain<CosetIterator<CirclePointIndex>, CosetIterator<CirclePointIndex>>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::CircleDomain;
    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn test_circle_domain_iterator() {
        let domain = CircleDomain::new(Coset::new(CirclePointIndex::generator(), 2));
        for (i, point) in domain.iter().enumerate() {
            if i < 4 {
                assert_eq!(
                    point,
                    (CirclePointIndex::generator() + CirclePointIndex::subgroup_gen(2) * i)
                        .to_point()
                );
            } else {
                assert_eq!(
                    point,
                    (-(CirclePointIndex::generator() + CirclePointIndex::subgroup_gen(2) * i))
                        .to_point()
                );
            }
        }
    }

    #[test]
    fn is_canonic_invalid_domain() {
        let half_coset = Coset::new(CirclePointIndex::generator(), 4);
        let not_canonic_domain = CircleDomain::new(half_coset);

        assert!(!not_canonic_domain.is_canonic());
    }

    #[test]
    fn test_at_circle_domain() {
        let domain = CanonicCoset::new(7).circle_domain();
        let half_domain_size = domain.size() / 2;

        for i in 0..half_domain_size {
            assert_eq!(domain.index_at(i), -domain.index_at(i + half_domain_size));
            assert_eq!(domain.at(i), domain.at(i + half_domain_size).conjugate());
        }
    }

    #[test]
    fn test_domain_split() {
        let domain = CanonicCoset::new(5).circle_domain();
        let (subdomain, shifts) = domain.split(2);

        let domain_points = domain.iter().collect::<Vec<_>>();
        let points_for_each_domain = shifts
            .iter()
            .map(|&shift| (subdomain.shift(shift)).iter().collect_vec())
            .collect::<Vec<_>>();
        // Interleave the points from each subdomain.
        let extended_points = (0..(1 << 3))
            .flat_map(|point_ind| {
                (0..(1 << 2))
                    .map(|shift_ind| points_for_each_domain[shift_ind][point_ind])
                    .collect_vec()
            })
            .collect_vec();
        assert_eq!(domain_points, extended_points);
    }
}
