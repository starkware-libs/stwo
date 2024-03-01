use super::CircleDomain;
use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
use crate::core::fields::m31::BaseField;

/// A coset of the form G_{2n} + <G_n>, where G_n is the generator of the
/// subgroup of order n. The ordering on this coset is G_2n + i * G_n.
/// These cosets can be used as a [CircleDomain], and be interpolated on.
/// Note that this changes the ordering on the coset to be like [CircleDomain],
/// which is G_2n + i * G_n/2 and then -G_2n -i * G_n/2.
/// For example, the Xs below are a canonic coset with n=8.
/// ```text
///    X O X
///  O       O
/// X         X
/// O         O
/// X         X
///  O       O
///    X O X
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CanonicCoset {
    pub coset: Coset,
}

impl CanonicCoset {
    pub fn new(log_size: u32) -> Self {
        assert!(log_size > 0);
        Self {
            coset: Coset::odds(log_size),
        }
    }

    /// Gets the full coset represented G_{2n} + <G_n>.
    pub fn coset(&self) -> Coset {
        self.coset
    }

    /// Gets half of the coset (its conjugate complements to the whole coset), G_{2n} + <G_{n/2}>
    pub fn half_coset(&self) -> Coset {
        Coset::half_odds(self.log_size() - 1)
    }

    /// Gets the [CircleDomain] representing the same point set (in another order).
    pub fn circle_domain(&self) -> CircleDomain {
        CircleDomain::new(Coset::half_odds(self.coset.log_size - 1))
    }

    /// Gets a good [CircleDomain] for extension of a poly defined on this coset.
    /// The reason the domain looks like this is a bit more intricate, and not covered here.
    pub fn evaluation_domain(&self, log_size: u32) -> CircleDomain {
        assert!(log_size > self.coset.log_size);
        // TODO(spapini): Document why this is like this.
        CircleDomain::new(Coset::new(
            CirclePointIndex::generator() + CirclePointIndex::subgroup_gen(self.coset.log_size + 1),
            log_size - 1,
        ))
    }

    /// Returns the log size of the coset.
    pub fn log_size(&self) -> u32 {
        self.coset.log_size
    }

    /// Returns the size of the coset.
    pub fn size(&self) -> usize {
        self.coset.size()
    }

    pub fn initial_index(&self) -> CirclePointIndex {
        self.coset.initial_index
    }

    pub fn step_size(&self) -> CirclePointIndex {
        self.coset.step_size
    }

    pub fn index_at(&self, index: usize) -> CirclePointIndex {
        self.coset.index_at(index)
    }

    pub fn at(&self, i: usize) -> CirclePoint<BaseField> {
        self.coset.at(i)
    }
}
