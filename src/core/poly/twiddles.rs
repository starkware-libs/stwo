use super::circle::PolyOps;
use crate::core::circle::Coset;

/// Precomputed twiddles for a specific coset tower.
/// A coset tower is every repeated doubling of a root coset
pub struct TwiddleTree<B: PolyOps> {
    pub root_coset: Coset,
    // TODO(spapini): Represent a slice, and grabbing, in a generic way
    pub twiddles: B::Twiddles,
    pub itwiddles: B::Twiddles,
}
