use itertools::Itertools;

use super::circle::PolyOps;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::CpuBackend;
use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;

/// Precomputed twiddles for a specific coset tower.
/// A coset tower is every repeated doubling of a `root_coset`.
/// The largest CircleDomain that can be ffted using these twiddles is one with `root_coset` as
/// its `half_coset`.
pub struct TwiddleTree<B: PolyOps> {
    pub root_coset: Coset,
    // TODO(spapini): Represent a slice, and grabbing, in a generic way
    pub twiddles: B::Twiddles,
    pub itwiddles: B::Twiddles,
}

impl TwiddleTree<SimdBackend> {
    pub fn to_cpu(&self) -> TwiddleTree<CpuBackend> {
        TwiddleTree {
            root_coset: self.root_coset,
            twiddles: self
                .twiddles
                .iter()
                .map(|x| BaseField::from(x >> 1))
                .collect_vec(),
            itwiddles: self
                .itwiddles
                .iter()
                .map(|x| BaseField::from(x >> 1))
                .collect_vec(),
        }
    }
}
