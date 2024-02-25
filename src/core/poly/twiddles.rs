use super::circle::PolyOps;
use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;

// TODO(spapini): If we decide to only have rectangular components, there will only be a single tree
// and we don't need a bank.
/// A bank holding precomputed [TwiddleTree]s for different coset towers.
/// A coset tower is every repeated doubling of a root coset
pub struct TwiddleBank<B: PolyOps<F>, F: ExtensionOf<BaseField>> {
    trees: Vec<TwiddleTree<B, F>>,
}
impl<B: PolyOps<F>, F: ExtensionOf<BaseField>> TwiddleBank<B, F> {
    pub fn get_tree(&self, coset: Coset) -> &TwiddleTree<B, F> {
        self.trees
            .iter()
            .find(|t| {
                t.root_coset.log_size() >= coset.log_size()
                    && t.root_coset
                        .repeated_double(t.root_coset.log_size() - coset.log_size())
                        == coset
            })
            .expect("No precomputed twiddles found.")
    }
    pub fn add_tree(&mut self, coset: Coset) {
        self.trees.push(B::precompute_twiddles(coset));
    }
}

/// Precomputed twiddles for a specific coset tower.
/// A coset tower is every repeated doubling of a root coset
pub struct TwiddleTree<B: PolyOps<F>, F: ExtensionOf<BaseField>> {
    pub root_coset: Coset,
    // TODO(spapini): Represent a slice, and grabbing, in a generic way
    pub twiddles: B::Twiddles,
    pub itwiddles: B::Twiddles,
}
