use std::fmt::{self, Display};

use super::hasher::Hasher;

/// A MerkleMultiLayer represents multiple sequential merkle-tree layers, as a SubTreeMajor array of
/// hash values. Each SubTree is a balanced binary tree of height `sub_trees_height`.
/// Intended to be used as a layer of L1/L2 cache-sized sub-trees and commited on serially, and
/// multithreaded within the multilayer.
// TODO(Ohad): Implement get_layer_view() and get_layer_mut() for subtrees.
// TODO(Ohad): Implement .commit(), .decommit(), .roots() for MerklePolyLayer.
// TODO(Ohad): Add as an attribute of the merkle tree.
pub struct MerkleMultiLayer<H: Hasher> {
    pub data: Vec<H::Hash>,
    cfg: MerkleMultiLayerConfig,
}

impl<H: Hasher> MerkleMultiLayer<H> {
    pub fn new(cfg: MerkleMultiLayerConfig) -> Self {
        // TODO(Ohad): investigate if this is the best way to initialize the vector. Consider unsafe
        // implementation.
        let data = vec![H::Hash::default(); cfg.sub_tree_size * cfg.n_sub_trees];
        Self { data, cfg }
    }
}

impl<H: Hasher> Display for MerkleMultiLayer<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data
            .chunks(self.cfg.sub_tree_size)
            .enumerate()
            .for_each(|(i, c)| {
                f.write_str(&std::format!("\nSubTree #[{}]:", i)).unwrap();
                for (i, h) in c.iter().enumerate() {
                    f.write_str(&std::format!("\nNode #[{}]: {}", i, h))
                        .unwrap();
                }
            });
        Ok(())
    }
}

pub struct MerkleMultiLayerConfig {
    pub n_sub_trees: usize,
    pub sub_tree_height: usize,
    pub sub_tree_size: usize,
}

impl MerkleMultiLayerConfig {
    pub fn new(sub_tree_height: usize, n_sub_trees: usize) -> Self {
        let sub_tree_size = (1 << sub_tree_height) - 1;
        Self {
            n_sub_trees,
            sub_tree_height,
            sub_tree_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;

    #[test]
    pub fn multi_layer_init_test() {
        let (sub_trees_height, amount) = (4, 4);
        let cfg = super::MerkleMultiLayerConfig::new(sub_trees_height, amount);
        let sub_tree_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(cfg);
        assert_eq!(
            sub_tree_layer.data.len(),
            ((1 << sub_trees_height) - 1) * amount
        );
    }

    #[test]
    pub fn multi_layer_display_test() {
        let (sub_trees_height, amount) = (4, 4);
        let cfg = super::MerkleMultiLayerConfig::new(sub_trees_height, amount);
        let multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(cfg);
        println!("{}", multi_layer);
    }
}
