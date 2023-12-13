use std::fmt::{self, Display};

use super::hasher::Hasher;
use super::merkle_polylayer_cfg::MerklePolyLayerConfig;

/// A MerklePolyLayer represents multiple sequential merkle-tree layers, as a SubTreeMajor array of
/// hash values. Each SubTree is a balanced binary tree of height `sub_trees_height`.
/// Intended to be used as a layer of L1/L2 cache-sized sub-trees and commited on serially, and
/// multithreaded within the polylayer.
// TODO(Ohad): Implement get_layer_view() and get_layer_mut() for subtrees.
// TODO(Ohad): Implement .commit(), .decommit(), .roots() for MerklePolyLayer.
// TODO(Ohad): Add as an attribute of the merkle tree.
pub struct MerklePolyLayer<H: Hasher> {
    pub data: Vec<H::Hash>,
    cfg: MerklePolyLayerConfig,
}

impl<H: Hasher> MerklePolyLayer<H> {
    pub fn new(cfg: MerklePolyLayerConfig) -> Self {
        // TODO(Ohad): investigate if this is the best way to initialize the vector. Consider unsafe
        // implementation.
        let vec = vec![H::Hash::default(); cfg.sub_tree_size * cfg.n_sub_trees];
        Self { data: vec, cfg }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn sub_tree_layer_init_test() {
        let (sub_trees_height, amount) = (4, 4);
        let cfg = super::MerklePolyLayerConfig::new(sub_trees_height, amount);
        let sub_tree_layer =
            super::MerklePolyLayer::<crate::commitment_scheme::blake3_hash::Blake3Hasher>::new(cfg);
        assert!(sub_tree_layer.data.len() == ((1 << sub_trees_height) - 1) * amount);
    }

    #[test]
    pub fn sub_tree_layer_display_test() {
        let (sub_trees_height, amount) = (4, 4);
        let cfg = super::MerklePolyLayerConfig::new(sub_trees_height, amount);
        let sub_tree_layer =
            super::MerklePolyLayer::<crate::commitment_scheme::blake3_hash::Blake3Hasher>::new(cfg);
        println!("{}", sub_tree_layer);
    }
}

impl<H: Hasher> Display for MerklePolyLayer<H> {
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
