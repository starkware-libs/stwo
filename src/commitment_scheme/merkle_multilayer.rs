use std::fmt::{self, Display};
use std::ops::{Deref, DerefMut};

use super::hasher::Hasher;

/// A MerkleMultiLayer represents multiple sequential merkle-tree layers, as a SubTreeMajor array of
/// hash values. Each SubTree is a balanced binary tree of height `sub_trees_height`.
/// Intended to be used as a layer of L1/L2 cache-sized sub-trees and commited on serially, and
/// multithreaded within the multilayer.
// TODO(Ohad): Implement get_layer_view() and get_layer_mut() for subtrees.
// TODO(Ohad): Implement .commit(), .decommit(), .roots() for MerklePolyLayer.
// TODO(Ohad): Add as an attribute of the merkle tree.
pub struct MerkleMultiLayer<H: Hasher> {
    data: Vec<H::Hash>,
    config: MerkleMultiLayerConfig,
}

impl<H: Hasher> MerkleMultiLayer<H> {
    pub fn new(config: MerkleMultiLayerConfig) -> Self {
        // TODO(Ohad): investigate if this is the best way to initialize the vector. Consider unsafe
        // implementation.
        let data = vec![H::Hash::default(); config.sub_tree_size * config.n_sub_trees];
        Self { data, config }
    }
}

impl<H: Hasher> Deref for MerkleMultiLayer<H> {
    type Target = Vec<H::Hash>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<H: Hasher> DerefMut for MerkleMultiLayer<H> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

// TODO(Ohad): change according to the future implementation of get_layer_view() and
// get_root().
impl<H: Hasher> Display for MerkleMultiLayer<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.chunks(self.config.sub_tree_size)
            .map(|sub_tree| MerkleMultiLayerSubTreeView::new(sub_tree))
            .collect::<Vec<MerkleMultiLayerSubTreeView<'_, H>>>()
            .into_iter()
            .enumerate()
            .for_each(|(i, it)| {
                f.write_str(&std::format!("\nSubTree #[{}]:", i)).unwrap();
                for (j, layer) in it.enumerate() {
                    f.write_str(&std::format!("\nLayer #[{}]:", j)).unwrap();
                    for (k, hash) in layer.iter().enumerate() {
                        f.write_str(&std::format!("\nHash #[{}]: {}", k, hash))
                            .unwrap();
                    }
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

/// Iterates over layers of a given tree's data.
pub struct MerkleMultiLayerSubTreeView<'a, H: Hasher> {
    pub data: &'a [H::Hash],
    cursor: usize,
}

impl<'a, H: Hasher> MerkleMultiLayerSubTreeView<'a, H> {
    pub fn new(data: &'a [H::Hash]) -> Self {
        assert!((data.len() + 1).is_power_of_two());
        Self { data, cursor: 0 }
    }
}

impl<'a, H: Hasher> Iterator for MerkleMultiLayerSubTreeView<'a, H> {
    type Item = &'a [H::Hash];

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.data.len() {
            None
        } else {
            let res =
                &self.data[self.cursor..self.cursor + (self.data.len() + 1 - self.cursor) / 2];
            self.cursor += (self.data.len() + 1 - self.cursor) / 2;
            Some(res)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;

    #[test]
    pub fn multi_layer_init_test() {
        let (sub_trees_height, n_sub_trees) = (4, 4);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, n_sub_trees);
        let sub_tree_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        assert_eq!(
            sub_tree_layer.data.len(),
            ((1 << sub_trees_height) - 1) * n_sub_trees
        );
    }

    #[test]
    pub fn multi_layer_display_test() {
        let (sub_trees_height, n_sub_trees) = (5, 2);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, n_sub_trees);
        let multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        println!("{}", multi_layer);
    }
}
