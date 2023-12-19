use std::fmt::{self, Display};
use std::slice::Iter;

use super::hasher::{HashState, Hasher};
use crate::core::fields::{Field, IntoSlice};

/// A MerkleMultiLayer represents multiple sequential merkle-tree layers, as a SubTreeMajor array of
/// hash values. Each SubTree is a balanced binary tree of height `sub_trees_height`.
/// Intended to be used as a layer of L1/L2 cache-sized sub-trees and commited on serially, and
/// multithreaded within the multilayer.
// TODO(Ohad): Implement get_layer_view() and get_layer_mut() for subtrees.
// TODO(Ohad): Implement .commit(), .decommit() for MerkleMultiLayer.
// TODO(Ohad): Add as an attribute of the merkle tree.
// TODO(Ohad): Implement Iterator for MerkleMultiLayer.
pub struct MerkleMultiLayer<H: Hasher> {
    pub data: Vec<H::Hash>,
    config: MerkleMultiLayerConfig,
}

impl<H: Hasher> MerkleMultiLayer<H> {
    pub fn new(config: MerkleMultiLayerConfig) -> Self {
        // TODO(Ohad): investigate if this is the best way to initialize the vector. Consider unsafe
        // implementation.
        let data = vec![H::Hash::default(); config.sub_tree_size * config.n_sub_trees];
        Self { data, config }
    }

    /// Returns the roots of the sub-trees.
    pub fn get_roots(&self) -> Vec<H::Hash> {
        self.data
            .chunks(self.config.sub_tree_size)
            .map(|sub_tree| {
                sub_tree
                    .last()
                    .expect("Tried to extract roots but MerkleMultiLayer is empty!")
                    .to_owned()
            })
            .collect()
    }
}

fn _hash_layer<H: Hasher, F: Field, const IS_INTERMEDIATE: bool>(
    prev_hashes: &[H::Hash],
    dst: &mut [H::Hash],
    input_columns: &Iter<'_, &[F]>,
) where
    F: IntoSlice<H::NativeType>,
{
    let produced_layer_length = dst.len();
    if IS_INTERMEDIATE {
        assert_eq!(prev_hashes.len(), produced_layer_length * 2);
    }

    let mut hash_state = H::State::new();
    match input_columns.clone().peekable().next() {
        Some(_) => {
            // Match the input columns to corresponding chunk sizes, calculate once
            let input_columns: Vec<_> = input_columns
                .clone()
                .zip(
                    input_columns
                        .clone()
                        .map(|c| c.len() / produced_layer_length),
                )
                .collect();
            dst.iter_mut().enumerate().for_each(|(i, dst)| {
                if IS_INTERMEDIATE {
                    _inject_previous_hash_values::<H>(i, &mut hash_state, prev_hashes);
                }
                for (column, n_elements_in_chunk) in input_columns.iter() {
                    let chunk = &column[i * n_elements_in_chunk..(i + 1) * n_elements_in_chunk];
                    hash_state.update(F::into_slice(chunk));
                }
                *dst = hash_state.finalize_reset();
            });
        }
        None if !IS_INTERMEDIATE => {
            panic!("Tried to hash bottom layer without input columns!")
        }
        _ => {
            dst.iter_mut().enumerate().for_each(|(i, dst)| {
                _inject_previous_hash_values::<H>(i, &mut hash_state, prev_hashes);
                *dst = hash_state.finalize_reset();
            });
        }
    }
}

fn _inject_previous_hash_values<H: Hasher>(
    i: usize,
    hash_state: &mut <H as Hasher>::State,
    prev_hashes: &[<H as Hasher>::Hash],
) {
    hash_state.update(prev_hashes[i * 2].as_ref());
    hash_state.update(prev_hashes[i * 2 + 1].as_ref());
}

// TODO(Ohad): change according to the future implementation of get_layer_view() and
// get_root().
impl<H: Hasher> Display for MerkleMultiLayer<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data
            .chunks(self.config.sub_tree_size)
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
    use crate::commitment_scheme::hasher::Hasher;

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
        let (sub_trees_height, n_sub_trees) = (8, 8);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, n_sub_trees);
        let multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        println!("{}", multi_layer);
    }

    #[test]
    pub fn get_roots_test() {
        let (sub_trees_height, n_sub_trees) = (4, 4);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, n_sub_trees);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        multi_layer
            .data
            .chunks_mut(multi_layer.config.sub_tree_size)
            .enumerate()
            .for_each(|(i, sub_tree)| {
                sub_tree[sub_tree.len() - 1] = Blake3Hasher::hash(&i.to_le_bytes());
            });

        let roots = multi_layer.get_roots();

        assert_eq!(roots.len(), n_sub_trees);
        roots
            .iter()
            .enumerate()
            .for_each(|(i, r)| assert_eq!(r, &Blake3Hasher::hash(&i.to_le_bytes())));
    }
}
