use std::fmt::{self, Display};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use super::hasher::Hasher;
use super::merkle_input::{LayerColumns, MerkleTreeInput};
use super::merkle_polylayer_cfg::MerklePolyLayerConfig;
use crate::core::fields::{Field, IntoSlice};

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

    pub fn commit_layer_mt<F: Field + Sync + IntoSlice<H::NativeType>>(
        &mut self,
        input: &MerkleTreeInput<'_, F>,
        prev_hashes: Option<&[H::Hash]>,
    ) {
        if let Some(prev_hashes) = prev_hashes {
            let sub_layer_size = 1 << self.cfg.sub_tree_height;
            self.data
                .par_chunks_mut(self.cfg.sub_tree_size)
                .zip(prev_hashes.par_chunks(sub_layer_size))
                .enumerate()
                .for_each(|(i, (tree_data, prev_hashes_slice))| {
                    commit_subtree::<F, H>(
                        tree_data,
                        input,
                        Some(prev_hashes_slice),
                        &self.cfg,
                        i,
                        0,
                    );
                });
        } else {
            self.data
                .par_chunks_mut(self.cfg.sub_tree_size)
                .enumerate()
                .for_each(|(i, tree_data)| {
                    commit_subtree::<F, H>(tree_data, input, prev_hashes, &self.cfg, i, 0);
                });
        }
    }

    pub fn get_root_layer(&self) -> Vec<H::Hash> {
        self.data
            .chunks(self.cfg.sub_tree_size)
            .map(|c| c[0])
            .collect()
    }
}

pub fn get_layer_view_from_tree_data<H: Sized>(data: &[H], layer_idx: usize) -> &[H] {
    let len = 1 << layer_idx;
    let idx = len - 1;
    &data[idx..idx + len]
}

pub fn get_mut_layer_from_tree_data<H: Sized>(data: &mut [H], layer_idx: usize) -> &mut [H] {
    let len = 1 << layer_idx;
    let idx = len - 1;
    data.split_at_mut(idx).1.split_at_mut(len).0
}

fn commit_subtree<F: Field, H: Hasher>(
    sub_tree_data: &mut [H::Hash],
    input: &MerkleTreeInput<'_, F>,
    prev_hashes: Option<&[H::Hash]>,
    cfg: &MerklePolyLayerConfig,
    index_in_layer: usize,
    stop_at_layer: usize,
) where
    F: IntoSlice<H::NativeType>,
{
    // First layer is special, as it is the only layer that might have inputs from the previous
    // MultiLayer, and does not need to look at the current sub_tree for previous hash values.
    let mut hash_inputs = vec![Vec::new();1 << (cfg.sub_tree_height - 1)];
    if let Some(prev_hashes) = prev_hashes {
        assert_eq!(prev_hashes.len(), (1 << (cfg.sub_tree_height)));
        for (j, hashes) in prev_hashes.chunks(2).enumerate() {
            hash_inputs[j].push(hashes[0].as_ref());
            hash_inputs[j].push(hashes[1].as_ref());
        }
    }

    if let Some(columns) = input.get_columns(cfg.sub_tree_height) {
        inject_layer_columns::<H, F>(
            &mut hash_inputs,
            columns,
            index_in_layer,
            cfg.sub_tree_height,
            cfg.n_sub_trees,
        );
    }
    let dst = sub_tree_data
        .split_at_mut((1 << (cfg.sub_tree_height - 1)) - 1)
        .1;
    H::hash_many_multi_src_in_place(&hash_inputs, dst);

    // Rest of the layers.
    for hashed_layer_idx in (1..(cfg.sub_tree_height - stop_at_layer)).rev() {
        let hashed_layer_len = 1 << hashed_layer_idx;
        let produced_layer_len = hashed_layer_len >> 1;
        let mut hash_inputs = vec![Vec::new();produced_layer_len];
        let (s1, s2) = sub_tree_data.split_at_mut(hashed_layer_len - 1);
        let (dst, prev_hashes) = (
            &mut s1[produced_layer_len - 1..],
            &s2[..produced_layer_len * 2],
        );
        
        for (j, hashes) in prev_hashes.chunks(2).enumerate() {
            hash_inputs[j].push(hashes[0].as_ref());
            hash_inputs[j].push(hashes[1].as_ref());
        }

        if let Some(columns) = input.get_columns(hashed_layer_idx) {
            inject_layer_columns::<H, F>(
                &mut hash_inputs,
                columns,
                index_in_layer,
                hashed_layer_idx,
                cfg.n_sub_trees,
            );
        }
        H::hash_many_multi_src_in_place(&hash_inputs, dst);
    }
}

pub fn inject_layer_columns<'b, 'a: 'b, H: Hasher, F: Field>(
    hash_inputs: &'b mut [Vec<&'a [H::NativeType]>],
    columns: &'a LayerColumns<'a, F>,
    index_in_layer: usize,
    hashed_layer_depth: usize,
    n_sub_trees: usize,
) where
    F: IntoSlice<H::NativeType>,
{
    let produced_layer_len = 1 << (hashed_layer_depth - 1);
    for column in columns.iter() {
        let slice_length = column.len() / n_sub_trees;
        let slice_start_idx = slice_length * index_in_layer;
        let column_slice = &column[slice_start_idx..slice_start_idx + slice_length];
        let n_rows_in_inject = slice_length / produced_layer_len;
        column_slice
            .chunks(n_rows_in_inject)
            .enumerate()
            .for_each(|(m, chunk)| {
                hash_inputs[m].push(<F as IntoSlice<H::NativeType>>::into_slice(chunk));
            });
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
