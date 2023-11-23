use std::collections::BTreeSet;
use std::fmt::{Debug, Display};

use super::hasher::Hasher;
use super::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::utils::{
    allocate_balanced_tree, column_to_row_major, hash_merkle_tree_from_bottom_layer, to_byte_slice,
    tree_data_as_mut_ref, ColumnArray, TreeData,
};

pub struct MerkleTree<T: Sized + Debug + Default + Display, H: Hasher> {
    pub bottom_layer: Vec<T>,
    pub bottom_layer_block_size: usize,
    pub bottom_layer_n_rows_in_node: usize,
    pub data: TreeData,
    pub height: usize,
    phantom: std::marker::PhantomData<H>,
}

impl<T: Sized + Copy + Default + Debug + Display, H: Hasher> MerkleTree<T, H> {
    /// Commits on a given trace(matrix).
    pub fn commit(trace: ColumnArray<T>) -> Self {
        let mut tree = Self::init_from_column_array(trace);

        let bottom_layer_as_byte_slice = to_byte_slice(&tree.bottom_layer);
        hash_merkle_tree_from_bottom_layer::<H>(
            bottom_layer_as_byte_slice,
            tree.bottom_layer_block_size * std::mem::size_of::<T>(),
            &mut tree_data_as_mut_ref(&mut tree.data)[..],
        );

        tree
    }

    /// Builds the base layer of the tree from the given trace.
    /// Allocates the rest of the tree.
    // TODO(Ohad): add support for columns of different lengths.
    fn init_from_column_array(trace: ColumnArray<T>) -> Self {
        assert!(!trace.is_empty());
        assert!(trace[0].len().is_power_of_two());
        trace.iter().for_each(|column| {
            assert_eq!(column.len(), trace[0].len());
        });

        let n_rows_in_node = std::cmp::min(
            crate::math::prev_pow_two(
                H::BLOCK_SIZE_IN_BYTES / (trace.len() * std::mem::size_of::<T>()),
            ),
            trace[0].len(),
        );

        let bottom_layer_block_size = n_rows_in_node * trace.len();
        let bottom_layer = column_to_row_major(trace);

        // Allocate rest of the tree.
        let bottom_layer_length_nodes =
            crate::math::usize_div_ceil(bottom_layer.len(), bottom_layer_block_size);
        let tree_data = allocate_balanced_tree(
            bottom_layer_length_nodes,
            H::BLOCK_SIZE_IN_BYTES,
            H::OUTPUT_SIZE_IN_BYTES,
        );

        Self {
            bottom_layer,
            bottom_layer_block_size,
            bottom_layer_n_rows_in_node: n_rows_in_node,
            height: tree_data.len() + 1, // +1 for the bottom layer.
            data: tree_data,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn root(&self) -> H::Hash {
        (&self.data.last().unwrap()[..]).into()
    }

    pub fn generate_decommitment(&self, queries: BTreeSet<usize>) -> MerkleDecommitment<T, H> {
        let mut proof = MerkleDecommitment::<T, H>::new();
        let leaf_block_indices: BTreeSet<usize> = queries
            .iter()
            .map(|query| query / self.bottom_layer_n_rows_in_node)
            .collect();

        // Input layer, every leaf-block holds 'bottom_layer_block_size' elements.
        leaf_block_indices.iter().for_each(|block_index| {
            proof.leaves.push(self.get_leaf_block(*block_index));
        });

        // Sorted indices of the current layer.
        let mut curr_layer_indices = leaf_block_indices
            .iter()
            .map(|index| index ^ 1)
            .collect::<BTreeSet<usize>>();
        for i in 0..self.height - 2 {
            let mut proof_layer = Vec::<H::Hash>::with_capacity(curr_layer_indices.len());
            let mut indices_iterator = curr_layer_indices.iter().peekable();
            while let Some(q) = indices_iterator.next() {
                let mut f = || -> Option<_> {
                    match indices_iterator.peek() {
                        // If both childs are in the layer, no extra data is needed to calculate
                        // parent.
                        Some(next_q) if *q % 2 == 0 && *q + 1 == **next_q => {
                            indices_iterator.next();
                            None
                        }
                        _ => {
                            let node: H::Hash = self.data[i]
                                [*q * H::OUTPUT_SIZE_IN_BYTES..(*q + 1) * H::OUTPUT_SIZE_IN_BYTES]
                                .try_into()
                                .expect("Error converting to hash object!");
                            Some(node)
                        }
                    }
                };
                if let Some(node) = f() {
                    proof_layer.push(node);
                }
            }
            proof.layers.push(proof_layer);

            // Next layer indices are the parents' siblings.
            curr_layer_indices = curr_layer_indices
                .iter()
                .map(|index| (index / 2) ^ 1)
                .collect();
        }
        proof
    }

    fn get_leaf_block(&self, block_index: usize) -> Vec<T> {
        assert!(block_index * self.bottom_layer_block_size < self.bottom_layer.len());
        Vec::from(
            &self.bottom_layer[block_index * self.bottom_layer_block_size
                ..(block_index + 1) * self.bottom_layer_block_size],
        )
    }
}
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rand::{thread_rng, Rng};

    use crate::commitment_scheme::blake3_hash::*;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::utils::to_byte_slice;
    use crate::core::fields::m31::M31;

    fn init_m31_test_trace(len: usize) -> Vec<M31> {
        assert!(len.is_power_of_two());
        (0..len as u32).map(M31::from_u32_unchecked).collect()
    }

    #[test]
    pub fn from_matrix_test() {
        const TRACE_LEN: usize = 16;
        let trace = init_m31_test_trace(TRACE_LEN);
        let matrix = vec![trace; 2];

        let tree = super::MerkleTree::<M31, Blake3Hasher>::init_from_column_array(matrix);

        assert_eq!(tree.bottom_layer.len(), 32);
        assert_eq!(tree.height, 3);
        (0..TRACE_LEN).for_each(|i| {
            assert_eq!(tree.bottom_layer[i * 2], M31::from_u32_unchecked(i as u32));
            assert_eq!(
                tree.bottom_layer[i * 2 + 1],
                M31::from_u32_unchecked(i as u32)
            );
        });
    }

    #[test]
    pub fn commit_test() {
        let trace = init_m31_test_trace(64);
        let matrix = vec![trace; 4];

        let tree_from_matrix = super::MerkleTree::<M31, Blake3Hasher>::commit(matrix);

        assert_eq!(
            hex::encode(tree_from_matrix.root()),
            "c07e98e8a5d745ea99c3c3eac4c43b9df5ceb9e78973a785d90b3ffe4d5fcf5e"
        );
    }

    #[test]
    pub fn get_leaf_block_test() {
        let trace = vec![init_m31_test_trace(128)];
        const BLOCK_LEN: usize = Blake3Hasher::BLOCK_SIZE_IN_BYTES / std::mem::size_of::<M31>();
        let tree_from_matrix = super::MerkleTree::<M31, Blake3Hasher>::commit(trace);
        let queries: BTreeSet<usize> = (0..100).map(|_| thread_rng().gen_range(0..128)).collect();

        for query in queries {
            let leaf_block_index = query / BLOCK_LEN;
            assert_eq!(
                tree_from_matrix.get_leaf_block(leaf_block_index),
                tree_from_matrix.bottom_layer
                    [leaf_block_index * BLOCK_LEN..(leaf_block_index + 1) * BLOCK_LEN]
            );
        }
    }

    #[test]
    pub fn test_decommitment() {
        let trace = vec![init_m31_test_trace(128)];

        let tree = super::MerkleTree::<M31, Blake3Hasher>::commit(trace);
        let queries: BTreeSet<usize> = (16..64).collect();
        let decommitment = tree.generate_decommitment(queries.clone());

        assert_eq!(decommitment.leaves.len(), 3);

        // Every leaf block in the first half of the trace is queried except for the first one,
        // therefore it should be the only one who's hash is in the decommitment's first
        // layer.
        assert_eq!(decommitment.layers[0].len(), 1);
        assert_eq!(
            decommitment.layers[0][0],
            Blake3Hasher::hash(to_byte_slice(&tree.get_leaf_block(0)))
        );

        // The queried leaves' parents can be computed by verifer therefore excluded from the proof.
        assert!(decommitment.layers[1].is_empty());

        // A verifer can compute the left child of the root from the previous layer, therefore
        // the proof only needs to contain the right child.
        assert_eq!(decommitment.layers[2].len(), 1);
    }
}
