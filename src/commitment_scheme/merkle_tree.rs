use super::hasher::Hasher;
use crate::commitment_scheme::utils::{
    allocate_balanced_tree, column_to_row_major, hash_merkle_tree_from_bottom_layer, to_byte_slice,
    tree_data_as_mut_ref, ColumnArray, TreeData,
};

pub struct MerkleTree<T: Sized, H: Hasher> {
    pub bottom_layer: Vec<T>,
    pub bottom_layer_node_size: usize,
    pub bottom_layer_n_rows_in_node: usize,
    pub data: TreeData,
    pub height: usize,
    phantom: std::marker::PhantomData<H>,
}

impl<T: Sized, H: Hasher> MerkleTree<T, H> {
    /// Commits on a given trace(matrix).
    pub fn commit(trace: ColumnArray<T>) -> Self {
        let mut tree = Self::init_from_column_array(trace);

        let bottom_layer_as_byte_slice = to_byte_slice(&tree.bottom_layer);
        hash_merkle_tree_from_bottom_layer::<H>(
            bottom_layer_as_byte_slice,
            tree.bottom_layer_node_size * std::mem::size_of::<T>(),
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

        let bottom_layer_node_size = n_rows_in_node * trace.len();
        let bottom_layer = column_to_row_major(trace);

        // Allocate rest of the tree.
        let bottom_layer_length_nodes =
            crate::math::usize_div_ceil(bottom_layer.len(), bottom_layer_node_size);
        let tree_data = allocate_balanced_tree(
            bottom_layer_length_nodes,
            H::BLOCK_SIZE_IN_BYTES,
            H::OUTPUT_SIZE_IN_BYTES,
        );

        // Allocate rest of the tree.
        let bottom_layer_length_nodes =
            crate::math::usize_div_ceil(bottom_layer.len(), bottom_layer_node_size);
        let tree_data = allocate_balanced_tree(
            bottom_layer_length_nodes,
            H::BLOCK_SIZE_IN_BYTES,
            H::OUTPUT_SIZE_IN_BYTES,
        );

        Self {
            bottom_layer,
            bottom_layer_node_size,
            bottom_layer_n_rows_in_node: n_rows_in_node,
            height: tree_data.len() + 1, // +1 for the bottom layer.
            data: tree_data,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn root(&self) -> H::Hash {
        (&self.data.last().unwrap()[..]).into()
    }
}
#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::*;
    use crate::core::fields::m31::M31;

    fn init_m31_test_trace(len: usize) -> Vec<M31> {
        assert!(len.is_power_of_two());
        (0..len as u32).map(M31::from_u32_unchecked).collect()
    }

    #[test]
    pub fn from_matrix_test() {
        let trace = init_m31_test_trace(16);
        let matrix = vec![trace.clone(), trace.clone()];

        let tree = super::MerkleTree::<M31, Blake3Hasher>::init_from_column_array(matrix);

        assert_eq!(tree.bottom_layer.len(), 32);
        assert_eq!(tree.height, 3);
    }

    #[test]
    pub fn commit_test() {
        let trace = init_m31_test_trace(64);
        let matrix = vec![trace.clone(), trace.clone(), trace.clone(), trace.clone()];

        let tree_from_matrix = super::MerkleTree::<M31, Blake3Hasher>::commit(matrix);

        assert_eq!(
            hex::encode(tree_from_matrix.root()),
            "c07e98e8a5d745ea99c3c3eac4c43b9df5ceb9e78973a785d90b3ffe4d5fcf5e"
        );
    }
}
