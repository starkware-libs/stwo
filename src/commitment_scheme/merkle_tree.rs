use super::hasher::Hasher;
use crate::commitment_scheme::utils::*;

pub struct MerkleTree<T: Sized> {
    pub bottom_layer: Vec<T>,
    pub bottom_layer_node_size: usize,
    pub bottom_layer_n_rows_in_node: usize,
    pub data: TreeData,
    pub height: usize,
}

impl<T: Sized + Sync> MerkleTree<T> {
    /// Commits after data was loaded into bottom layer.
    pub fn commit<H: Hasher>(&mut self) {
        let bottom_layer_as_byte_slice = to_byte_slice(&self.bottom_layer);

        // Hash.
        hash_merkle_tree_from_bottom_layer::<H>(
            bottom_layer_as_byte_slice,
            self.bottom_layer_node_size * std::mem::size_of::<T>(),
            &mut tree_data_as_mut_ref(&mut self.data)[..],
        );
    }

    /// Builds the base layer of the tree from the given matrix.
    /// Allocates the rest of the tree.
    /// Does not commit(hash).
    // TODO(Ohad): add support for columns of different lengths.
    pub fn init_from_column_array<H: Hasher>(columns: &ColumnArray<T>) -> Self {
        // Check input is not empty, column length is a power two and all columns are of the same
        // length.
        assert!(!columns.is_empty());
        assert!(columns[0].len().is_power_of_two());
        columns.iter().for_each(|column| {
            assert_eq!(column.len(), columns[0].len());
        });

        // Allocate bottom layer.
        let mut bottom_layer: Vec<T> = Vec::with_capacity(columns[0].len() * columns.len());

        // Decide bottom_layer node size.
        let n_rows_in_node = std::cmp::min(
            crate::math::prev_pow_two(
                H::BLOCK_SIZE_IN_BYTES / (columns.len() * std::mem::size_of::<T>()),
            ),
            columns[0].len(),
        );
        let bottom_layer_node_size = n_rows_in_node * columns.len();

        // Inject(transpose).
        // Safe because enough memory is allocated.
        unsafe {
            bottom_layer.set_len(columns[0].len() * columns.len());
            let bottom_layer_byte_slice = to_byte_slice_mut(&mut bottom_layer);
            inject(columns, bottom_layer_byte_slice, 1, 0);
        }

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
            height: tree_data.len() + 1,
            data: tree_data,
        }
    }

    /// Builds the base layer of the tree from the given column.
    /// Allocates the rest of the tree.
    /// Does not commit(hash).
    /// Does not transpose, the column is already the base layer of the tree.
    pub fn init_from_single_column<H: Hasher>(column: Vec<T>) -> Self {
        assert!(column.len().is_power_of_two());

        // Decide bottom_layer node size.
        let bottom_layer_node_size = usize::min(
            H::BLOCK_SIZE_IN_BYTES / std::mem::size_of::<T>(),
            column.len(),
        );
        let bottom_layer_length_nodes = column.len() / bottom_layer_node_size;

        // Allocate rest of the tree.
        let tree_data = allocate_balanced_tree(
            bottom_layer_length_nodes,
            bottom_layer_node_size * std::mem::size_of::<T>(),
            H::OUTPUT_SIZE_IN_BYTES,
        );
        Self {
            bottom_layer: column,
            bottom_layer_node_size,
            bottom_layer_n_rows_in_node: bottom_layer_node_size,
            height: tree_data.len() + 1,
            data: tree_data,
        }
    }

    pub fn root(&self) -> Vec<u8> {
        self.data.last().unwrap().to_vec()
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

        let tree = super::MerkleTree::<M31>::init_from_column_array::<Blake3Hasher>(&matrix);

        assert_eq!(tree.bottom_layer.len(), 32);
        assert_eq!(tree.height, 3);
    }

    #[test]
    pub fn from_column_test() {
        let trace = init_m31_test_trace(16);

        let tree = super::MerkleTree::<M31>::init_from_single_column::<Blake3Hasher>(trace.clone());

        assert_eq!(tree.bottom_layer.len(), 16);
        assert_eq!(tree.bottom_layer, trace);
        assert_eq!(tree.height, 2);
    }

    #[test]
    pub fn commit_test() {
        let trace = init_m31_test_trace(64);
        let matrix = vec![trace.clone(), trace.clone(), trace.clone(), trace.clone()];

        let mut tree_from_matrix =
            super::MerkleTree::<M31>::init_from_column_array::<Blake3Hasher>(&matrix);
        let mut tree_from_column =
            super::MerkleTree::<M31>::init_from_single_column::<Blake3Hasher>(trace);
        tree_from_matrix.commit::<Blake3Hasher>();
        tree_from_column.commit::<Blake3Hasher>();

        assert_eq!(
            hex::encode(tree_from_matrix.root()),
            "c07e98e8a5d745ea99c3c3eac4c43b9df5ceb9e78973a785d90b3ffe4d5fcf5e"
        );
        assert_eq!(
            hex::encode(tree_from_column.root()),
            "945efd1734d66bc05878c801750cb76136a06da978d7d3353eafde8580e02aec"
        );
    }
}
