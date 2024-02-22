use super::utils::get_column_chunk;
use crate::core::fields::Field;

/// The Input of a Merkle-Tree Mixed-Degree commitment scheme.
/// A map from the depth of the tree requested to be injected to the to-be-injected columns.
/// A layer of depth 'd' in a merkle tree, holds 2^(d-1) hash buckets, each containing 2 sibling
/// hashes and the injected values of that depth.
///
/// # Example
///
/// ```rust
/// use stwo::commitment_scheme::merkle_input::MerkleTreeInput;
/// use stwo::core::fields::m31::M31;
///
/// let mut input = MerkleTreeInput::<M31>::new();
/// let column = vec![M31::from_u32_unchecked(0); 1024];
/// input.insert_column(2, &column);
/// input.insert_column(3, &column);
/// input.insert_column(3, &column);
///
/// assert_eq!(input.get_columns(2).len(), 1);
/// assert_eq!(input.get_columns(3).len(), 2);
/// assert_eq!(input.max_injected_depth(), 3);
/// ````
// `columns_to_inject` - A vector of columns to be injected to the merkle tree, ordered as
//  inserted.
// `injected_depths_map` - A mapping from a depth of the tree to the columns injected at that
//  depth, ordered as inserted.
#[derive(Default)]
pub struct MerkleTreeInput<'a, F: Field> {
    columns_to_inject: Vec<&'a [F]>,
    injected_depths_map: Vec<Vec<usize>>,
}

pub type LayerColumns<'a, F> = Vec<&'a [F]>;

impl<'a, F: Field> MerkleTreeInput<'a, F> {
    pub fn new() -> Self {
        Self {
            columns_to_inject: vec![],
            injected_depths_map: vec![],
        }
    }

    pub fn insert_column(&mut self, depth: usize, column: &'a [F]) {
        assert_ne!(depth, 0, "Injection to layer 0 undefined!");
        assert!(
            column.len().is_power_of_two(),
            "Column is of size: {}, not a power of 2!",
            column.len()
        );

        // Column is spread over 'hash buckets' in the layer, every layer holds 2^(depth-1) buckets.
        // TODO(Ohad): implement embedd by repeatition and remove assert.
        assert!(
            column.len() >= 2usize.pow((depth - 1) as u32),
            "Column of size: {} is too small for injection at layer:{}",
            column.len(),
            depth
        );

        if self.injected_depths_map.len() < depth {
            self.injected_depths_map.resize(depth, vec![]);
        }
        self.injected_depths_map[depth - 1].push(self.columns_to_inject.len());
        self.columns_to_inject.push(column);
    }

    pub fn get_columns(&'a self, depth: usize) -> Vec<&[F]> {
        match self.injected_depths_map.get(depth - 1) {
            Some(injected_column_indices) => injected_column_indices
                .iter()
                .map(|&index| self.columns_to_inject[index])
                .collect::<Vec<&[F]>>(),
            _ => panic!(
                "Attempted extraction of columns from depth: {}, but max injected depth is: {}",
                depth,
                self.max_injected_depth()
            ),
        }
    }

    pub fn max_injected_depth(&self) -> usize {
        self.injected_depths_map.len()
    }

    pub fn get_injected_elements(&self, depth: usize, bag_index: usize) -> Vec<F> {
        let n_bags_in_layer = 1 << (depth - 1);
        let mut injected_elements = Vec::<F>::new();
        for column in self.get_columns(depth).iter() {
            let col_chunk = get_column_chunk(column, bag_index, n_bags_in_layer);
            injected_elements.extend(col_chunk);
        }
        injected_elements
    }

    pub fn n_injected_columns(&self) -> usize {
        self.columns_to_inject.len()
    }

    // Returns the structure of the merkle tree. i.e. for each depth, the length of the columns
    // assigned to it.
    // TODO(Ohad): implement this logic for the verifier.
    pub fn configuration(&self) -> MerkleTreeConfig {
        let column_sizes = self
            .columns_to_inject
            .iter()
            .map(|col| col.len())
            .collect::<Vec<usize>>();
        MerkleTreeConfig {
            column_sizes,
            injected_depths_map: self.injected_depths_map.clone(),
        }
    }
}

/// The structure of a mixed degree merkle tree.
/// The sizes of columns assigned to every layer, ordered as they were inserted & injected into hash
/// blocks.
pub struct MerkleTreeConfig {
    column_sizes: Vec<usize>,
    injected_depths_map: Vec<Vec<usize>>,
}

impl MerkleTreeConfig {
    pub fn sort_queries_by_layer(&self, queries: &[Vec<usize>]) -> Vec<Vec<Vec<usize>>> {
        let mut queries_to_layers = vec![vec![]; self.height()];
        (1..=self.height()).for_each(|i| {
            let columns_in_layer = self.column_indices_at(i);
            columns_in_layer.iter().for_each(|&column_index| {
                queries_to_layers[i - 1].push(queries[column_index].clone());
            });
        });
        queries_to_layers
    }

    pub fn column_lengths_at_depth(&self, depth: usize) -> Vec<usize> {
        self.column_indices_at(depth)
            .iter()
            .map(|&index| self.column_sizes[index])
            .collect::<Vec<usize>>()
    }

    pub fn height(&self) -> usize {
        self.injected_depths_map.len()
    }

    fn column_indices_at(&self, depth: usize) -> &[usize] {
        &self.injected_depths_map[depth - 1]
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::core::fields::m31::M31;
    use crate::m31;

    #[test]
    pub fn md_input_insert_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];

        input.insert_column(3, &column);
        input.insert_column(3, &column);
        input.insert_column(2, &column);

        assert_eq!(input.get_columns(3).len(), 2);
        assert_eq!(input.get_columns(2).len(), 1);
    }

    #[test]
    pub fn md_input_max_depth_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];

        input.insert_column(3, &column);
        input.insert_column(2, &column);

        assert_eq!(input.max_injected_depth(), 3);
    }

    #[test]
    #[should_panic]
    pub fn get_invalid_depth_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];
        input.insert_column(3, &column);

        input.get_columns(4);
    }

    #[test]
    pub fn merkle_tree_input_empty_vec_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];
        input.insert_column(3, &column);

        assert_eq!(input.get_columns(2), Vec::<Vec<M31>>::new().as_slice());
    }
    #[test]
    #[should_panic]
    pub fn mt_input_column_too_short_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];

        input.insert_column(12, &column);
    }

    #[test]
    #[should_panic]
    pub fn mt_input_wrong_size_test() {
        let mut input = super::MerkleTreeInput::<M31>::default();
        let not_pow_2_column = vec![M31::from_u32_unchecked(0); 1023];

        input.insert_column(2, &not_pow_2_column);
    }

    #[test]
    fn get_injected_elements_test() {
        let trace_column = (0..4).map(M31::from_u32_unchecked).collect::<Vec<_>>();
        let mut merkle_input = super::MerkleTreeInput::<M31>::new();
        merkle_input.insert_column(3, &trace_column);
        merkle_input.insert_column(2, &trace_column);

        let injected_elements_30 = merkle_input.get_injected_elements(3, 0);
        let injected_elements_31 = merkle_input.get_injected_elements(3, 1);
        let injected_elements_32 = merkle_input.get_injected_elements(3, 2);
        let injected_elements_33 = merkle_input.get_injected_elements(3, 3);
        let injected_elements_20 = merkle_input.get_injected_elements(2, 0);
        let injected_elements_21 = merkle_input.get_injected_elements(2, 1);

        assert_eq!(injected_elements_30, vec![m31!(0)]);
        assert_eq!(injected_elements_31, vec![m31!(1)]);
        assert_eq!(injected_elements_32, vec![m31!(2)]);
        assert_eq!(injected_elements_33, vec![m31!(3)]);
        assert_eq!(injected_elements_20, vec![m31!(0), m31!(1)]);
        assert_eq!(injected_elements_21, vec![m31!(2), m31!(3)]);
    }

    #[test]
    fn n_injected_columns_test() {
        let mut merkle_input = super::MerkleTreeInput::<M31>::new();
        let trace_column = (0..4).map(M31::from_u32_unchecked).collect::<Vec<_>>();
        merkle_input.insert_column(3, &trace_column);
        merkle_input.insert_column(2, &trace_column);
        merkle_input.insert_column(2, &trace_column);

        assert_eq!(merkle_input.n_injected_columns(), 3);
    }

    #[test]
    fn config_length_at_depth_test() {
        let mut merkle_input = super::MerkleTreeInput::<M31>::new();
        let column_length_4 = (0..4).map(M31::from_u32_unchecked).collect::<Vec<_>>();
        let column_length_8 = (0..8).map(M31::from_u32_unchecked).collect::<Vec<_>>();
        let column_length_16 = (0..16).map(M31::from_u32_unchecked).collect::<Vec<_>>();
        merkle_input.insert_column(3, &column_length_4);
        merkle_input.insert_column(2, &column_length_8);
        merkle_input.insert_column(2, &column_length_4);
        merkle_input.insert_column(3, &column_length_16);

        let merkle_config = merkle_input.configuration();

        assert_eq!(merkle_config.column_lengths_at_depth(3), vec![4, 16]);
        assert_eq!(merkle_config.column_lengths_at_depth(2), vec![8, 4]);
    }

    #[test]
    fn sort_queries_by_layer_test() {
        let mut merkle_input = super::MerkleTreeInput::<M31>::new();
        let column = [M31::from_u32_unchecked(0); 64];
        merkle_input.insert_column(3, &column);
        merkle_input.insert_column(2, &column);
        merkle_input.insert_column(4, &column);
        merkle_input.insert_column(2, &column);
        merkle_input.insert_column(4, &column);
        merkle_input.insert_column(3, &column);

        let queries = vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]];

        let merkle_config = merkle_input.configuration();
        let sorted_queries = merkle_config.sort_queries_by_layer(&queries);

        assert_eq!(
            sorted_queries,
            vec![
                vec![],
                vec![vec![1], vec![3]],
                vec![vec![0], vec![5]],
                vec![vec![2], vec![4]]
            ]
        );
    }
}
