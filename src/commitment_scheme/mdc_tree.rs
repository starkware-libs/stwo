use crate::math;

use super::{hasher::Hasher, NUM_BYTES_FELT};
use std::{collections::BTreeMap, iter::repeat, marker::PhantomData};

type ColumnArray = Vec<Vec<u32>>;
type ColumnLengthMap = BTreeMap<usize, Vec<Vec<u32>>>;

//TODO(Ohad): TODO(Ohad): Remove #[allow(dead_code)].
#[allow(dead_code)]
const MAX_SUBTREE_BOTTOM_LAYER_LENGTH: usize = 64;

/// Merkle Tree interface. Namely: Commit & Decommit.
// TODO(Ohad): Decommitment, Multi-Treading.
// TODO(Ohad): Remove #[allow(dead_code)].
#[allow(dead_code)]
pub struct MixedDegreeTree<T: Hasher> {
    pub data: Vec<Vec<Vec<u8>>>,
    pub height: usize,
    columns_size_map: ColumnLengthMap,
    sub_trees: Option<Vec<MixedDegreeTree<T>>>,
    _marker: PhantomData<T>,
}

/// TODO(Ohad): Remove #[allow(dead_code)].
#[allow(dead_code)]
impl<T: Hasher> MixedDegreeTree<T> {
    /// This operation runs on a single thread, and operates within L1 cache (ideally).
    /// TODO(Ohad): Dynamically decide the amount of sub-trees and divide work accordingly.
    pub fn commit(cols: Vec<Vec<u32>>) -> Self {
        // Initiliaze map from column_length to corresponding column array.
        let (columns_size_map, remainder_columns) =
            self::sort_columns_and_extract_remainder(cols, MAX_SUBTREE_BOTTOM_LAYER_LENGTH);
        let mut partial_trees: Option<Vec<MixedDegreeTree<T>>> = None;
        let mut data: Vec<Vec<Vec<u8>>> = Vec::new();

        // Deal with first layer first.
        let first_layer_hash = if remainder_columns.is_empty() {
            // We are at the bottom layer of the tree, no need to inject.
            let bottom_layer_length = *columns_size_map.last_key_value().expect("Empty Map!").0;
            let longest_columns = columns_size_map
                .get(&bottom_layer_length)
                .expect("Empty map!");
            let bottom_layer_pre_hash = prepare_pre_hash_no_inject(longest_columns);
            let first_layer_hash: Vec<T::Hash> = T::hash_many(&bottom_layer_pre_hash);
            data.push(bottom_layer_pre_hash);
            first_layer_hash
        } else {
            // There are sub-trees. collect them.
            let partial_trees_vec = get_bottom_layer_from_subtree_roots::<T>(remainder_columns);
            let roots = partial_trees_vec.iter().map(|tree| tree.root()).collect();
            partial_trees = Some(partial_trees_vec);
            roots
        };
        let height = math::log2_ceil(first_layer_hash.len());

        // Collect data for the every layer then hash.
        // TODO(Ohad): reduce copying, optimize further after hash interface is changed (direct hash-injection).
        let mut child_values = first_layer_hash;
        for _ in 0..height {
            // Prepare values form previous layer. Inject if necessary.
            let current_layer_length = child_values.len() / 2;
            let columns_to_inject = columns_size_map.get(&current_layer_length);
            child_values = match columns_to_inject {
                Some(cols) => {
                    let injected_pre_hash = inject::<T>(&child_values, cols);
                    let injected_hash_results = T::hash_many(&injected_pre_hash);
                    data.push(injected_pre_hash);
                    injected_hash_results
                }
                None => {
                    // Hash straight from pointed at address to avoid copying.
                    //TODO(Ohad): working with slices is still more expensive than raw pointers, as building the slice pointer is redundent. optimize this.
                    let child_values_byte_slice = unsafe {
                        std::slice::from_raw_parts(
                            child_values.as_ptr() as *const u8,
                            current_layer_length * 2 * T::OUTPUT_SIZE_IN_BYTES,
                        )
                    };

                    // Hash
                    let hash_results = (0..current_layer_length)
                        .map(|i| {
                            T::hash(
                                &child_values_byte_slice[i * 2 * T::OUTPUT_SIZE_IN_BYTES
                                    ..i * 2 * T::OUTPUT_SIZE_IN_BYTES
                                        + 2 * T::OUTPUT_SIZE_IN_BYTES],
                            )
                        })
                        .collect();
                    data.push(child_values.into_iter().map(|h| h.into()).collect());
                    hash_results
                }
            };
        }
        assert_eq!(child_values.len(), 1);
        data.push(vec![child_values[0].into()]);

        Self {
            data,
            height,
            columns_size_map,
            sub_trees: partial_trees,
            _marker: PhantomData::<T>,
        }
    }

    pub fn root(&self) -> T::Hash {
        let hash_bytes = self
            .data
            .last()
            .expect("Root extraction error!")
            .last()
            .expect("Root extraction error!");
        T::Hash::from(hash_bytes.clone())
    }
}

// Recursive call to commit. Slices column array and commits on every slice.
// TODO(Ohad): Consider unsafe ptr conversions instead of iter slicing.
pub fn get_bottom_layer_from_subtree_roots<T: Hasher>(
    cols: Vec<Vec<u32>>,
) -> Vec<MixedDegreeTree<T>> {
    assert!(cols.last().unwrap().len() >= MAX_SUBTREE_BOTTOM_LAYER_LENGTH);

    // Slice columns.
    let mut sliced_cols = Vec::<Vec<Vec<u32>>>::with_capacity(MAX_SUBTREE_BOTTOM_LAYER_LENGTH);
    sliced_cols.extend(repeat(Vec::<Vec<u32>>::new()).take(MAX_SUBTREE_BOTTOM_LAYER_LENGTH));
    cols.into_iter().for_each(|c| {
        c.chunks(c.len() / MAX_SUBTREE_BOTTOM_LAYER_LENGTH)
            .enumerate()
            .for_each(|(i, c)| sliced_cols[i].push(c.to_vec()))
    });

    // Recursive commit on smalled trees
    sliced_cols
        .into_iter()
        .map(|cols| MixedDegreeTree::<T>::commit(cols))
        .collect()
}

/// Takes columns that should be commited on, sorts and maps by length.
/// Extracts columns that are too long for handling by subtrees.
pub fn sort_columns_and_extract_remainder(
    cols: ColumnArray,
    max_layer_length_to_keep: usize,
) -> (ColumnLengthMap, ColumnArray) {
    let mut columns_length_map: ColumnLengthMap = BTreeMap::new();
    let mut remainder_columns: ColumnArray = Vec::new();
    for c in cols {
        if c.len() <= (max_layer_length_to_keep) {
            let length_index_entry = columns_length_map.entry(c.len()).or_default();
            length_index_entry.push(c);
        } else {
            remainder_columns.push(c);
        }
    }
    (columns_length_map, remainder_columns)
}

/// Given columns of the same length, transforms to bytes and concatenates corresponding column elements.
/// Assumes columns are of the same length.
pub fn prepare_pre_hash_no_inject(eq_length_cols: &ColumnArray) -> Vec<Vec<u8>> {
    let column_length = eq_length_cols[0].len();
    let n_collumns = eq_length_cols.len();
    let mut pre_hash: Vec<Vec<u8>> = Vec::with_capacity(column_length);
    for i in 0..column_length {
        let mut row: Vec<u8> = Vec::with_capacity(n_collumns * NUM_BYTES_FELT);
        let mut row_ptr = row.as_mut_ptr();
        unsafe {
            for c in eq_length_cols {
                std::ptr::copy_nonoverlapping(
                    c.as_ptr().add(i) as *mut u8,
                    row_ptr,
                    NUM_BYTES_FELT,
                );
                row_ptr = row_ptr.add(NUM_BYTES_FELT);
            }
            row.set_len(n_collumns * NUM_BYTES_FELT);
        }
        pre_hash.push(row);
    }
    pre_hash
}

/// Take previous results and concatenate with corresponding "for injection" columns.
// There is a reallocation as the previous layer is a continous array.
// TODO(Ohad): Re-write the hashing module so that the above stated realloc-copy is redundent and can be optimized.
pub fn inject<T: Hasher>(
    prev_layer_hash: &Vec<T::Hash>,
    cols_to_inject: &[Vec<u32>],
) -> Vec<Vec<u8>> {
    // Assumes columns are of the same length.
    // TODO(Ohad): Consider asserting the above statement.
    let column_length = cols_to_inject[0].len();
    let new_layer_len = prev_layer_hash.len() / 2;
    assert_eq!(new_layer_len, column_length);

    let mut prev_layer_hash_injected: Vec<Vec<u8>> = Vec::with_capacity(new_layer_len);
    let n_collumns = cols_to_inject.len();

    // Allocate more space, copy values over, inject.
    let prev_layer_as_byte_ptr = prev_layer_hash.as_ptr() as *const u8;
    for i in 0..new_layer_len {
        let mut vec_to_fill: Vec<u8> =
            Vec::with_capacity(T::OUTPUT_SIZE_IN_BYTES * 2 + n_collumns * NUM_BYTES_FELT);
        let mut vec_to_fill_ptr = vec_to_fill.as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(
                prev_layer_as_byte_ptr.add(i * T::OUTPUT_SIZE_IN_BYTES * 2),
                vec_to_fill_ptr,
                T::OUTPUT_SIZE_IN_BYTES * 2,
            );
            vec_to_fill_ptr = vec_to_fill_ptr.add(T::OUTPUT_SIZE_IN_BYTES * 2);
            for c in cols_to_inject {
                std::ptr::copy_nonoverlapping(c.as_ptr().add(i) as *mut u8, vec_to_fill_ptr, 4);
                vec_to_fill_ptr = vec_to_fill_ptr.add(4);
            }
            vec_to_fill.set_len(T::OUTPUT_SIZE_IN_BYTES * 2 + n_collumns * NUM_BYTES_FELT);
        }
        prev_layer_hash_injected.push(vec_to_fill);
    }

    prev_layer_hash_injected
}

#[cfg(test)]
mod tests {
    use super::{sort_columns_and_extract_remainder, ColumnArray, MAX_SUBTREE_BOTTOM_LAYER_LENGTH};
    use crate::commitment_scheme::{
        blake3_hash::Blake3Hasher,
        hasher::Hasher,
        mdc_tree::{
            get_bottom_layer_from_subtree_roots, inject, prepare_pre_hash_no_inject,
            MixedDegreeTree,
        },
    };

    fn init_test_trace() -> Vec<Vec<u32>> {
        let col0 = std::iter::repeat(0)
            .take(2 * MAX_SUBTREE_BOTTOM_LAYER_LENGTH)
            .collect();
        let col1 = vec![1, 2, 3, 4];
        let col2 = vec![5, 6];
        let col3 = vec![7, 8];
        let col4 = vec![9];
        let cols: ColumnArray = vec![col0, col1, col2, col3, col4];
        cols
    }

    fn init_test_trace2() -> Vec<Vec<u32>> {
        let col1 = vec![1, 2, 3, 4];
        let col2 = vec![5, 6];
        let col3 = vec![7, 8];
        let col4 = vec![9];
        let cols: ColumnArray = vec![col1, col2, col3, col4];
        cols
    }

    fn init_short_single_column() -> Vec<Vec<u32>> {
        let col: Vec<u32> = std::iter::repeat(0).take(256).collect();
        let cols = vec![col];
        cols
    }

    fn init_long_single_column() -> Vec<Vec<u32>> {
        let col: Vec<u32> = std::iter::repeat(0).take(16777216).collect();
        let cols = vec![col];
        cols
    }

    #[test]
    fn sort_columns_and_extract_remainder_test() {
        let cols = init_test_trace();
        let (mut col_length_map, remainder_columns) =
            sort_columns_and_extract_remainder(cols.clone(), MAX_SUBTREE_BOTTOM_LAYER_LENGTH);

        // Test map.
        assert_eq!(col_length_map.get(&4).expect("no such key: 4").len(), 1);
        assert_eq!(col_length_map.get(&2).expect("no such key: 2").len(), 2);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 1);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[1][0], 7);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 9);

        // Test remainder.
        assert_eq!(remainder_columns, vec![cols[0].clone()]);
    }

    #[test]
    fn pre_hash_no_inject_test() {
        let cols = init_test_trace();
        let mut map = sort_columns_and_extract_remainder(cols, MAX_SUBTREE_BOTTOM_LAYER_LENGTH).0;
        let layer_0_pre_hash = prepare_pre_hash_no_inject(&map.pop_last().expect("Empty map!").1);
        assert_eq!(
            format!("{:?}", layer_0_pre_hash),
            "[[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]"
        );
    }

    #[test]
    fn inject_test() {
        let hash_a = Blake3Hasher::hash(b"a");
        let prev_layer = vec![hash_a, hash_a, hash_a, hash_a];
        let col1 = vec![1, 2];
        let col2 = vec![3, 4];
        let cols = vec![col1, col2];

        let injected_hash = inject::<Blake3Hasher>(&prev_layer, &cols);

        assert_eq!(hex::encode(&injected_hash[0]),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f0100000003000000");
        assert_eq!(hex::encode(&injected_hash[1]),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f0200000004000000");
    }

    #[test]
    fn partial_tree_commit_test() {
        let cols = init_test_trace2();
        let partial_tree: MixedDegreeTree<Blake3Hasher> =
            MixedDegreeTree::<Blake3Hasher>::commit(cols);
        assert_eq!(
            partial_tree.root().to_string(),
            "748cc35381cd5a7af1d95c563beeff0585454891e602d83845ea7569009acd76"
        );
    }

    #[test]
    #[ignore]
    fn large_partial_tree_commit_test() {
        let cols = init_long_single_column();
        let partial_tree = MixedDegreeTree::<Blake3Hasher>::commit(cols);
        assert_eq!(
            partial_tree.root().to_string(),
            "e2bc3329cfb7d030e598fc9c1320997762b2f8eb2328599b061d3898ffccb60b"
        );
    }

    #[test]
    fn get_bottom_layer_test() {
        let cols = init_short_single_column();
        let bottom_layer = get_bottom_layer_from_subtree_roots::<Blake3Hasher>(cols);
        assert_eq!(
            bottom_layer[0].root().to_string(),
            "1da27fbdfb31a12605cb450b13aa42bb3b83b6ce9c6567934c948b5c29948cb4"
        )
    }
}
