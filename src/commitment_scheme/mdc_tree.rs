use super::{hasher::Hasher, NUM_BYTES_FELT};
use std::collections::BTreeMap;

type ColumnArray = Vec<Vec<u32>>;
type ColumnLengthMap = BTreeMap<usize, Vec<Vec<u32>>>;

const MAX_LAYER_LENGTH_TO_HASH: usize = 64;

/// Merkle Tree interface. Namely: Commit & Decommit.
/// Splits a the work to sub-trees according to the number of cores available/defined in the system and input size.
// TODO(Ohad): Decommitment, Multi-Treading.
// TODO(Ohad): Remove #[allow(dead_code)].
#[allow(dead_code)]
pub struct MixedDegreeTree<T: Hasher> {
    pub data: Vec<Vec<T::Hash>>,
    pub height: usize,
    columns_size_map: ColumnLengthMap,
    partial_trees: Vec<MixedDegreeTree<T>>,
}

/// Takes columns that should be commited on, sorts and maps by length.
/// Extracts columns that are too long for handling by subtrees.
pub fn sort_columns_and_extract_remainder(cols: ColumnArray) -> (ColumnLengthMap, ColumnArray) {
    let mut columns_length_map: ColumnLengthMap = BTreeMap::new();
    let mut remainder_columns: ColumnArray = Vec::new();
    for c in cols {
        if c.len() <= (MAX_LAYER_LENGTH_TO_HASH) {
            let length_index_entry = columns_length_map.entry(c.len()).or_default();
            length_index_entry.push(c);
        } else {
            remainder_columns.push(c);
        }
    }
    (columns_length_map, remainder_columns)
}

/// Takes field element columns, transforms to bytes and concatenates corresponding column elements.
pub fn prepare_pre_hash_no_inject(eq_length_cols: &[Vec<u32>]) -> Vec<Vec<u8>> {
    let column_length = eq_length_cols[0].len();
    let num_collumns = eq_length_cols.len();
    let mut pre_hash: Vec<Vec<u8>> = Vec::with_capacity(column_length);
    for i in 0..column_length {
        let mut row: Vec<u8> = Vec::with_capacity(num_collumns * NUM_BYTES_FELT);
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
            row.set_len(num_collumns * NUM_BYTES_FELT);
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
    // Columns should be of same lengeth.
    // TODO(Ohad): Consider asserting the above statement.
    let column_length = cols_to_inject[0].len();
    let new_layer_len = prev_layer_hash.len() / 2;
    assert_eq!(new_layer_len, column_length);

    let mut prev_layer_hash_injected: Vec<Vec<u8>> = Vec::with_capacity(new_layer_len);
    let num_collumns = cols_to_inject.len();

    // Allocate more space, copy values over, inject.
    let prev_layer_as_byte_ptr = prev_layer_hash.as_ptr() as *const u8;
    for i in 0..new_layer_len {
        let mut vec_to_fill: Vec<u8> =
            Vec::with_capacity(T::OUTPUT_SIZE_IN_BYTES * 2 + num_collumns * NUM_BYTES_FELT);
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
            vec_to_fill.set_len(T::OUTPUT_SIZE_IN_BYTES * 2 + num_collumns * NUM_BYTES_FELT);
        }
        prev_layer_hash_injected.push(vec_to_fill);
    }

    prev_layer_hash_injected
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::{
        blake3_hash::Blake3Hasher,
        hasher::Hasher,
        mdc_tree::{inject, prepare_pre_hash_no_inject},
    };

    use super::{sort_columns_and_extract_remainder, ColumnArray, MAX_LAYER_LENGTH_TO_HASH};

    fn init_test_trace() -> Vec<Vec<u32>> {
        let col0 = std::iter::repeat(0)
            .take(2 * MAX_LAYER_LENGTH_TO_HASH)
            .collect();
        let col1 = vec![1, 2, 3, 4];
        let col2 = vec![5, 6];
        let col3 = vec![7, 8];
        let col4 = vec![9];
        let cols: ColumnArray = vec![col0, col1, col2, col3, col4];
        cols
    }

    #[test]
    fn sort_columns_and_extract_remainder_test() {
        let cols = init_test_trace();
        let (mut col_length_map, remainder_columns) =
            sort_columns_and_extract_remainder(cols.clone());

        // Test map
        assert_eq!(col_length_map.get(&4).expect("no such key: 4").len(), 1);
        assert_eq!(col_length_map.get(&2).expect("no such key: 2").len(), 2);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 1);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[1][0], 7);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 9);

        //Test remainder
        assert_eq!(remainder_columns, vec![cols[0].clone()]);
    }

    #[test]
    fn pre_hash_no_inject_test() {
        let cols = init_test_trace();
        let mut map = sort_columns_and_extract_remainder(cols).0;
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
}
