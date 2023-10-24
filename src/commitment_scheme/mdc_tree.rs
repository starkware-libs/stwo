use super::hasher::Hasher;
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

#[cfg(test)]
mod tests {
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
        let (mut col_length_map, remainder_columns) = sort_columns_and_extract_remainder(cols);

        // Test map
        assert_eq!(col_length_map.get(&4).expect("no such key: 4").len(), 1);
        assert_eq!(col_length_map.get(&2).expect("no such key: 2").len(), 2);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 1);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[1][0], 7);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 9);

        //Test remainder
        assert_eq!(
            remainder_columns,
            vec![std::iter::repeat(0)
                .take(2 * MAX_LAYER_LENGTH_TO_HASH)
                .collect::<Vec<u32>>()]
        );
    }
}
