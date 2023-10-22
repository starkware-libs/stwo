use super::{hasher::Hasher, NUM_BYTES_FELT};
use crate::math;
use log::info;
use std::{collections::BTreeMap, iter::repeat};

const MAX_LAYER_LENGTH_TO_HASH: u32 = 64; //2^6

/// The interface for interacting with the merkle tree. Namely: Commit & Decommit.
/// Splits a the work to sub-trees(see PartialTree) according to the number of cores available/defined in the system and input size.
// TODO(Ohad): Decommitment, Multi-Treading.
// TODO(Ohad): Remove #[allow(dead_code)].
#[allow(dead_code)]
pub struct MdcMerkleTree<T: Hasher> {
    pub data: Vec<Vec<T::Hash>>,
    pub height: usize,
    columns_size_map: BTreeMap<usize, Vec<usize>>,
    partial_trees: Vec<PartialTree<T>>,
}

/// These are meant to be 16KiB Trees.
// TODO(Ohad): Remove #[allow(dead_code)].
#[allow(dead_code)]
pub struct PartialTree<T: Hasher> {
    pub data: Vec<Vec<T::Hash>>,
    pub height: usize,
    columns_size_map: BTreeMap<usize, Vec<Vec<u32>>>,
    partial_trees: Vec<PartialTree<T>>,
}

/// TODO(Ohad): Remove #[allow(dead_code)].
#[allow(dead_code)]
impl<T: Hasher> PartialTree<T> {
    /// This operation runs on a single thread, and operates withi L1 cache (ideally).
    /// TODO(Ohad): Dynamically decide the amount of sub-trees (currently 0) and divide work accordingly.
    pub fn commit(cols: Vec<Vec<u32>>) -> Self {
        // Initiliaze column_length to corresponding column array.
        let (columns_size_map, remainder_columns) = self::init_col_length_map(cols);

        // Deal with first layer first.
        let bottom_layer = if remainder_columns.is_empty() {
            let bottom_layer_length = *columns_size_map.last_key_value().expect("Empty Map!").0;
            let longest_columns = columns_size_map
                .get(&bottom_layer_length)
                .expect("Empty map!");
            //error!("Started pre-hash first layer at:{:?}!", start.elapsed());
            let bottom_layer_pre_hash = prepare_pre_hash_no_inject(longest_columns);
            //error!("finished pre-hash first layer at:{:?}!", start.elapsed());
            let bottom_layer: Vec<T::Hash> = T::hash_many(&bottom_layer_pre_hash);
            bottom_layer
        } else {
            get_bottom_layer_from_subtree_roots::<T>(remainder_columns)
        };

        let height = math::log2_ceil(bottom_layer.len());
        let mut data: Vec<Vec<T::Hash>> = Vec::with_capacity(height);
        data.push(bottom_layer);

        // Collect data for the every layer then hash.
        // TODO(Ohad): reduce copying, optimize further after hash interface is changed (direct hash-injection).
        for _ in 0..height {
            // Prepare values form previous layer. Inject if necessary.
            let child_values = data.last().expect("msg");
            let current_layer_length = child_values.len() / 2;
            let columns_to_inject = columns_size_map.get(&current_layer_length);
            let next_hashed_layer: Vec<T::Hash> = match columns_to_inject {
                Some(cols) => {
                    let injected_pre_hash = inject::<T>(child_values, cols);
                    T::hash_many(&injected_pre_hash)
                }
                None => {
                    let child_values_byte_slice = unsafe {
                        std::slice::from_raw_parts(
                            child_values.as_ptr() as *const u8,
                            current_layer_length * 2 * T::OUTPUT_SIZE_IN_BYTES,
                        )
                    };
                    (0..current_layer_length)
                        .map(|i| {
                            T::hash(
                                &child_values_byte_slice[i * 2 * T::OUTPUT_SIZE_IN_BYTES
                                    ..i * 2 * T::OUTPUT_SIZE_IN_BYTES
                                        + 2 * T::OUTPUT_SIZE_IN_BYTES],
                            )
                        })
                        .collect()
                }
            };
            data.push(next_hashed_layer);
        }

        Self {
            data,
            height,
            columns_size_map,
            partial_trees: Vec::new(),
        }
    }

    pub fn root(&self) -> T::Hash {
        *self
            .data
            .last()
            .expect("Root extraction error!")
            .last()
            .expect("Root extraction error!")
    }
}

#[allow(clippy::type_complexity)]
pub fn init_col_length_map(cols: Vec<Vec<u32>>) -> (BTreeMap<usize, Vec<Vec<u32>>>, Vec<Vec<u32>>) {
    let mut columns_size_map: BTreeMap<usize, Vec<Vec<u32>>> = BTreeMap::new();
    let mut remainder_columns: Vec<Vec<u32>> = Vec::new();
    for c in cols {
        if c.len() <= (MAX_LAYER_LENGTH_TO_HASH as usize) {
            let length_index_entry = columns_size_map.entry(c.len()).or_default();
            info!("pushing column with length {}", c.len());
            length_index_entry.push(c);
        } else {
            remainder_columns.push(c);
        }
    }
    (columns_size_map, remainder_columns)
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

/// Take previous results and concatenate with corresponding "injected" columns.
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

// Recursive call to commit. Slices column array and commits on every slice.
// TODO(Ohad): Consider unsafe ptr conversions instead of iter slicing.
pub fn get_bottom_layer_from_subtree_roots<T: Hasher>(cols: Vec<Vec<u32>>) -> Vec<T::Hash> {
    assert!(cols.last().unwrap().len() >= MAX_LAYER_LENGTH_TO_HASH as usize);

    // Slice columns.
    let mut sliced_cols = Vec::<Vec<Vec<u32>>>::with_capacity(MAX_LAYER_LENGTH_TO_HASH as usize);
    sliced_cols.extend(repeat(Vec::<Vec<u32>>::new()).take(MAX_LAYER_LENGTH_TO_HASH as usize));
    cols.into_iter().for_each(|c| {
        c.chunks(c.len() / MAX_LAYER_LENGTH_TO_HASH as usize)
            .enumerate()
            .for_each(|(i, c)| sliced_cols[i].push(c.to_vec()))
    });

    sliced_cols
        .into_iter()
        .map(|cols| PartialTree::<T>::commit(cols).root())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        get_bottom_layer_from_subtree_roots, init_col_length_map, prepare_pre_hash_no_inject,
        PartialTree,
    };
    use crate::commitment_scheme::{blake3_hash::Blake3Hasher, hasher::Hasher, mdc_tree::inject};

    #[test]
    fn map_init_test() {
        let cols = init_test_trace();
        let mut map = init_col_length_map(cols).0;
        assert_eq!(map.get(&4).expect("no such key: 4").len(), 1);
        assert_eq!(map.get(&2).expect("no such key: 2").len(), 2);
        assert_eq!(map.pop_last().expect("Empty map").1[0][0], 1);
        assert_eq!(map.pop_last().expect("Empty map").1[1][0], 7);
        assert_eq!(map.pop_last().expect("Empty map").1[0][0], 9);
    }

    #[test]
    fn pre_hash_test() {
        let cols = init_test_trace();
        let mut map = init_col_length_map(cols).0;
        let layer_0_pre_hash = prepare_pre_hash_no_inject(&map.pop_last().expect("Empty map!").1);
        assert_eq!(
            format!("{:?}", layer_0_pre_hash),
            "[[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]"
        );
    }

    fn init_test_trace() -> Vec<Vec<u32>> {
        let col1 = vec![1, 2, 3, 4];
        let col2 = vec![5, 6];
        let col3 = vec![7, 8];
        let col4 = vec![9];
        let cols = vec![col1, col2, col3, col4];
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
    fn partial_tree_commit_test() {
        let cols = init_test_trace();
        let partial_tree = PartialTree::<Blake3Hasher>::commit(cols);
        assert_eq!(
            partial_tree.root().to_string(),
            "748cc35381cd5a7af1d95c563beeff0585454891e602d83845ea7569009acd76"
        );
    }

    #[test]
    #[ignore]
    fn large_partial_tree_commit_test() {
        let cols = init_long_single_column();
        let partial_tree = PartialTree::<Blake3Hasher>::commit(cols);
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
            bottom_layer[0].to_string(),
            "1da27fbdfb31a12605cb450b13aa42bb3b83b6ce9c6567934c948b5c29948cb4"
        )
    }

    #[test]
    fn inject_test() {
        let hash_a = Blake3Hasher::hash(b"a");
        let prev_layer = vec![hash_a, hash_a, hash_a, hash_a];
        let col1 = vec![1, 2];
        let col2 = vec![3, 4];
        let cols = vec![col1, col2];

        let injected_hash = inject::<Blake3Hasher>(&prev_layer, &cols);

        println!("{:?}", injected_hash)
    }
}
