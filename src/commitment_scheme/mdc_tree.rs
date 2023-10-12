use super::hasher::Hasher;
use crate::math;
use log::info;
use std::collections::BTreeMap;

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
struct PartialTree<T: Hasher> {
    pub data: Vec<Vec<T::Hash>>,
    pub height: usize,
    columns_size_map: BTreeMap<usize, Vec<Vec<u32>>>,
    partial_trees: Vec<PartialTree<T>>,
}

/// TODO(Ohad): Remove #[allow(dead_code)].
#[allow(dead_code)]
impl<T: Hasher> PartialTree<T> {
    /// This operation runs on a single thread, and operates within
    /// L1 cache (ideally).
    /// TODO(Ohad): Dynamically decide the amount of sub-trees (currently 0) and divide work accordingly.
    pub fn commit(cols: Vec<Vec<u32>>) -> Self {
        // Initiliaze column_length to corresponding column array.
        let columns_size_map = self::init_col_length_map(cols);

        // Get bottom layer size and info on tree size.
        let bottom_layer_length = *columns_size_map.last_key_value().expect("Empty Map!").0;
        let height = math::log2_ceil(bottom_layer_length);
        let mut data: Vec<Vec<T::Hash>> = Vec::with_capacity(height);

        // Deal with first layer first.
        let longest_columns = columns_size_map
            .get(&bottom_layer_length)
            .expect("Empty map!");
        let bottom_layer_pre_hash = prepare_pre_hash(longest_columns);
        let bottom_layer: Vec<T::Hash> = bottom_layer_pre_hash.iter().map(|v| T::hash(v)).collect();
        data.push(bottom_layer);

        // Collect data for the every layer then hash.
        for _i in 0..height {
            // Prepare values form previous layer. Inject if necessary.
            let child_values = data.last().expect("msg");
            let current_layer_length = child_values.len() / 2;
            let columns_to_inject = columns_size_map.get(&current_layer_length);
            let injected_pre_hash = match columns_to_inject {
                Some(cols) => {
                    let pre_hash_to_inject = prepare_pre_hash(cols);
                    child_values
                        .chunks(2)
                        .zip(pre_hash_to_inject)
                        .map(|x| [x.0[0].into(), x.0[1].into(), x.1].concat())
                        .collect::<Vec<Vec<u8>>>()
                }
                None => child_values
                    .chunks(2)
                    .map(|x| [x[0].into(), x[1].into()].concat())
                    .collect::<Vec<Vec<u8>>>(),
            };

            // Hash.
            // TODO(Ohad): Hash using SIMD parallelism.
            let next_hashed_layer: Vec<T::Hash> = T::hash_many(&injected_pre_hash);
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

pub fn init_col_length_map(cols: Vec<Vec<u32>>) -> BTreeMap<usize, Vec<Vec<u32>>> {
    let mut columns_size_map: BTreeMap<usize, Vec<Vec<u32>>> = BTreeMap::new();
    for c in cols {
        let length_index_entry = columns_size_map.entry(c.len()).or_default();
        info!("pushing column with length {}", c.len());
        length_index_entry.push(c);
    }
    columns_size_map
}

// Takes field element columns, transforms to bytes and concatenates corresponding column elements.
// TODO(Ohad): Consider optimizing using unsafe ops.
pub fn prepare_pre_hash(eq_length_cols: &[Vec<u32>]) -> Vec<Vec<u8>> {
    // Transpose columnns.
    let len = eq_length_cols[0].len();

    let transposed_col: Vec<Vec<[u8; 4]>> = (0..len)
        .map(|i| {
            eq_length_cols
                .iter()
                .map(|c| c[i].to_le_bytes())
                .collect::<Vec<[u8; 4]>>()
        })
        .collect();

    transposed_col
        .iter()
        .map(|r| r.concat())
        .collect::<Vec<Vec<u8>>>()
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;

    use super::{init_col_length_map, prepare_pre_hash, PartialTree};

    #[test]
    fn map_init_test() {
        let cols = init_test_trace();
        let mut map = init_col_length_map(cols);
        assert_eq!(map.get(&4).expect("no such key: 4").len(), 1);
        assert_eq!(map.get(&2).expect("no such key: 2").len(), 2);
        assert_eq!(map.pop_last().expect("Empty map").1[0][0], 1);
        assert_eq!(map.pop_last().expect("Empty map").1[1][0], 7);
        assert_eq!(map.pop_last().expect("Empty map").1[0][0], 9);
    }

    #[test]
    fn pre_hash_test() {
        let cols = init_test_trace();
        let mut map = init_col_length_map(cols);
        let layer_0_pre_hash = prepare_pre_hash(&map.pop_last().expect("Empty map!").1);
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

    #[test]
    fn partial_tree_commit_test() {
        let cols = init_test_trace();
        let partial_tree = PartialTree::<Blake3Hasher>::commit(cols);
        assert_eq!(
            partial_tree.root().to_string(),
            "748cc35381cd5a7af1d95c563beeff0585454891e602d83845ea7569009acd76"
        );
    }
}
