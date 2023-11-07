use std::collections::BTreeMap;

use super::hasher::Hasher;

pub type ColumnArray = Vec<Vec<u32>>;
pub type ColumnLengthMap = BTreeMap<usize, Vec<Vec<u32>>>;
pub type TreeLayer = Box<[u8]>;
pub type TreeData = Box<[TreeLayer]>;

pub fn allocate_layer(n_bytes: usize) -> TreeLayer {
    // Safe bacuase 0 is a valid u8 value.
    unsafe { Box::<[u8]>::new_zeroed_slice(n_bytes).assume_init() }
}

pub fn allocate_balanced_tree(
    bottom_layer_length: usize,
    size_of_node_bytes: usize,
    output_size_bytes: usize,
) -> TreeData {
    assert!(output_size_bytes.is_power_of_two());
    let tree_height =
        crate::math::log2_ceil(bottom_layer_length * size_of_node_bytes / output_size_bytes);

    // Safe bacuse pointers are initilized later.
    let mut data: TreeData = unsafe { TreeData::new_zeroed_slice(tree_height).assume_init() };
    for i in 0..tree_height {
        let layer = allocate_layer(
            2_usize.pow((tree_height - i - 1).try_into().expect("Failed cast!"))
                * output_size_bytes,
        );
        data[i] = layer;
    }
    data
}

/// Performes a 2-to-1 hash on a layer of a merkle tree.
pub fn hash_layer<T: Hasher>(layer: &[u8], node_size: usize, dst: &mut [u8]) {
    let n_nodes_in_layer = crate::math::usize_div_ceil(layer.len(), node_size);
    assert!(n_nodes_in_layer.is_power_of_two());
    assert!(n_nodes_in_layer / 2 <= dst.len() / T::OUTPUT_SIZE_IN_BYTES);

    let src_ptrs: Vec<*const u8> = (0..n_nodes_in_layer)
        .map(|i| unsafe { layer.as_ptr().add(node_size * i) })
        .collect();
    let dst_ptrs: Vec<*mut u8> = (0..n_nodes_in_layer)
        .map(|i| unsafe { dst.as_mut_ptr().add(T::OUTPUT_SIZE_IN_BYTES * i) })
        .collect();

    // Safe because pointers are valid and distinct.
    unsafe {
        T::hash_many_in_place(&src_ptrs, node_size, &dst_ptrs);
    }
}

/// Takes columns that should be commited on, sorts and maps by length.
/// Extracts columns that are too long for handling by subtrees.
pub fn map_columns_sorted(
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
/// # Safety
/// Pointers in 'dst' should point to pre-allocated memory with enough space to store column_array.len() amount of u32 elements.
// TODO(Ohad): Think about endianess.
pub unsafe fn transpose_to_bytes<const ELEMENT_SIZE_BYTES: usize>(
    column_array: &ColumnArray,
    dst: &[*mut u8],
) {
    let column_length = column_array[0].len();
    assert_eq!(column_length, dst.len());

    for (i, ptr) in dst.iter().enumerate().take(column_length) {
        unsafe {
            let mut dst_ptr = *ptr;
            for c in column_array {
                std::ptr::copy_nonoverlapping(
                    c.as_ptr().add(i) as *mut u8,
                    dst_ptr,
                    ELEMENT_SIZE_BYTES,
                );
                dst_ptr = dst_ptr.add(ELEMENT_SIZE_BYTES);
            }
        }
    }
}

/// Inject columns to pre-allocated arrays.
/// # Safety
/// Pointers in 'dst' should point to pre-allocated memory with enough space to store column_array.len() amount of u32 elements + 2*OUTPUT_SIZE of bytes.
pub unsafe fn inject<const OUTPUT_SIZE_BYTES: usize, const ELEMENT_SIZE_BYTES: usize>(
    column_array: &ColumnArray,
    dst: &mut [u8],
) {
    let offset = column_array.len() * ELEMENT_SIZE_BYTES + 2 * OUTPUT_SIZE_BYTES;
    let offseted_pointers: Vec<*mut u8> = (2 * OUTPUT_SIZE_BYTES..dst.len())
        .step_by(offset)
        .map(|i| unsafe { dst.as_mut_ptr().add(i) })
        .collect();
    transpose_to_bytes::<ELEMENT_SIZE_BYTES>(column_array, &offseted_pointers);
}

#[cfg(test)]
mod tests {
    use super::{allocate_balanced_tree, map_columns_sorted, ColumnArray};
    use crate::commitment_scheme::{
        blake3_hash::Blake3Hasher,
        hasher::Hasher,
        utils::{allocate_layer, hash_layer, inject, transpose_to_bytes},
    };

    const MAX_SUBTREE_BOTTOM_LAYER_LENGTH: usize = 64;

    fn init_test_trace() -> ColumnArray {
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

    fn init_transpose_test_trace() -> ColumnArray {
        let col1 = vec![1, 2, 3, 4];
        let col2 = vec![5, 6, 7, 8];
        let col3 = vec![9, 10];
        let col4 = vec![11];
        let cols: ColumnArray = vec![col1, col2, col3, col4];
        cols
    }

    #[test]
    fn sort_columns_and_extract_remainder_test() {
        let cols = init_test_trace();
        let (mut col_length_map, remainder_columns) =
            map_columns_sorted(cols.clone(), MAX_SUBTREE_BOTTOM_LAYER_LENGTH);

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
    fn transpose_test() {
        let cols = init_transpose_test_trace();
        let mut map = map_columns_sorted(cols, MAX_SUBTREE_BOTTOM_LAYER_LENGTH).0;
        let columns_to_transpose = map.pop_last().expect("msg").1;

        let mut out1 = [0_u8; 8];
        let mut out2 = [0_u8; 8];
        let mut out3 = [0_u8; 8];
        let mut out4 = [0_u8; 8];

        let ptrs = [
            out1.as_mut_ptr(),
            out2.as_mut_ptr(),
            out3.as_mut_ptr(),
            out4.as_mut_ptr(),
        ];
        unsafe {
            transpose_to_bytes::<4>(&columns_to_transpose, &ptrs);
        }

        let outs = [out1, out2, out3, out4];
        assert_eq!(
            format!("{:?}", outs),
            "[[1, 0, 0, 0, 5, 0, 0, 0], [2, 0, 0, 0, 6, 0, 0, 0], [3, 0, 0, 0, 7, 0, 0, 0], [4, 0, 0, 0, 8, 0, 0, 0]]"
        );
    }

    // TODO(Ohad): generelize over a hash function and use hash-in-place functions to initialize output arrays instead of zeros.
    #[test]
    fn inject_test() {
        let cols = init_transpose_test_trace();
        let mut map = map_columns_sorted(cols, MAX_SUBTREE_BOTTOM_LAYER_LENGTH).0;
        let columns_to_transpose = map.pop_last().expect("msg").1;
        let mut out = [0_u8; 288];
        unsafe {
            inject::<{ Blake3Hasher::OUTPUT_SIZE_IN_BYTES }, 4>(
                &columns_to_transpose,
                &mut out[..],
            );
        }
        assert_eq!(hex::encode(out), "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000005000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000200000006000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000300000007000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000008000000");
    }

    #[test]
    fn allocate_layer_test() {
        let layer = allocate_layer(10);
        assert_eq!(layer.len(), 10);
    }

    #[test]
    fn allocate_empty_layer_test() {
        let layer = allocate_layer(0);
        assert_eq!(layer.len(), 0);
    }

    #[test]
    fn allocate_balanced_tree_test() {
        let tree = allocate_balanced_tree(8, 64, 32);
        assert_eq!(tree.len(), 4);
        assert_eq!(tree[0].len(), 8 * 32);
        assert_eq!(tree[1].len(), 4 * 32);
        assert_eq!(tree[2].len(), 2 * 32);
        assert_eq!(tree[3].len(), 32);
    }

    #[test]
    fn hash_layer_test() {
        let layer = allocate_layer(16);
        let mut res_layer = allocate_layer(64);
        hash_layer::<Blake3Hasher>(&layer, 8, &mut res_layer);
        assert_eq!(
            hex::encode(&res_layer[..Blake3Hasher::OUTPUT_SIZE_IN_BYTES]),
            Blake3Hasher::hash(&0u64.to_le_bytes()).to_string()
        );
    }
}
