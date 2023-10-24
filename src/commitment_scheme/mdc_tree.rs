use super::{hasher::Hasher, NUM_BYTES_FELT};
use std::collections::BTreeMap;

// TODO(Ohad): consider more type-defs.
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
    pub data: Vec<Vec<T::Hash>>,
    pub height: usize,
    columns_size_map: ColumnLengthMap,
    partial_trees: Vec<MixedDegreeTree<T>>,
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
/// Endianess derived from platform. i.e: in X86 '1_u32' will be transposed to '1','0','0','0'.
/// # Safety
/// Pointers in 'dst' should point to pre-allocated memory with enough space to store column_array.len() amount of u32 elements.
// TODO(Ohad): Think about endianess.
pub unsafe fn transpose_to_bytes(column_array: &ColumnArray, dst: &[*mut u8]) {
    let column_length = column_array[0].len();
    assert_eq!(column_length, dst.len());

    for (i, ptr) in dst.iter().enumerate().take(column_length) {
        unsafe {
            let mut dst_ptr = *ptr;
            for c in column_array {
                std::ptr::copy_nonoverlapping(
                    c.as_ptr().add(i) as *mut u8,
                    dst_ptr,
                    NUM_BYTES_FELT,
                );
                dst_ptr = dst_ptr.add(NUM_BYTES_FELT);
            }
        }
    }
}

/// Transposes field-element columns directly into previous layer's hash results.
/// # Safety
/// Pointers in 'dst' should point to pre-allocated memory with enough space to store column_array.len() amount of u32 elements + 2*OUTPUT_SIZE of bytes.
pub unsafe fn inject<const OUTPUT_SIZE_BYTES: usize>(column_array: &ColumnArray, dst: &[*mut u8]) {
    let offseted_pointers: Vec<*mut u8> = dst
        .iter()
        .map(|p| unsafe { p.add(2 * OUTPUT_SIZE_BYTES) })
        .collect();
    transpose_to_bytes(column_array, &offseted_pointers);
}

#[cfg(test)]
mod tests {
    use super::{sort_columns_and_extract_remainder, ColumnArray, MAX_SUBTREE_BOTTOM_LAYER_LENGTH};
    use crate::commitment_scheme::mdc_tree::{inject, transpose_to_bytes};

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
            sort_columns_and_extract_remainder(cols.clone(), MAX_SUBTREE_BOTTOM_LAYER_LENGTH);

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
        let mut map = sort_columns_and_extract_remainder(cols, MAX_SUBTREE_BOTTOM_LAYER_LENGTH).0;
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
            transpose_to_bytes(&columns_to_transpose, &ptrs);
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
        let mut map = sort_columns_and_extract_remainder(cols, MAX_SUBTREE_BOTTOM_LAYER_LENGTH).0;
        let columns_to_transpose = map.pop_last().expect("msg").1;

        let mut out1 = [0_u8; 72];
        let mut out2 = [0_u8; 72];
        let mut out3 = [0_u8; 72];
        let mut out4 = [0_u8; 72];

        let ptrs = [
            out1.as_mut_ptr(),
            out2.as_mut_ptr(),
            out3.as_mut_ptr(),
            out4.as_mut_ptr(),
        ];
        unsafe {
            inject::<32>(&columns_to_transpose, &ptrs);
        }

        assert_eq!(hex::encode(out1), "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000005000000");
        assert_eq!(hex::encode(out2), "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000200000006000000");
        assert_eq!(hex::encode(out3), "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000300000007000000");
        assert_eq!(hex::encode(out4), "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000008000000");
    }
}
