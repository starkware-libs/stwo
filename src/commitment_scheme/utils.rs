use std::collections::BTreeMap;
use std::slice::Iter;

use super::hasher::Hasher;
use crate::core::fields::{Field, IntoSlice};
use crate::math::utils::{log2_ceil, usize_safe_div};

pub type ColumnArray<T> = Vec<Vec<T>>;
pub type ColumnLengthMap<T> = BTreeMap<usize, ColumnArray<T>>;
pub type TreeLayer<T> = Box<[T]>;
pub type TreeData<T> = Box<[TreeLayer<T>]>;

pub fn allocate_layer<T: Sized>(n_bytes: usize) -> TreeLayer<T> {
    // Safe bacuase 0 is a valid u8 value.
    unsafe { Box::<[T]>::new_zeroed_slice(n_bytes).assume_init() }
}

pub fn allocate_balanced_tree<T: Sized>(
    bottom_layer_length: usize,
    size_of_node_bytes: usize,
    output_size_bytes: usize,
) -> TreeData<T> {
    assert!(output_size_bytes.is_power_of_two());
    let tree_height = log2_ceil(bottom_layer_length * size_of_node_bytes / output_size_bytes);

    // Safe because pointers are initialized later.
    let mut data: TreeData<T> = unsafe { TreeData::new_zeroed_slice(tree_height).assume_init() };
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
pub fn hash_layer<H: Hasher>(layer: &[H::NativeType], node_size: usize, dst: &mut [H::NativeType]) {
    let n_nodes_in_layer = usize_safe_div(layer.len(), node_size);
    assert!(n_nodes_in_layer.is_power_of_two());
    assert!(n_nodes_in_layer <= dst.len() / H::OUTPUT_SIZE);

    let src_ptrs: Vec<*const H::NativeType> = (0..n_nodes_in_layer)
        .map(|i| unsafe { layer.as_ptr().add(node_size * i) })
        .collect();
    let dst_ptrs: Vec<*mut H::NativeType> = (0..n_nodes_in_layer)
        .map(|i| unsafe { dst.as_mut_ptr().add(H::OUTPUT_SIZE * i) })
        .collect();

    // Safe because pointers are valid and distinct.
    unsafe {
        H::hash_many_in_place(&src_ptrs, node_size, &dst_ptrs);
    }
}

// Given a data of a tree, hashes the entire tree.
pub fn hash_merkle_tree<H: Hasher>(data: &mut [&mut [H::NativeType]]) {
    (0..data.len() - 1).for_each(|i| {
        let (src, dst) = data.split_at_mut(i + 1);
        let src = src.get(i).unwrap();
        let dst = dst.get_mut(0).unwrap();
        hash_layer::<H>(src, H::BLOCK_SIZE, dst)
    })
}

/// Given a data of a tree, and a bottom layer of 'bottom_layer_node_size_bytes' sized nodes, hashes
/// the entire tree. Nodes are hashed individually at the bottom layer.
// TODO(Ohad): Write a similiar function for when F does not implement IntoSlice(Non le platforms).
pub fn hash_merkle_tree_from_bottom_layer<'a, F: Field, H: Hasher>(
    bottom_layer: &[F],
    bottom_layer_node_size_bytes: usize,
    data: &mut [&mut [H::NativeType]],
) where
    F: IntoSlice<H::NativeType>,
    H::NativeType: 'a,
{
    // Hash bottom layer.
    let dst_slice = data.get_mut(0).expect("Empty tree!");
    let bottom_layer_data: &[H::NativeType] =
        <F as IntoSlice<H::NativeType>>::into_slice(bottom_layer);
    hash_layer::<H>(bottom_layer_data, bottom_layer_node_size_bytes, dst_slice);

    // Rest of the sub-tree
    hash_merkle_tree::<H>(data);
}

/// Maps columns by length.
/// Mappings are sorted by length. i.e the first entry is a matrix of the shortest columns.
pub fn map_columns_sorted<T: Sized>(cols: ColumnArray<T>) -> ColumnLengthMap<T> {
    let mut columns_length_map: ColumnLengthMap<T> = BTreeMap::new();
    for c in cols {
        let length_index_entry = columns_length_map.entry(c.len()).or_default();
        length_index_entry.push(c);
    }
    columns_length_map
}

/// Given columns of the same length, transforms to bytes and concatenates corresponding column
/// elements. Assumes columns are of the same length.
///
/// # Safety
///
/// Pointers in 'dst' should point to pre-allocated memory with enough space to store
/// column_array.len() amount of u32 elements.
// TODO(Ohad): Change tree impl and remove.
pub unsafe fn transpose_to_bytes<T: Sized>(column_array: &ColumnArray<T>, dst: &[*mut u8]) {
    let column_length = column_array[0].len();

    for (i, ptr) in dst.iter().enumerate().take(column_length) {
        unsafe {
            let mut dst_ptr = *ptr;
            for c in column_array {
                std::ptr::copy_nonoverlapping(
                    c.as_ptr().add(i) as *mut u8,
                    dst_ptr,
                    std::mem::size_of::<T>(),
                );
                dst_ptr = dst_ptr.add(std::mem::size_of::<T>());
            }
        }
    }
}

pub fn tree_data_as_mut_ref<T: Sized>(tree_data: &mut TreeData<T>) -> Vec<&mut [T]> {
    tree_data.iter_mut().map(|layer| &mut layer[..]).collect()
}

/// Inject columns to pre-allocated arrays.
///
/// # Arguments
///
/// * 'gap_offset' - The offset in bytes between end of target to the beginning of the next.
///
/// # Safety
///
/// dst should point to pre-allocated memory with enough space to store the entire column array +
/// offset*(n_rows/n_rows_in_node) amount of T elements.
// TODO(Ohad): Change tree impl and remove.
pub unsafe fn inject<T: Sized>(
    column_array: &ColumnArray<T>,
    dst: &mut [u8],
    n_rows_in_node: usize,
    gap_offset: usize,
) {
    let ptr_offset = column_array.len() * n_rows_in_node * std::mem::size_of::<T>() + gap_offset;
    let offseted_pointers: Vec<*mut u8> = (gap_offset..dst.len())
        .step_by(ptr_offset)
        .map(|i| unsafe { dst.as_mut_ptr().add(i) })
        .collect();
    transpose_to_bytes::<T>(column_array, &offseted_pointers);
}

/// Given a matrix, returns a vector of the matrix elements in row-major order.
/// Assumes all columns are of the same length and non-zero.
// TODO(Ohad): Change tree impl and remove.
pub fn column_to_row_major<T>(mut mat: ColumnArray<T>) -> Vec<T> {
    if mat.len() == 1 {
        return mat.remove(0);
    };

    // Flattening the matrix into a single vector.
    let vec_length = mat.len() * mat[0].len();
    let mut row_major_matrix_vec: Vec<T> = Vec::with_capacity(vec_length);

    // Inject(transpose).
    // Safe because enough memory is allocated.
    unsafe {
        let row_major_matrix_byte_slice = std::slice::from_raw_parts_mut(
            row_major_matrix_vec.as_mut_ptr() as *mut u8,
            vec_length * std::mem::size_of::<T>(),
        );
        inject(&mat, row_major_matrix_byte_slice, 1, 0);
        row_major_matrix_vec.set_len(vec_length);
    }
    row_major_matrix_vec
}

pub fn inject_hash_in_pairs<'a: 'b, 'b, H: Hasher>(
    hash_inputs: &'b mut [Vec<&'a [H::NativeType]>],
    values_to_inject: &'a [H::Hash],
) {
    assert_eq!(
        values_to_inject.len(),
        hash_inputs.len() * 2,
        "Attempted injecting {} hash values into {} hash inputs",
        values_to_inject.len(),
        hash_inputs.len()
    );
    for (j, hashes) in values_to_inject.chunks(2).enumerate() {
        // TODO(Ohad): Implement 'IntoSlice' for H::Hash and reduce here to one push.
        hash_inputs[j].push(hashes[0].as_ref());
        hash_inputs[j].push(hashes[1].as_ref());
    }
}

/// Injects Field element values into existing hash inputs.
///
/// For Large [MerkleTree] constructions, holding reference-arrays for every node
/// in a layer hinders performance greatly. Hence, the input is traversed in small chunks,
/// and the refrences are discarded upon use.
///
/// # Arguments
///
///    * `columns` - an array of injection element columns.
///    * `hash_inputs` - The hash inputs to inject into.
///    * `chunk_idx` - The index of the chunk to inject.
///    * `n_chunks_in_column` - The number of chunks every column is divided into.
///  
pub fn inject_column_chunks<'b, 'a: 'b, H: Hasher, F: Field>(
    columns: &'a [&'a [F]],
    hash_inputs: &'b mut [Vec<&'a [H::NativeType]>],
    chunk_idx: usize,
    n_chunks_in_column: usize,
) where
    F: IntoSlice<H::NativeType>,
{
    for column in columns {
        let column_slice = get_column_chunk(column, chunk_idx, n_chunks_in_column);

        // TODO(Ohad): consider implementing a 'duplicate' feature and changing or removing this
        // assert.
        assert_eq!(
            column_slice.len() % hash_inputs.len(),
            0,
            "Column of size {} can not be divided into {} hash inputs",
            column_slice.len(),
            hash_inputs.len()
        );
        let n_elements_in_chunk = column_slice.len() / hash_inputs.len();
        column_slice
            .chunks(n_elements_in_chunk)
            .zip(hash_inputs.iter_mut())
            .for_each(|(chunk, hash_input)| {
                hash_input.push(F::into_slice(chunk));
            });
    }
}

fn get_column_chunk<F>(column: &[F], index_to_view: usize, n_total_chunks: usize) -> &[F] {
    let slice_length = column.len() / n_total_chunks;
    let slice_start_idx = slice_length * index_to_view;
    &column[slice_start_idx..slice_start_idx + slice_length]
}

// TODO(Ohad): Consider renaming after current hash_layer function is deprecated.
pub fn inject_and_hash_layer<H: Hasher, F: Field, const IS_INTERMEDIATE: bool>(
    prev_hashes: &[H::Hash],
    dst: &mut [H::Hash],
    input_columns: &Iter<'_, &[F]>,
) where
    F: IntoSlice<H::NativeType>,
{
    let produced_layer_length = dst.len();
    if IS_INTERMEDIATE {
        assert_eq!(prev_hashes.len(), produced_layer_length * 2);
    }

    let mut hasher = H::new();
    match input_columns.clone().peekable().next() {
        Some(_) => {
            // Match the input columns to corresponding chunk sizes.
            let input_columns: Vec<_> = input_columns
                .clone()
                .zip(
                    input_columns
                        .clone()
                        .map(|c| c.len() / produced_layer_length),
                )
                .collect();
            dst.iter_mut().enumerate().for_each(|(i, dst)| {
                // Inject previous hash values if intermediate layer, and input columns.
                if IS_INTERMEDIATE {
                    _inject_previous_hash_values::<H>(i, &mut hasher, prev_hashes);
                }
                for (column, n_elements_in_chunk) in input_columns.iter() {
                    let chunk = &column[i * n_elements_in_chunk..(i + 1) * n_elements_in_chunk];
                    hasher.update(F::into_slice(chunk));
                }
                *dst = hasher.finalize_reset();
            });
        }
        None => {
            // No input columns, but intermediate layer.
            dst.iter_mut().enumerate().for_each(|(i, dst)| {
                _inject_previous_hash_values::<H>(i, &mut hasher, prev_hashes);
                *dst = hasher.finalize_reset();
            });
        }
    }
}

fn _inject_previous_hash_values<H: Hasher>(
    i: usize,
    hash_state: &mut H,
    prev_hashes: &[<H as Hasher>::Hash],
) {
    hash_state.update(prev_hashes[i * 2].as_ref());
    hash_state.update(prev_hashes[i * 2 + 1].as_ref());
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::{
        allocate_balanced_tree, inject_and_hash_layer, inject_column_chunks, map_columns_sorted,
        ColumnArray,
    };
    use crate::commitment_scheme;
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_input::MerkleTreeInput;
    use crate::commitment_scheme::utils::{
        allocate_layer, hash_layer, hash_merkle_tree, hash_merkle_tree_from_bottom_layer, inject,
        inject_hash_in_pairs, transpose_to_bytes, tree_data_as_mut_ref,
    };
    use crate::core::fields::m31::M31;
    use crate::math::utils::log2_ceil;

    fn init_test_trace() -> ColumnArray<u32> {
        let col0 = std::iter::repeat(0).take(8).collect();
        let col1 = vec![1, 2, 3, 4];
        let col2 = vec![5, 6];
        let col3 = vec![7, 8];
        let col4 = vec![9];
        let cols: ColumnArray<u32> = vec![col0, col1, col2, col3, col4];
        cols
    }

    fn init_transpose_test_trace() -> ColumnArray<u32> {
        let col1 = vec![1, 2, 3, 4];
        let col2 = vec![5, 6, 7, 8];
        let col3 = vec![9, 10];
        let col4 = vec![11];
        let cols: ColumnArray<u32> = vec![col1, col2, col3, col4];
        cols
    }

    #[test]
    fn sort_columns_and_extract_remainder_test() {
        let cols = init_test_trace();
        let mut col_length_map = map_columns_sorted(cols.clone());

        // Test map
        assert_eq!(col_length_map.get(&4).expect("no such key: 4").len(), 1);
        assert_eq!(col_length_map.get(&2).expect("no such key: 2").len(), 2);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 0);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 1);
        assert_eq!(col_length_map.pop_last().expect("Empty map").1[0][0], 5);
    }

    #[test]
    fn transpose_test() {
        let cols = init_transpose_test_trace();
        let mut map = map_columns_sorted(cols);
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

    // TODO(Ohad): generelize over a hash function and use hash-in-place functions
    // to initialize output arrays instead of zeros.
    #[test]
    fn inject_test() {
        let cols = init_transpose_test_trace();
        let mut map = map_columns_sorted(cols);
        let columns_to_transpose = map.pop_last().expect("msg").1; // [[1, 2, 3, 4],[5, 6, 7, 8]].
        let gap_offset: usize = 1;
        let mut out = [0_u8; 36];
        unsafe {
            inject::<u32>(&columns_to_transpose, &mut out[..], 1, gap_offset);
        }

        assert_eq!(
            hex::encode(&out[..]),
            hex::encode(0u8.to_le_bytes())
                + &hex::encode(1u32.to_le_bytes())
                + &hex::encode(5u32.to_le_bytes())
                + &hex::encode(0u8.to_le_bytes())
                + &hex::encode(2u32.to_le_bytes())
                + &hex::encode(6u32.to_le_bytes())
                + &hex::encode(0u8.to_le_bytes())
                + &hex::encode(3u32.to_le_bytes())
                + &hex::encode(7u32.to_le_bytes())
                + &hex::encode(0u8.to_le_bytes())
                + &hex::encode(4u32.to_le_bytes())
                + &hex::encode(8u32.to_le_bytes())
        );
    }

    #[test]
    fn allocate_layer_test() {
        let layer = allocate_layer::<u8>(10);
        assert_eq!(layer.len(), 10);
    }

    #[test]
    fn allocate_empty_layer_test() {
        let layer = allocate_layer::<u8>(0);
        assert_eq!(layer.len(), 0);
    }

    #[test]
    fn allocate_balanced_tree_test() {
        let n_nodes = 8;
        let node_size = Blake3Hasher::BLOCK_SIZE;
        let output_size = Blake3Hasher::OUTPUT_SIZE;
        let tree = allocate_balanced_tree::<u8>(n_nodes, node_size, output_size);

        assert_eq!(tree.len(), log2_ceil(n_nodes) + 1);
        assert_eq!(tree[0].len(), n_nodes * output_size);
        assert_eq!(tree[1].len(), 4 * output_size);
        assert_eq!(tree[2].len(), 2 * output_size);
        assert_eq!(tree[3].len(), output_size);
    }

    #[test]
    fn hash_layer_test() {
        let layer = allocate_layer(16);
        let mut res_layer = allocate_layer(64);
        hash_layer::<Blake3Hasher>(&layer, 8, &mut res_layer);
        assert_eq!(
            hex::encode(&res_layer[..Blake3Hasher::OUTPUT_SIZE]),
            Blake3Hasher::hash(&0u64.to_le_bytes()).to_string()
        );
    }

    #[test]
    fn hash_tree_test() {
        let mut tree_data =
            allocate_balanced_tree(16, Blake3Hasher::BLOCK_SIZE, Blake3Hasher::OUTPUT_SIZE);

        hash_merkle_tree::<Blake3Hasher>(&mut tree_data_as_mut_ref(&mut tree_data)[..]);

        assert_eq!(
            hex::encode(tree_data.last().unwrap()),
            "31b471b27b22b57b1ac82c9ed537231d53faf017fbe0c903c9668f47dc4151e1"
        )
    }

    #[test]
    fn hash_tree_from_bottom_layer_test() {
        const TEST_SIZE: usize = 512;

        let bottom_layer = [M31::one(); TEST_SIZE];
        let mut tree_data = allocate_balanced_tree(
            TEST_SIZE * std::mem::size_of::<M31>() / Blake3Hasher::BLOCK_SIZE,
            Blake3Hasher::BLOCK_SIZE,
            Blake3Hasher::OUTPUT_SIZE,
        );

        hash_merkle_tree_from_bottom_layer::<M31, Blake3Hasher>(
            &bottom_layer[..],
            Blake3Hasher::BLOCK_SIZE,
            &mut tree_data_as_mut_ref(&mut tree_data)[..],
        );

        assert_eq!(
            hex::encode(tree_data.last().unwrap()),
            "234d7011f24adb0fec6604ff1fdfe4745340886418b6e2cd0633f6ad1c7e52d9"
        )
    }

    #[test]
    fn inject_hash_in_pairs_test() {
        let mut hash_inputs = vec![vec![], vec![]];
        let values_to_inject = vec![
            Blake3Hasher::hash(b"a"),
            Blake3Hasher::hash(b"b"),
            Blake3Hasher::hash(b"c"),
            Blake3Hasher::hash(b"d"),
        ];

        inject_hash_in_pairs::<Blake3Hasher>(&mut hash_inputs, &values_to_inject);

        assert_eq!(
            hex::encode(hash_inputs[0][0]),
            Blake3Hasher::hash(b"a").to_string()
        );
        assert_eq!(
            hex::encode(hash_inputs[0][1]),
            Blake3Hasher::hash(b"b").to_string()
        );
        assert_eq!(
            hex::encode(hash_inputs[1][0]),
            Blake3Hasher::hash(b"c").to_string()
        );
        assert_eq!(
            hex::encode(hash_inputs[1][1]),
            Blake3Hasher::hash(b"d").to_string()
        );
    }

    #[test]
    fn inject_column_single_chunk_test() {
        let col1: Vec<M31> = (0..4).map(M31::from_u32_unchecked).collect();
        let col2: Vec<M31> = (4..8).map(M31::from_u32_unchecked).collect();
        let columns = vec![&col1[..], &col2[..]];
        let mut hash_inputs = vec![vec![], vec![]];

        inject_column_chunks::<Blake3Hasher, M31>(&columns, &mut hash_inputs, 0, 1);

        assert_eq!(
            format!("{:?}", hash_inputs[0]),
            "[[0, 0, 0, 0, 1, 0, 0, 0], [4, 0, 0, 0, 5, 0, 0, 0]]"
        );
        assert_eq!(
            format!("{:?}", hash_inputs[1]),
            "[[2, 0, 0, 0, 3, 0, 0, 0], [6, 0, 0, 0, 7, 0, 0, 0]]"
        );
    }

    #[test]
    fn inject_column_multiple_chunks_test() {
        let col1: Vec<M31> = (0..4).map(M31::from_u32_unchecked).collect();
        let col2: Vec<M31> = (4..8).map(M31::from_u32_unchecked).collect();
        let columns = vec![&col1[..], &col2[..]];
        let mut hash_inputs = vec![vec![], vec![]];

        // Inject the second half of the columns.
        inject_column_chunks::<Blake3Hasher, M31>(&columns, &mut hash_inputs, 1, 2);

        assert_eq!(
            format!("{:?}", hash_inputs[0]),
            "[[2, 0, 0, 0], [6, 0, 0, 0]]"
        );
        assert_eq!(
            format!("{:?}", hash_inputs[1]),
            "[[3, 0, 0, 0], [7, 0, 0, 0]]"
        );
    }

    #[test]
    #[should_panic]
    fn inject_columns_too_short_test() {
        let col1: Vec<M31> = (0..1).map(M31::from_u32_unchecked).collect();
        let columns = vec![&col1[..]];
        let mut hash_inputs = vec![vec![], vec![]];

        inject_column_chunks::<Blake3Hasher, M31>(&columns, &mut hash_inputs, 0, 1);
    }

    #[test]
    #[should_panic]
    fn inject_columns_size_not_divisible_test() {
        let col1: Vec<M31> = (0..3).map(M31::from_u32_unchecked).collect();
        let columns = vec![&col1[..]];
        let mut hash_inputs = vec![vec![], vec![]];

        inject_column_chunks::<Blake3Hasher, M31>(&columns, &mut hash_inputs, 0, 1);
    }

    #[test]
    fn inject_and_hash_bottom_layer_test() {
        // trace_column: [M31;16] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        let trace_column = (0..16).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let sub_trees_height = 3;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column);
        let mut dst_layer = vec![<commitment_scheme::blake3_hash::Blake3Hasher as commitment_scheme::hasher::Hasher>::Hash::default();1 << sub_trees_height];

        inject_and_hash_layer::<commitment_scheme::blake3_hash::Blake3Hasher, M31, false>(
            &[],
            &mut dst_layer,
            &input.get_columns(sub_trees_height).iter(),
        );

        // dst_layer: [Blake3Hash;8] = [h(0,1),h(2,3),...h(14,15)]
        dst_layer.iter().enumerate().for_each(|(i, h)| {
            let mut hasher = Blake3Hasher::new();
            hasher.update(&(2 * i as u32).to_le_bytes());
            hasher.update(&((2 * i + 1) as u32).to_le_bytes());
            assert_eq!(h, &hasher.finalize());
        })
    }

    #[test]
    fn inject_and_hash_intermediate_layer_test() {
        // trace_column: [M31;16] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        let trace_column_0 = (0..16).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let trace_column_1 = (0..8).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let sub_trees_height = 3;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column_0);
        input.insert_column(sub_trees_height - 1, &trace_column_1);
        let mut dst_layer_0 = vec![<commitment_scheme::blake3_hash::Blake3Hasher as commitment_scheme::hasher::Hasher>::Hash::default();1 << sub_trees_height];
        let mut dst_layer_1 = vec![<commitment_scheme::blake3_hash::Blake3Hasher as commitment_scheme::hasher::Hasher>::Hash::default();1 << (sub_trees_height-1)];

        inject_and_hash_layer::<commitment_scheme::blake3_hash::Blake3Hasher, M31, false>(
            &[],
            &mut dst_layer_0,
            &input.get_columns(sub_trees_height).iter(),
        );
        inject_and_hash_layer::<commitment_scheme::blake3_hash::Blake3Hasher, M31, true>(
            &dst_layer_0[..],
            &mut dst_layer_1,
            &input.get_columns(sub_trees_height - 1).iter(),
        );

        // dst_layer_1: [Blake3Hash;4] = [h(h(0,1),h(2,3),0,1),,...h(h(12,13),h(14,15),6,7)]
        dst_layer_1.iter().enumerate().for_each(|(i, h)| {
            let mut hasher = Blake3Hasher::new();
            hasher.update(dst_layer_0[i * 2].as_ref());
            hasher.update(dst_layer_0[i * 2 + 1].as_ref());
            hasher.update(&((2 * i) as u32).to_le_bytes());
            hasher.update(&((2 * i + 1) as u32).to_le_bytes());
            assert_eq!(h, &hasher.finalize());
        })
    }

    #[test]
    fn hash_intermediate_layer_no_inject_test() {
        // trace_column: [M31;16] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        let trace_column_0 = (0..16).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let sub_trees_height = 3;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column_0);
        let mut dst_layer_0 = vec![<commitment_scheme::blake3_hash::Blake3Hasher as commitment_scheme::hasher::Hasher>::Hash::default();1 << sub_trees_height];
        let mut dst_layer_1 = vec![<commitment_scheme::blake3_hash::Blake3Hasher as commitment_scheme::hasher::Hasher>::Hash::default();1 << (sub_trees_height-1)];

        inject_and_hash_layer::<commitment_scheme::blake3_hash::Blake3Hasher, M31, false>(
            &[],
            &mut dst_layer_0,
            &input.get_columns(sub_trees_height).iter(),
        );
        inject_and_hash_layer::<commitment_scheme::blake3_hash::Blake3Hasher, M31, true>(
            &dst_layer_0[..],
            &mut dst_layer_1,
            &input.get_columns(sub_trees_height - 1).iter(),
        );

        // dst_layer_1: [Blake3Hash;4] = [h(h(0,1),h(2,3)),...h(h(12,13),h(14,15))]
        dst_layer_1.iter().enumerate().for_each(|(i, h)| {
            let mut hasher = Blake3Hasher::new();
            hasher.update(dst_layer_0[i * 2].as_ref());
            hasher.update(dst_layer_0[i * 2 + 1].as_ref());
            assert_eq!(h, &hasher.finalize());
        })
    }
}
