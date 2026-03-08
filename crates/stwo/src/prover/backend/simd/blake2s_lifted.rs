use std::array;
use std::mem::transmute;
use std::simd::u32x16;

use bytemuck::cast_slice;
use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::m31::LOG_N_LANES;
use super::utils::to_lifted_simd;
use super::SimdBackend;
use crate::core::fields::m31::{BaseField, N_BYTES_FELT};
use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::utils::uninit_vec;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
use crate::parallel_iter;
use crate::prover::backend::simd::blake2s::{
    compress_finalize, compress_unfinalized, transpose_msgs, untranspose_states, INITIAL_STATE,
};
use crate::prover::backend::simd::column::BaseColumn;
use crate::prover::backend::simd::m31::{reduce_to_m31_simd, PackedBaseField, N_LANES};
use crate::prover::backend::simd::utils::transpose_packed_leaf;
use crate::prover::backend::{Col, Column, CpuBackend};
use crate::prover::vcs_lifted::ops::{MerkleOpsLifted, PackLeavesOps};

const N_FELTS_IN_BLAKE_MESSAGE: usize = 16;
const N_FELTS_IN_BLAKE_STATE: usize = 8;
const N_BYTES_IN_BLAKE_MESSAGE: u64 = N_FELTS_IN_BLAKE_MESSAGE as u64 * N_BYTES_FELT as u64;
const LOG_N_HASHES_PER_SIMD_STATE: u32 = 4;

impl<const IS_M31_OUTPUT: bool> MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>
    for SimdBackend
{
    /// See the docs of [`crate::prover::backend::cpu::blake2s_lifted`].
    ///
    /// This function assumes that `columns` is sorted increasingly by column length.
    ///
    /// # Note
    ///
    /// If the length of a smallest column (e.g. the first) is smaller than `N_LANES`, the
    /// implementation falls back to the CPU implementation.
    #[allow(clippy::uninit_vec)]
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Blake2sHash> {
        if columns.is_empty() {
            let hasher = Blake2sMerkleHasherGeneric::<IS_M31_OUTPUT>::default();
            return vec![hasher.finalize()];
        }
        if columns.first().unwrap().len() < N_LANES {
            let cpu_cols = columns.iter().map(|column| column.to_cpu()).collect_vec();
            return <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_leaves(
                &cpu_cols.iter().collect_vec(),
                lifting_log_size,
            );
        }
        // Note that, in this function, all variables that track log sizes
        // refer to the "size" in terms of PackedM31 (e.g. the log size of a column
        // of 4 PackedM31 elements is 2).
        let max_log_size: u32 = columns.last().unwrap().data.len().ilog2();

        // Initialize the vector of Blake2s states. The state is of type `[u32x16; 8]`.
        //
        // We use two large buffers to hold the intermediate results of the computation.
        // In every iteration, a possibly larger chunk of the buffer is used. This
        // saves memory allocations.
        // Safety: no index in `next_layer_states` and `prev_layer_states` is ever read without
        // having been written to before.
        let mut prev_layer_states: Vec<[u32x16; N_FELTS_IN_BLAKE_STATE]> =
            unsafe { uninit_vec(1 << max_log_size) };
        let mut next_layer_states: Vec<[u32x16; N_FELTS_IN_BLAKE_STATE]> =
            unsafe { uninit_vec(1 << max_log_size) };

        #[cfg(not(feature = "parallel"))]
        prev_layer_states.fill(INITIAL_STATE);
        #[cfg(feature = "parallel")]
        prev_layer_states
            .par_iter_mut()
            .for_each(|uninit| *uninit = INITIAL_STATE);

        // The last column chunk, which requires the `compress_finalize` permutation, is
        // `columns[last_chunk_index..]`. This chunk is treated on its own towards the end of the
        // function.
        let last_chunk_index =
            (columns.len() - 1) / N_FELTS_IN_BLAKE_MESSAGE * N_FELTS_IN_BLAKE_MESSAGE;
        let lifting_indices =
            get_lifting_indices(columns.iter().map(|c| c.data.len()), last_chunk_index);
        let mut byte_count = 0_u64;

        // The actual log size of `prev_layer_states` is equal to `max_log_size`, but only the first
        // two entries are accessed for the computation of the first iteration.
        let mut prev_chunk_max_log_size = 0;
        for (start, end) in lifting_indices.into_iter().tuple_windows() {
            let chunk_max_log_size: u32 = columns[end - 1].data.len().ilog2();
            let next_layer_state_slice = &mut next_layer_states[0..1 << chunk_max_log_size];
            let log_ratio = chunk_max_log_size - prev_chunk_max_log_size;
            // Compute the new states of the current layer.
            #[cfg(not(feature = "parallel"))]
            let iter_states = next_layer_state_slice.iter_mut();
            #[cfg(feature = "parallel")]
            let iter_states = next_layer_state_slice.par_iter_mut();

            iter_states.enumerate().for_each(|(i, state)| {
                let mut local_byte_count = byte_count + N_BYTES_IN_BLAKE_MESSAGE;
                // Lift `prev_layer_states` and the first chunk `columns[start..start + 16]`.
                let prev_state = std::array::from_fn(|j| {
                    let prev_state_limb = prev_layer_states[i >> log_ratio][j];
                    to_lifted_simd(prev_state_limb, log_ratio, i)
                });
                let msgs: [u32x16; N_FELTS_IN_BLAKE_MESSAGE] = std::array::from_fn(|j| {
                    let column = columns[start + j];
                    let log_size = column.data.len().ilog2();
                    let log_ratio = chunk_max_log_size - log_size;
                    to_lifted_simd(column.data[i >> log_ratio].into_simd(), log_ratio, i)
                });

                *state = compress_unfinalized(prev_state, msgs, local_byte_count);
                // Deal with the subsequent chunks in columns[start + 16..end]`. Note that since
                // `start < end` and both are multiples of 16, we have `start + 16 <= end`,
                // therefore the indexing range below doesn't panic. All columns in
                // `columns[start + 16..end]` are guaranteed to be of the same size (hence no
                // lifting is required).
                for chunk_columns in &mut columns[start + 16..end].chunks(N_FELTS_IN_BLAKE_MESSAGE)
                {
                    let msgs: [u32x16; N_FELTS_IN_BLAKE_MESSAGE] =
                        std::array::from_fn(|j| chunk_columns[j].data[i].into_simd());
                    local_byte_count += N_BYTES_IN_BLAKE_MESSAGE;
                    *state = compress_unfinalized(*state, msgs, local_byte_count);
                }
            });
            // We hashed `((end - start) / N_FELTS_IN_BLAKE_MESSAGE) * N_BYTES_IN_BLAKE_MESSAGE = 4
            // * (end - start)` bytes.
            byte_count += 4 * (end - start) as u64;
            std::mem::swap(&mut prev_layer_states, &mut next_layer_states);
            prev_chunk_max_log_size = chunk_max_log_size;
        }

        // Process last chunk.
        let chunk_max_log_size: u32 = max_log_size;
        let next_layer_state_slice = &mut next_layer_states[0..1 << chunk_max_log_size];
        let log_ratio = chunk_max_log_size - prev_chunk_max_log_size;
        #[cfg(not(feature = "parallel"))]
        let iter_states = next_layer_state_slice.iter_mut();
        #[cfg(feature = "parallel")]
        let iter_states = next_layer_state_slice.par_iter_mut();

        byte_count += ((columns.len() - last_chunk_index) * N_BYTES_FELT) as u64;
        iter_states.enumerate().for_each(|(i, state)| {
            let prev_state = std::array::from_fn(|j| {
                let prev_state_limb = prev_layer_states[i >> log_ratio][j];
                to_lifted_simd(prev_state_limb, log_ratio, i)
            });
            let mut msgs: [u32x16; N_FELTS_IN_BLAKE_MESSAGE] = unsafe { std::mem::zeroed() };
            for (j, column) in columns[last_chunk_index..].iter().enumerate() {
                let log_size = column.data.len().ilog2();
                let log_ratio = chunk_max_log_size - log_size;
                msgs[j] = to_lifted_simd(column.data[i >> log_ratio].into_simd(), log_ratio, i);
            }
            *state = compress_finalize(prev_state, msgs, byte_count);
        });

        // let additional_lifting_ratio = (lifting_log_size - LOG_N_LANES) - max_log_size;
        let lifting_log_size_packed = lifting_log_size - LOG_N_LANES;
        // Prepare the output buffer.
        // TODO(Leo): ideally, we wouldn't need to write to a new buffer and instead we could
        // transmute `next_layer_states`, but there are alignment issues. Think about how to avoid
        // this copy.
        // Safety: we never read from `res`, only write to it.
        let mut res =
            unsafe { uninit_vec(1 << (lifting_log_size_packed + LOG_N_HASHES_PER_SIMD_STATE)) };

        // Lift the next_layer_states if needed.
        let mut trasposed_states = if lifting_log_size_packed == max_log_size {
            next_layer_states
        } else {
            let mut buf: Vec<[u32x16; N_FELTS_IN_BLAKE_STATE]> =
                unsafe { uninit_vec(1 << lifting_log_size_packed) };
            let log_ratio = lifting_log_size_packed - max_log_size;

            #[cfg(not(feature = "parallel"))]
            let iter = buf.iter_mut();
            #[cfg(feature = "parallel")]
            let iter = buf.par_iter_mut();

            iter.enumerate().for_each(|(i, dest)| {
                let packed_before_lift: [u32x16; N_FELTS_IN_BLAKE_STATE] =
                    next_layer_states[i >> log_ratio];
                let packed_after_lift =
                    std::array::from_fn(|j| to_lifted_simd(packed_before_lift[j], log_ratio, i));
                *dest = packed_after_lift;
            });
            buf
        };

        // Untranspose the states and reduce modulo M31 if `IS_M31_OUTPUT == true`.
        #[cfg(not(feature = "parallel"))]
        let iter_states = trasposed_states
            .iter_mut()
            .zip(res.chunks_mut(1 << LOG_N_HASHES_PER_SIMD_STATE));
        #[cfg(feature = "parallel")]
        let iter_states = trasposed_states
            .par_iter_mut()
            .zip(res.par_chunks_exact_mut(1 << LOG_N_HASHES_PER_SIMD_STATE));

        iter_states.for_each(|(state, dst)| {
            let untransposed = if IS_M31_OUTPUT {
                let tmp = untranspose_states(*state);
                std::array::from_fn(|i| reduce_to_m31_simd(tmp[i]))
            } else {
                untranspose_states(*state)
            };
            let dst: &mut [Blake2sHash; 16] = dst.try_into().unwrap();
            *dst = unsafe { transmute::<[u32x16; 8], [Blake2sHash; 16]>(untransposed) };
        });

        res
    }

    #[allow(clippy::uninit_vec)]
    fn build_next_layer(prev_layer: &Vec<Blake2sHash>) -> Vec<Blake2sHash> {
        // The log size of the current layer that needs to be built.
        let log_size: u32 = prev_layer.len().ilog2() - 1;
        if log_size < LOG_N_LANES {
            return parallel_iter!(0..1 << log_size)
                .map(|i| {
                    Blake2sMerkleHasherGeneric::<IS_M31_OUTPUT>::hash_children((
                        prev_layer[2 * i],
                        prev_layer[2 * i + 1],
                    ))
                })
                .collect();
        }
        // Safety: no index in `res` is ever read without having been written to
        // before.
        let mut res: Vec<Blake2sHash> = unsafe { uninit_vec(1 << log_size) };

        #[cfg(not(feature = "parallel"))]
        let iter = res.chunks_mut(1 << LOG_N_LANES);
        #[cfg(feature = "parallel")]
        let iter = res.par_chunks_mut(1 << LOG_N_LANES);

        iter.enumerate().for_each(|(i, dst)| {
            let state = INITIAL_STATE;
            let prev_chunk_u32s = cast_slice::<_, u32>(&prev_layer[(i << 5)..((i + 1) << 5)]);
            let msgs: [u32x16; N_FELTS_IN_BLAKE_MESSAGE] = array::from_fn(|j| {
                u32x16::from_array(std::array::from_fn(|k| prev_chunk_u32s[16 * j + k]))
            });
            let state = compress_finalize(state, transpose_msgs(msgs), N_BYTES_IN_BLAKE_MESSAGE);
            let mut untransposed = untranspose_states(state);
            if IS_M31_OUTPUT {
                untransposed = std::array::from_fn(|i| reduce_to_m31_simd(untransposed[i]));
            }
            let dst: &mut [Blake2sHash; 16] = dst.try_into().unwrap();
            *dst = unsafe { transmute::<[u32x16; 8], [Blake2sHash; 16]>(untransposed) };
        });
        res
    }
}

impl PackLeavesOps for SimdBackend {
    fn pack_leaves_input(
        values: &[&Col<SimdBackend, BaseField>; SECURE_EXTENSION_DEGREE],
    ) -> [Col<SimdBackend, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] {
        let input_len = values[0].len();
        assert!(values.iter().all(|c| c.len() == input_len));
        assert!(input_len.is_multiple_of(PACKED_LEAF_SIZE));
        let output_len = input_len / PACKED_LEAF_SIZE;
        let output_packed_len = output_len.div_ceil(N_LANES);

        let mut packed_simd: [Vec<PackedBaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] =
            unsafe { core::array::from_fn(|_| uninit_vec(output_packed_len)) };

        let output_packed_len_floor = output_len / N_LANES;

        // TODO(Leo): parallelize.
        for row in 0..output_packed_len_floor {
            let packed_start_idx = row * PACKED_LEAF_SIZE;
            let packed_values = core::array::from_fn(|j| {
                core::array::from_fn(|i| values[i].data[packed_start_idx + j])
            });
            let packed_row = transpose_packed_leaf(packed_values);
            for (offset, packed_leaf_column) in packed_row.into_iter().enumerate() {
                for coord in 0..SECURE_EXTENSION_DEGREE {
                    packed_simd[coord + offset * SECURE_EXTENSION_DEGREE][row] =
                        packed_leaf_column[coord];
                }
            }
        }

        // Transpose the tail. If `tail_rows > 0` then necessarily we haven't entered the previous
        // loop.
        let tail_rows = output_len % N_LANES;
        if tail_rows > 0 {
            // The last `N_LANES - tail_rows` rows are zeros. Note that this padding is effectively
            // ignored by the Merkle prover because we return an array of `BaseColumns` with length
            // = `output_len`.
            let mut tail_columns: [[BaseField; N_LANES];
                SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] =
                core::array::from_fn(|_| [BaseField::zero(); N_LANES]);
            for row in 0..tail_rows {
                // The index in the input vector corresponding to `row`.
                let source_row_start = (output_packed_len_floor * N_LANES + row) * PACKED_LEAF_SIZE;
                for offset in 0..PACKED_LEAF_SIZE {
                    let coords: [BaseField; 4] =
                        core::array::from_fn(|i| values[i].at(source_row_start + offset));
                    for coord in 0..SECURE_EXTENSION_DEGREE {
                        tail_columns[coord + offset * SECURE_EXTENSION_DEGREE][row] = coords[coord];
                    }
                }
            }
            for column_idx in 0..SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE {
                *packed_simd[column_idx].last_mut().unwrap() =
                    PackedBaseField::from_array(tail_columns[column_idx]);
            }
        }

        packed_simd.map(|data| BaseColumn {
            data,
            length: output_len,
        })
    }
}
/// Given a vector of columns sorted by size (in ascending order) and an index `last_chunk_index`
/// which is a multiple of N_FELTS_IN_BLAKE_MESSAGE, returns a vector of indices `0 = i₁ < i₂ < ...
/// < iₙ = last_chunk_index` (if `last_chunk_index = 0` then n = 1 and i₁ = 0) such that:
/// * All indices are multiples of N_FELTS_IN_BLAKE_MESSAGE.
/// * For all 1 <= k < n:
///     1. the sizes in `col_sizes[iₖ + N_FELTS_IN_BLAKE_MESSAGE..iₖ₊₁]` are all equal.
///     2. `col_sizes[iₖ] < col_sizes[iₖ₊₁]`.
fn get_lifting_indices(
    col_sizes: impl Iterator<Item = usize>,
    last_chunk_index: usize,
) -> Vec<usize> {
    let mut prev_size = 0;
    let mut res = vec![];
    for (idx, col_size) in col_sizes
        .enumerate()
        .step_by(N_FELTS_IN_BLAKE_MESSAGE)
        .skip(1)
    {
        if col_size > prev_size {
            res.push(idx - N_FELTS_IN_BLAKE_MESSAGE);
            prev_size = col_size;
        }
    }
    res.push(last_chunk_index);
    // Sanity check that there are no duplicates.
    debug_assert!(res.iter().duplicates().next().is_none());
    res
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;

    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::vcs_lifted::blake2_merkle::{Blake2sMerkleHasher, Blake2sMerkleHasherGeneric};
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
    use crate::prover::vcs_lifted::prover::MerkleProverLifted;

    #[test]
    fn test_build_next_layer() {
        const LOG_SIZE: u32 = 6;
        let layer: Vec<Blake2sHash> = (0u32..1 << (LOG_SIZE + 1))
            .map(|i| Blake2sHasher::hash(&i.to_le_bytes()))
            .collect();
        assert_eq!(
            <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_next_layer(&layer),
            <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_next_layer(&layer)
        );
    }

    fn prepare_blake_merkle_commit<const IS_M31_OUTPUT: bool>() -> (Blake2sHash, Blake2sHash) {
        const MAX_LOG_N_ROWS: u32 = 9;
        const N_COLS: u32 = 100;
        let mut cols: Vec<Vec<BaseField>> = (0..N_COLS)
            .map(|i| {
                (0..1 << MAX_LOG_N_ROWS)
                    .map(|j| M31::from(100 * i + j))
                    .collect_vec()
            })
            .collect();

        // Make the first two columns smaller to test a non-uniform sized trace.
        cols[0] = (0..1 << (MAX_LOG_N_ROWS - 4))
            .map(M31::from_u32_unchecked)
            .collect_vec();
        cols[1] = (0..1 << (MAX_LOG_N_ROWS - 3))
            .map(M31::from_u32_unchecked)
            .collect_vec();
        let cols_simd: Vec<BaseColumn> = cols.iter().map(|c| BaseColumn::from_cpu(c)).collect();

        (
            MerkleProverLifted::<CpuBackend, Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>::commit(
                cols.iter().collect(),
                MAX_LOG_N_ROWS,
                0,
            )
            .root(),
            MerkleProverLifted::<SimdBackend, Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>::commit(
                cols_simd.iter().collect(),
                MAX_LOG_N_ROWS,
                0,
            )
            .root(),
        )
    }

    #[test]
    fn test_blake_merkle_commit() {
        let (cpu_root, simd_root) = prepare_blake_merkle_commit::<false>();
        assert_eq!(cpu_root, simd_root);
    }

    #[test]
    fn test_blake_merkle_m31_commit() {
        let (cpu_root, simd_root) = prepare_blake_merkle_commit::<true>();
        assert_eq!(cpu_root, simd_root);
    }

    #[test]
    fn test_merkle_commit_small_column() {
        for log_size in 1..8 {
            let col = BaseColumn::from_cpu(&(0..1 << log_size).map(M31::from).collect_vec());

            assert_eq!(
                <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
                    &[&col.clone().to_cpu()],
                    log_size
                ),
                <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
                    &[&col],
                    log_size
                )
            );
        }
    }
}
