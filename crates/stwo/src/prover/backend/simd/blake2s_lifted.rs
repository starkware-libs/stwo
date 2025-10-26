use std::array;
use std::mem::transmute;
use std::simd::{simd_swizzle, u32x16};

use bytemuck::cast_slice;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::m31::LOG_N_LANES;
use super::SimdBackend;
use crate::core::fields::m31::{BaseField, N_BYTES_FELT};
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::parallel_iter;
use crate::prover::backend::simd::blake2s::{
    compress_finalize, compress_unfinalized, transpose_msgs, untranspose_states,
    SIMD_LEAF_INITIAL_STATE, SIMD_NODE_INITIAL_STATE, ZEROS,
};
use crate::prover::backend::{Col, Column};
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

const N_FELTS_IN_BLAKE_MESSAGE: usize = 16;
const N_FELTS_IN_BLAKE_STATE: usize = 8;
const N_BYTES_IN_BLAKE_MESSAGE: u64 = N_FELTS_IN_BLAKE_MESSAGE as u64 * N_BYTES_FELT as u64;
const N_BYTES_IN_PREFIX: u64 = 64;

impl MerkleOpsLifted<Blake2sMerkleHasher> for SimdBackend {
    /// See the docs of [`crate::prover::backend::cpu::blake2s_lifted`].
    ///
    /// Note that, in this function, all variables that track log sizes
    /// refer to the "size" in terms of PackedM31 (e.g. the log size of a column
    /// of 4 PackedM31 elements is 2).
    fn build_leaves(columns: &[&Col<Self, BaseField>]) -> Col<Self, Blake2sHash> {
        if columns.first().is_some_and(|c| c.len() < 1 << LOG_N_LANES) {
            unimplemented!("Support for small columns is not implemented.")
        }

        let max_log_size: u32 = columns.last().unwrap().data.len().ilog2();
        // Hash columns in chunks of 16.
        let mut col_chunk_iter = columns.chunks(N_FELTS_IN_BLAKE_MESSAGE);
        let last_chunk = unsafe { col_chunk_iter.next_back().unwrap_unchecked() };
        // Initialize the vector of Blake2s states. The state is of type `[u32x16; 8]`.
        //
        // We use two large buffers to hold the intermediate results of the computation.
        // In every iteration, a possibly larger chunk of the buffer is used. This
        // saves memory allocations.
        let mut prev_layer_states: Vec<[u32x16; N_FELTS_IN_BLAKE_STATE]> =
            vec![SIMD_LEAF_INITIAL_STATE; 1 << (max_log_size)];
        let mut next_layer_states: Vec<[u32x16; N_FELTS_IN_BLAKE_STATE]> =
            vec![[ZEROS; N_FELTS_IN_BLAKE_STATE]; 1 << (max_log_size)];

        // The actual log size of `prev_layer_states` is equal to `max_log_size`, but only the first
        // two entries are accessed for the computation of the first iteration.
        let mut prev_chunk_max_log_size = 1;
        for (chunk_idx, chunk_columns) in &mut col_chunk_iter.enumerate() {
            let chunk_max_log_size: u32 = chunk_columns.iter().last().unwrap().data.len().ilog2();
            let next_layer_state_slice = &mut next_layer_states[0..1 << chunk_max_log_size];
            // Compute the new states of the current layer.
            #[cfg(not(feature = "parallel"))]
            let iter_states = next_layer_state_slice.iter_mut();
            #[cfg(feature = "parallel")]
            let iter_states = next_layer_state_slice.par_iter_mut();

            iter_states.enumerate().for_each(|(i, curr_state)| {
                let log_ratio = chunk_max_log_size - prev_chunk_max_log_size;
                let prev_state = std::array::from_fn(|j| {
                    let prev_state_limb = prev_layer_states[i >> log_ratio][j];
                    to_lifted_simd(prev_state_limb, log_ratio, i)
                });

                // The first summand corresponds to the leaf prefix.
                // `chunk_idx` is incremented by 1 because it's zero-based.
                let byte_count =
                    N_BYTES_IN_PREFIX + (N_BYTES_IN_BLAKE_MESSAGE * (chunk_idx + 1) as u64);
                let mut msgs: [u32x16; N_FELTS_IN_BLAKE_MESSAGE] = unsafe { std::mem::zeroed() };
                for (j, column) in chunk_columns.iter().enumerate() {
                    let log_size = column.data.len().ilog2();
                    let log_ratio = chunk_max_log_size - log_size;
                    msgs[j] = to_lifted_simd(column.data[i >> log_ratio].into_simd(), log_ratio, i);
                }

                let state = compress_unfinalized(prev_state, msgs, byte_count);
                curr_state.copy_from_slice(&state);
            });
            std::mem::swap(&mut prev_layer_states, &mut next_layer_states);
            prev_chunk_max_log_size = chunk_max_log_size;
        }

        // Process last chunk.
        // TODO(Leo): can we avoid the code duplication with the iteration
        // on the chunks?
        #[cfg(not(feature = "parallel"))]
        let iter_states = next_layer_states.iter_mut();
        #[cfg(feature = "parallel")]
        let iter_states = next_layer_states.par_iter_mut();

        iter_states.enumerate().for_each(|(i, curr_state)| {
            let log_ratio = (max_log_size) - prev_chunk_max_log_size;
            let prev_state = std::array::from_fn(|j| {
                let prev_state_limb = prev_layer_states[i >> log_ratio][j];
                to_lifted_simd(prev_state_limb, log_ratio, i)
            });

            let byte_count = N_BYTES_IN_PREFIX + (N_BYTES_FELT as u64) * (columns.len() as u64);
            let mut msgs: [u32x16; N_FELTS_IN_BLAKE_MESSAGE] = unsafe { std::mem::zeroed() };
            for (j, column) in last_chunk.iter().enumerate() {
                let log_size = column.data.len().ilog2();
                let log_ratio = (max_log_size) - log_size;
                msgs[j] = to_lifted_simd(column.data[i >> log_ratio].into_simd(), log_ratio, i);
            }
            let state = compress_finalize(prev_state, msgs, byte_count);
            curr_state.copy_from_slice(&state);
        });

        next_layer_states
            .iter()
            .flat_map(|x| {
                let state: [Blake2sHash; 16] = unsafe { transmute(untranspose_states(*x)) };
                state
            })
            .collect_vec()
    }

    fn build_next_layer(prev_layer: &Vec<Blake2sHash>) -> Vec<Blake2sHash> {
        // The log size of the current layer that needs to be built.
        let log_size: u32 = prev_layer.len().ilog2() - 1;

        if log_size < LOG_N_LANES {
            return parallel_iter!(0..1 << log_size)
                .map(|i| {
                    Blake2sMerkleHasher::hash_children((prev_layer[2 * i], prev_layer[2 * i + 1]))
                })
                .collect();
        }

        // Commit to columns.
        let mut res = vec![Blake2sHash::default(); 1 << log_size];

        #[cfg(not(feature = "parallel"))]
        let iter = res.chunks_mut(1 << LOG_N_LANES);
        #[cfg(feature = "parallel")]
        let iter = res.par_chunks_mut(1 << LOG_N_LANES);

        iter.enumerate().for_each(|(i, chunk)| {
            let state = SIMD_NODE_INITIAL_STATE;
            let prev_chunk_u32s = cast_slice::<_, u32>(&prev_layer[(i << 5)..((i + 1) << 5)]);
            let msgs: [u32x16; N_FELTS_IN_BLAKE_MESSAGE] = array::from_fn(|j| {
                u32x16::from_array(std::array::from_fn(|k| prev_chunk_u32s[16 * j + k]))
            });
            let state = compress_finalize(
                state,
                transpose_msgs(msgs),
                N_BYTES_IN_PREFIX + N_BYTES_IN_BLAKE_MESSAGE,
            );
            let state: [Blake2sHash; 16] = unsafe { transmute(untranspose_states(state)) };
            chunk.copy_from_slice(&state);
        });
        res
    }
}

/// A helper function to compute the lift of a column of PackedM31 values.
///
/// # Intro
///
/// Given a column C of log_size n, containing u32x16 values, the goal is to compute
/// its "lifting" to log_size m (m >= n). Here, "lifting" means the following:
///
/// 1. Interpret column C as the vector of evaluations of a circle polynomial `p`, of degree < n, on
///    the canonical coset of log_size n, in bit reversed order.
///
/// 2. The lift of C to log_size m is, by definition, the vector of evaluations of the polynomial `p
///    ∘ πᵐ⁻ⁿ` on the canonical coset of log_size m, in bit reversed order. Here `π` is the doubling
///    map.
///
/// # Arguments
///
/// - `x`: the evaluation of the un-lifted polynomial that we wish to lift. Note: the function
///   **assumes** that `x` is the evaluation (of the un-lifted polynomial) which is needed to
///   compute the lifted polynomial at the `idx`-th point of the lifted domain.
/// - `log_ratio`: the log ratio between the lifted domain and the base domain (in the above
///   example, it's m - n).
/// - `idx`: the index in the vector of lifted evaluations that we wish to compute.
///
/// # Returns
///
/// - A PackedM31 corresponding to the values of the lifted polynomial on the `idx`-th, ..., `idx +
///   15`-th points of the lifted domain, where the order is the bit reversed order.
fn to_lifted_simd(x: u32x16, log_ratio: u32, idx: usize) -> u32x16 {
    let idx_mod_ratio = idx % (1 << log_ratio);
    match log_ratio {
        0 => x,
        1 => match idx_mod_ratio % 2 {
            0 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_1[0]),
            1 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_1[1]),
            _ => unreachable!(),
        },
        2 => match idx_mod_ratio % 4 {
            0 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_2[0]),
            1 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_2[1]),
            2 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_2[2]),
            3 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_2[3]),
            _ => unreachable!(),
        },
        _ => match idx_mod_ratio >> (log_ratio - 3) {
            0 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[0]),
            1 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[1]),
            2 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[2]),
            3 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[3]),
            4 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[4]),
            5 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[5]),
            6 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[6]),
            7 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[7]),
            _ => unreachable!(),
        },
    }
}

#[rustfmt::skip]
const LIFTING_SWIZZLES_LOG_RATIO_1: [[usize; 16]; 2] = [
    [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7],
    [8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15],
];
#[rustfmt::skip]
const LIFTING_SWIZZLES_LOG_RATIO_2: [[usize; 16]; 4] = [
    [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
    [4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7],
    [8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11],
    [12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15],
];
#[rustfmt::skip]
const LIFTING_SWIZZLES_LOG_RATIO_GREATER_2: [[usize; 16]; 8] = [
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
    [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5],
    [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
    [8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9],
    [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11],
    [12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13],
    [14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15],
];

#[cfg(test)]
mod tests {

    use itertools::Itertools;

    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::CpuBackend;
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
            SimdBackend::build_next_layer(&layer)
        );
    }

    #[test]
    fn test_merkle_commit() {
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
        let cols_simd: Vec<BaseColumn> = cols
            .iter()
            .map(|c| BaseColumn::from_cpu(c.clone()))
            .collect();

        assert_eq!(
            MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(cols.iter().collect())
                .root(),
            MerkleProverLifted::<SimdBackend, Blake2sMerkleHasher>::commit(
                cols_simd.iter().collect()
            )
            .root()
        );
    }
}
