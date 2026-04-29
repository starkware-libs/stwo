//! SIMD-parallel Keccak256 sponge and lifted-Merkle `build_leaves`.

use std::simd::num::SimdUint;
use std::simd::{simd_swizzle, u32x8, u64x8};

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::keccak256_permutation::{keccak_f1600x8, N_LANES_KECCAK, PLEN};
use super::m31::{LOG_N_LANES, N_LANES};
use super::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::utils::uninit_vec;
use crate::core::vcs::keccak256_hash::{Keccak256Hash, Keccak256Hasher};
use crate::core::vcs_lifted::keccak256_merkle::Keccak256MerkleHasher;
use crate::prover::backend::{Col, Column, CpuBackend};
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

/// Keccak256 rate in u64 lanes (136 bytes / 8).
pub(super) const RATE_LANES: usize = 17;

/// `log2(N_LANES_KECCAK)`. The SIMD sponge processes `1 << LOG_N_LANES_KECCAK = 8` leaves at a
/// time.
const LOG_N_LANES_KECCAK: u32 = 3;

/// Initial Keccak sponge state — all zeros, since Keccak has no IV.
const INITIAL_STATE: [u64x8; PLEN] = [u64x8::from_array([0; N_LANES_KECCAK]); PLEN];

/// Absorb one M31 per parallel lane into `state`. Each lane's value is BE-encoded into 4
/// bytes and XORed at `block_byte_offset` of the lane's rate region. Returns the new
/// `block_byte_offset` (wrapping to `0` when the rate fills, in which case
/// `keccak_f1600x8` was applied to `state`).
fn absorb_m31_lanes(
    state: &mut [u64x8; PLEN],
    block_byte_offset: usize,
    lane_values: u32x8,
) -> usize {
    // Invariant: block_byte_offset advances by 4 each call and the rate (136 bytes) is
    // divisible by 4 — so the 4-byte chunk never crosses a u64 lane boundary.
    debug_assert!(block_byte_offset < RATE_LANES * 8);
    debug_assert!(block_byte_offset.is_multiple_of(4));

    let lane_idx = block_byte_offset / 8;
    let shift = u64x8::splat(((block_byte_offset % 8) * 8) as u64);

    // Byte-swap each u32 lane so its BE byte 0 (MSB) sits at LSB, zero-extend to u64,
    // then shift into position within the lane's u64.
    let widened: u64x8 = lane_values.swap_bytes().cast();
    state[lane_idx] ^= widened << shift;

    let new_off = block_byte_offset + 4;
    if new_off == RATE_LANES * 8 {
        keccak_f1600x8(state);
        0
    } else {
        new_off
    }
}

/// Pad (Keccak: `0x01 ... 0x80`), permute, and squeeze 32 bytes per parallel lane.
fn finalize_state(
    mut state: [u64x8; PLEN],
    block_byte_offset: usize,
) -> [[u8; 32]; N_LANES_KECCAK] {
    // Pad with 0x01 at `block_byte_offset`. `block_byte_offset == RATE_LANES * 8` can't
    // happen here because `absorb_m31_lanes` permutes-and-resets when full.
    debug_assert!(block_byte_offset < RATE_LANES * 8);
    let pad_lane = block_byte_offset / 8;
    let pad_shift = ((block_byte_offset % 8) * 8) as u64;
    state[pad_lane] ^= u64x8::splat(0x01u64 << pad_shift);

    // Pad with 0x80 at byte 135 (last byte of the rate region: lane 16, byte 7). If
    // `block_byte_offset == 135` (impossible here since it's a multiple of 4) the two
    // pad bits would land on the same byte and combine to 0x81; we don't need to handle
    // that.
    state[RATE_LANES - 1] ^= u64x8::splat(0x80u64 << 56);

    keccak_f1600x8(&mut state);

    // Squeeze first 32 bytes of each lane: state lanes 0..4, little-endian.
    let mut out = [[0u8; 32]; N_LANES_KECCAK];
    for (j, limb) in state[..4].iter().enumerate() {
        let words = limb.as_array();
        for (i, dst) in out.iter_mut().enumerate() {
            dst[j * 8..(j + 1) * 8].copy_from_slice(&words[i].to_le_bytes());
        }
    }
    out
}

impl MerkleOpsLifted<Keccak256MerkleHasher> for SimdBackend {
    /// SIMD `build_leaves` for Keccak256.
    ///
    /// For each leaf row `r`, hashes the M31 values `columns[*][r]` in column order,
    /// then squeezes 32 bytes — matching the CPU reference byte-for-byte. Lifts the
    /// per-leaf sponge state between size groups using the same per-leaf mapping as
    /// the CPU/Blake2s lifted paths.
    ///
    /// Assumes `columns` is sorted in ascending order of length. Falls back to the
    /// CPU implementation when the smallest column has fewer than `N_LANES` elements
    /// (the SIMD sponge layout requires at least one full `PackedM31` per column).
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Keccak256Hash> {
        if columns.is_empty() {
            return vec![Keccak256Hasher::default().finalize()];
        }

        if columns.first().unwrap().len() < N_LANES {
            let cpu_cols = columns.iter().map(|c| c.to_cpu()).collect_vec();
            return <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasher>>::build_leaves(
                &cpu_cols.iter().collect_vec(),
                lifting_log_size,
            );
        }

        let max_log_size_m31 = columns.last().unwrap().len().ilog2();
        let max_log_n_sponges = max_log_size_m31 - LOG_N_LANES_KECCAK;
        let lifting_log_n_sponges = lifting_log_size - LOG_N_LANES_KECCAK;
        let buf_log_size = max_log_n_sponges.max(lifting_log_n_sponges);

        // Two pre-allocated buffers, swapped between groups (same pattern as `blake2s_lifted`).
        // Safety: each entry is written before being read; we track the valid prefix via
        // `prev_log_n_sponges` and `lift_state` only indexes within that prefix.
        let mut prev_layer: Vec<[u64x8; PLEN]> = unsafe { uninit_vec(1 << buf_log_size) };
        let mut next_layer: Vec<[u64x8; PLEN]> = unsafe { uninit_vec(1 << buf_log_size) };

        // The first group lifts every child from `prev_layer[0]`, so seed that one entry
        // to the zero state — the lift then propagates zero state to every child.
        prev_layer[0] = INITIAL_STATE;
        let mut prev_log_n_sponges: u32 = 0;

        // All parallel sponges absorb in lockstep, so the byte offset is shared.
        let mut block_byte_offset: usize = 0;

        let mut col_idx = 0;
        while col_idx < columns.len() {
            let log_size_m31 = columns[col_idx].len().ilog2();
            debug_assert!(log_size_m31 >= LOG_N_LANES);
            let mut group_end = col_idx + 1;
            while group_end < columns.len() && columns[group_end].len().ilog2() == log_size_m31 {
                group_end += 1;
            }

            let log_n_sponges = log_size_m31 - LOG_N_LANES_KECCAK;
            let log_ratio = log_n_sponges - prev_log_n_sponges;
            let group_cols = &columns[col_idx..group_end];

            let prev_slice = &prev_layer[..1usize << prev_log_n_sponges];
            let next_slice = &mut next_layer[..1usize << log_n_sponges];

            // Each parallel task handles one `PackedM31` row's worth of leaves: a pair of
            // sponges (low half = lanes 0..8, high half = lanes 8..16). This lets us absorb
            // both halves of every column without any per-iteration `if half == 0` branch.
            #[cfg(not(feature = "parallel"))]
            let iter = next_slice.chunks_exact_mut(2);
            #[cfg(feature = "parallel")]
            let iter = next_slice.par_chunks_exact_mut(2);

            iter.enumerate().for_each(|(packed_row, pair)| {
                let [low_state, high_state] = <&mut [_; 2]>::try_from(pair).unwrap();
                *low_state = lift_state(prev_slice, 2 * packed_row, log_ratio);
                *high_state = lift_state(prev_slice, 2 * packed_row + 1, log_ratio);

                let mut offset = block_byte_offset;
                for col in group_cols {
                    let packed = col.data[packed_row].into_simd();
                    let low_lanes: u32x8 = simd_swizzle!(packed, [0, 1, 2, 3, 4, 5, 6, 7]);
                    let high_lanes: u32x8 = simd_swizzle!(packed, [8, 9, 10, 11, 12, 13, 14, 15]);
                    let new_offset = absorb_m31_lanes(low_state, offset, low_lanes);
                    absorb_m31_lanes(high_state, offset, high_lanes);
                    offset = new_offset;
                }
            });

            // All sponges absorbed the same byte count in lockstep; advance the shared offset.
            block_byte_offset = (block_byte_offset + 4 * group_cols.len()) % (RATE_LANES * 8);

            std::mem::swap(&mut prev_layer, &mut next_layer);
            prev_log_n_sponges = log_n_sponges;
            col_idx = group_end;
        }

        // Final lift to `lifting_log_size` if requested.
        if lifting_log_n_sponges > prev_log_n_sponges {
            let log_ratio = lifting_log_n_sponges - prev_log_n_sponges;
            let prev_slice = &prev_layer[..1usize << prev_log_n_sponges];
            let next_slice = &mut next_layer[..1usize << lifting_log_n_sponges];

            #[cfg(not(feature = "parallel"))]
            let iter = next_slice.iter_mut();
            #[cfg(feature = "parallel")]
            let iter = next_slice.par_iter_mut();

            iter.enumerate().for_each(|(i, state)| {
                *state = lift_state(prev_slice, i, log_ratio);
            });

            std::mem::swap(&mut prev_layer, &mut next_layer);
            prev_log_n_sponges = lifting_log_n_sponges;
        }

        // Truncate the buffer to its valid prefix and finalize.
        prev_layer.truncate(1usize << prev_log_n_sponges);
        finalize_all(prev_layer, block_byte_offset)
    }

    fn build_next_layer(prev_layer: &Vec<Keccak256Hash>) -> Vec<Keccak256Hash> {
        <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasher>>::build_next_layer(prev_layer)
    }
}

/// Lifts a single child state from its parent, mirroring the CPU per-leaf formula
/// `prev_layer[(idx >> (log_ratio + 1) << 1) + (idx & 1)]`. Each child state derives
/// its 8 lanes from a single parent state (verified by debug-asserts), permuted
/// according to the per-leaf mapping.
fn lift_state(parent: &[[u64x8; PLEN]], j: usize, log_ratio: u32) -> [u64x8; PLEN] {
    if log_ratio == 0 {
        return parent[j];
    }

    let parent_leaf_of = |k: usize| {
        let child_leaf = j * N_LANES_KECCAK + k;
        (child_leaf >> (log_ratio + 1) << 1) + (child_leaf & 1)
    };

    let parent_sponge_idx = parent_leaf_of(0) / N_LANES_KECCAK;
    let lane_indices: [usize; N_LANES_KECCAK] = std::array::from_fn(|k| {
        let pl = parent_leaf_of(k);
        debug_assert_eq!(
            pl / N_LANES_KECCAK,
            parent_sponge_idx,
            "lifted lanes must share a parent sponge"
        );
        pl % N_LANES_KECCAK
    });

    let parent_state = &parent[parent_sponge_idx];
    std::array::from_fn(|s| {
        let arr = parent_state[s].as_array();
        u64x8::from_array(std::array::from_fn(|k| arr[lane_indices[k]]))
    })
}

fn finalize_all(states: Vec<[u64x8; PLEN]>, block_byte_offset: usize) -> Vec<Keccak256Hash> {
    let n_total = states.len() * N_LANES_KECCAK;
    // Safety: every chunk is fully written by the for_each below.
    let mut leaves: Vec<Keccak256Hash> = unsafe { uninit_vec(n_total) };

    #[cfg(not(feature = "parallel"))]
    let iter = states
        .into_iter()
        .zip(leaves.chunks_exact_mut(N_LANES_KECCAK));
    #[cfg(feature = "parallel")]
    let iter = states
        .into_par_iter()
        .zip(leaves.par_chunks_exact_mut(N_LANES_KECCAK));

    iter.for_each(|(state, dst)| {
        let hashes = finalize_state(state, block_byte_offset);
        for (k, h) in hashes.into_iter().enumerate() {
            dst[k] = Keccak256Hash(h);
        }
    });

    leaves
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use sha3::{Digest, Keccak256};

    use super::*;
    use crate::core::fields::m31::M31;
    use crate::prover::backend::simd::column::BaseColumn;

    fn check_absorb_finalize(n: usize) {
        let inputs: Vec<[u32; N_LANES_KECCAK]> = (0..n)
            .map(|j| std::array::from_fn(|i| (i * 31 + j * 17 + 1) as u32))
            .collect();
        let mut state = INITIAL_STATE;
        let mut offset = 0;
        for row in &inputs {
            offset = absorb_m31_lanes(&mut state, offset, u32x8::from_array(*row));
        }
        let hashes = finalize_state(state, offset);
        for i in 0..N_LANES_KECCAK {
            let mut h = Keccak256::new();
            for row in &inputs {
                h.update(row[i].to_be_bytes());
            }
            let expected: [u8; 32] = h.finalize().into();
            assert_eq!(hashes[i], expected, "lane {i}, n = {n}");
        }
    }

    #[test]
    fn absorb_then_finalize_matches_sha3_for_small_input() {
        check_absorb_finalize(5);
    }

    #[test]
    fn absorb_finalize_matches_sha3_empty() {
        check_absorb_finalize(0);
    }

    #[test]
    fn absorb_finalize_matches_sha3_just_before_block_boundary() {
        // 33 m31s = 132 bytes, just below the 136-byte rate.
        check_absorb_finalize(33);
    }

    #[test]
    fn absorb_finalize_matches_sha3_at_block_boundary() {
        // 34 m31s = 136 bytes = exactly one block; finalize starts a fresh padded second block.
        check_absorb_finalize(34);
    }

    #[test]
    fn absorb_finalize_matches_sha3_just_after_block_boundary() {
        check_absorb_finalize(35);
    }

    #[test]
    fn absorb_finalize_matches_sha3_multiple_blocks() {
        check_absorb_finalize(100);
    }

    /// Builds CPU and SIMD column representations from per-column M31 vectors and
    /// asserts the two `build_leaves` outputs match.
    fn assert_simd_matches_cpu(cpu_cols: Vec<Vec<M31>>, lifting_log_size: u32) {
        let simd_cols: Vec<BaseColumn> = cpu_cols.iter().map(|c| BaseColumn::from_cpu(c)).collect();

        let cpu_leaves = <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasher>>::build_leaves(
            &cpu_cols.iter().collect_vec(),
            lifting_log_size,
        );
        let simd_leaves = <SimdBackend as MerkleOpsLifted<Keccak256MerkleHasher>>::build_leaves(
            &simd_cols.iter().collect_vec(),
            lifting_log_size,
        );

        assert_eq!(cpu_leaves, simd_leaves);
    }

    #[test]
    fn build_leaves_uniform_matches_cpu() {
        const LOG_SIZE: u32 = 7;
        const N_COLS: u32 = 50;
        let cpu_cols: Vec<Vec<M31>> = (0..N_COLS)
            .map(|c| {
                (0..1 << LOG_SIZE)
                    .map(|r| M31::from(c * 1000 + r))
                    .collect_vec()
            })
            .collect();
        assert_simd_matches_cpu(cpu_cols, LOG_SIZE);
    }

    #[test]
    fn build_leaves_mixed_size_matches_cpu() {
        // Two smaller columns (sizes 2^4 and 2^5) followed by a uniform tail at 2^7,
        // forcing two distinct lifts (1+2 ratio steps).
        const MAX_LOG_SIZE: u32 = 7;
        let mut cpu_cols: Vec<Vec<M31>> = Vec::new();
        cpu_cols.push((0..1 << 4).map(|r| M31::from(7 * r + 1)).collect_vec());
        cpu_cols.push((0..1 << 5).map(|r| M31::from(11 * r + 3)).collect_vec());
        for c in 0..50u32 {
            cpu_cols.push(
                (0..1 << MAX_LOG_SIZE)
                    .map(|r| M31::from(c * 1000 + r))
                    .collect_vec(),
            );
        }
        assert_simd_matches_cpu(cpu_cols, MAX_LOG_SIZE);
    }

    #[test]
    fn build_leaves_lifting_log_size_exceeds_max_matches_cpu() {
        // Forces a final lift after all groups are absorbed.
        const MAX_LOG_SIZE: u32 = 5;
        const LIFTING_LOG_SIZE: u32 = 8;
        let cpu_cols: Vec<Vec<M31>> = (0..20u32)
            .map(|c| {
                (0..1 << MAX_LOG_SIZE)
                    .map(|r| M31::from(c * 31 + r * 7 + 1))
                    .collect_vec()
            })
            .collect();
        assert_simd_matches_cpu(cpu_cols, LIFTING_LOG_SIZE);
    }

    #[test]
    fn build_leaves_small_column_fallback_matches_cpu() {
        // Smallest column < N_LANES (16) → SIMD path falls back to CPU.
        for log_size in 1..8 {
            let col: Vec<M31> = (0..1u32 << log_size).map(M31::from).collect_vec();
            let simd_col = BaseColumn::from_cpu(&col);

            let cpu_leaves = <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasher>>::build_leaves(
                &[&col],
                log_size,
            );
            let simd_leaves = <SimdBackend as MerkleOpsLifted<Keccak256MerkleHasher>>::build_leaves(
                &[&simd_col],
                log_size,
            );

            assert_eq!(cpu_leaves, simd_leaves, "log_size = {log_size}");
        }
    }
}
