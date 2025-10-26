//! A SIMD implementation of the BLAKE2s compression function.
//! Based on <https://github.com/oconnor663/blake2_simd/blob/master/blake2s/src/avx2.rs>.

use std::array;
use std::mem::transmute;
use std::simd::u32x16;

use bytemuck::cast_slice;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::m31::LOG_N_LANES;
use super::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{reduce_to_m31, Blake2sHash};
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::core::vcs::MerkleHasher;
use crate::parallel_iter;
use crate::prover::backend::{Col, Column, ColumnOps};
use crate::prover::vcs::ops::MerkleOps;

pub const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

pub const LEAF_INITIAL_STATE: [u32; 8] = [
    0x6510b1f7, 0xfd531f42, 0xcff75ec3, 0x382935d0, 0xab15dbf2, 0x950eb564, 0xe8e92866, 0x28047aca,
];

pub const NODE_INITIAL_STATE: [u32; 8] = [
    0xe5cf8926, 0x841cea30, 0x7b4acada, 0xfc5d8d28, 0xfc6ef857, 0xb29da528, 0xc0d319c7, 0x8ae795c8,
];

pub const SIMD_LEAF_INITIAL_STATE: [u32x16; 8] = [
    u32x16::splat(LEAF_INITIAL_STATE[0]),
    u32x16::splat(LEAF_INITIAL_STATE[1]),
    u32x16::splat(LEAF_INITIAL_STATE[2]),
    u32x16::splat(LEAF_INITIAL_STATE[3]),
    u32x16::splat(LEAF_INITIAL_STATE[4]),
    u32x16::splat(LEAF_INITIAL_STATE[5]),
    u32x16::splat(LEAF_INITIAL_STATE[6]),
    u32x16::splat(LEAF_INITIAL_STATE[7]),
];

pub const SIMD_NODE_INITIAL_STATE: [u32x16; 8] = [
    u32x16::splat(NODE_INITIAL_STATE[0]),
    u32x16::splat(NODE_INITIAL_STATE[1]),
    u32x16::splat(NODE_INITIAL_STATE[2]),
    u32x16::splat(NODE_INITIAL_STATE[3]),
    u32x16::splat(NODE_INITIAL_STATE[4]),
    u32x16::splat(NODE_INITIAL_STATE[5]),
    u32x16::splat(NODE_INITIAL_STATE[6]),
    u32x16::splat(NODE_INITIAL_STATE[7]),
];

pub const INITIAL_STATE: [u32x16; 8] = [
    // |Key| = 0x00, |HashLength| = 0x20.
    u32x16::splat(IV[0] ^ 0x01010020),
    u32x16::splat(IV[1]),
    u32x16::splat(IV[2]),
    u32x16::splat(IV[3]),
    u32x16::splat(IV[4]),
    u32x16::splat(IV[5]),
    u32x16::splat(IV[6]),
    u32x16::splat(IV[7]),
];

pub const SIGMA: [[u8; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

impl ColumnOps<Blake2sHash> for SimdBackend {
    type Column = Vec<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Blake2sHash> {
        if log_size < LOG_N_LANES {
            return parallel_iter!(0..1 << log_size)
                .map(|i| {
                    Blake2sMerkleHasher::hash_node(
                        prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                        &columns.iter().map(|column| column.at(i)).collect_vec(),
                    )
                })
                .collect();
        }

        if let Some(prev_layer) = prev_layer {
            assert_eq!(prev_layer.len(), 1 << (log_size + 1));
        }

        // Commit to columns.
        let mut res = vec![Blake2sHash::default(); 1 << log_size];
        #[cfg(not(feature = "parallel"))]
        let iter = res.chunks_mut(1 << LOG_N_LANES);

        #[cfg(feature = "parallel")]
        let iter = res.par_chunks_mut(1 << LOG_N_LANES);

        iter.enumerate().for_each(|(i, chunk)| {
            let mut state = if prev_layer.is_some() {
                SIMD_NODE_INITIAL_STATE
            } else {
                SIMD_LEAF_INITIAL_STATE
            };
            // No columns in the layer.
            if columns.is_empty() {
                let prev_layer = prev_layer.expect("Empty leafs are not supported");
                let prev_chunk_u32s = cast_slice::<_, u32>(&prev_layer[(i << 5)..((i + 1) << 5)]);
                let msgs: [u32x16; 16] = array::from_fn(|j| {
                    u32x16::from_array(std::array::from_fn(|k| prev_chunk_u32s[16 * j + k]))
                });
                let state = compress_finalize(state, transpose_msgs(msgs), 128);
                let state: [Blake2sHash; 16] = unsafe { transmute(untranspose_states(state)) };
                chunk.copy_from_slice(&state);
                return;
            }

            let mut t: u64 = 64;
            // Hash prev_layer, if exists.
            if let Some(prev_layer) = prev_layer {
                let prev_chunk_u32s = cast_slice::<_, u32>(&prev_layer[(i << 5)..((i + 1) << 5)]);
                t += 64;
                // Note: prev_layer might be unaligned.
                let msgs: [u32x16; 16] = array::from_fn(|j| {
                    u32x16::from_array(std::array::from_fn(|k| prev_chunk_u32s[16 * j + k]))
                });
                state = compress_unfinalized(state, transpose_msgs(msgs), t);
            }

            // Hash columns in chunks of 16.
            let mut col_chunk_iter = columns.chunks(16);
            let last_chunk = unsafe { col_chunk_iter.next_back().unwrap_unchecked() };
            for col_chunk in &mut col_chunk_iter {
                t += 64;
                let mut msgs: [u32x16; 16] = unsafe { std::mem::zeroed() };
                for (j, column) in col_chunk.iter().enumerate() {
                    msgs[j] = column.data[i].into_simd();
                }
                state = compress_unfinalized(state, msgs, t);
            }

            t += last_chunk.len() as u64 * 4;
            let mut last_block: [u32x16; 16] = unsafe { std::mem::zeroed() };
            for (j, column) in last_chunk.iter().enumerate() {
                last_block[j] = column.data[i].into_simd();
            }
            let state = compress_finalize(state, last_block, t);
            let state: [Blake2sHash; 16] = unsafe { transmute(untranspose_states(state)) };
            chunk.copy_from_slice(&state);
        });
        res
    }
}

impl MerkleOps<Blake2sM31MerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Blake2sHash> {
        let mut res = <SimdBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size, prev_layer, columns,
        );
        for x in res.iter_mut() {
            x.0 = reduce_to_m31(x.0);
        }
        res
    }
}

pub const ZEROS: u32x16 = u32x16::splat(0);

// `t` is the number of compressed bytes including the current block.
pub fn compress_unfinalized(state: [u32x16; 8], chunk: [u32x16; 16], t: u64) -> [u32x16; 8] {
    compress16(
        state,
        chunk,
        u32x16::splat(t as u32),
        // t >> 32 is 0 for our use case.
        ZEROS,
        ZEROS,
        ZEROS,
    )
}

pub fn compress_finalize(state: [u32x16; 8], last_block: [u32x16; 16], t: u64) -> [u32x16; 8] {
    compress16(
        state,
        last_block,
        u32x16::splat(t as u32),
        ZEROS,
        u32x16::splat(0xFFFFFFFF),
        ZEROS,
    )
}

/// Compresses and finalizes 16 instances of BLAKE2s.
///
/// # Arguments
///
/// * `msg_vecs` - 16 instances of BLAKE2s message (512 bits each).
/// * `bytes_in_msg` - Number of bytes in the message. NOTE: number of bytes above 64 is most likely
///   incorrect.
pub fn hash_16(msg_vecs: [u32x16; 16], bytes_in_msg: u64) -> [u32x16; 8] {
    compress_finalize(INITIAL_STATE, msg_vecs, bytes_in_msg)
}

/// Applies [`u32::rotate_right(N)`] to each element of the vector
///
/// [`u32::rotate_right(N)`]: u32::rotate_right
#[inline(always)]
fn rotate<const N: u32>(x: u32x16) -> u32x16 {
    (x >> N) | (x << (u32::BITS - N))
}

// `inline(always)` can cause code parsing errors for wasm: "locals exceed maximum".
#[cfg_attr(not(target_arch = "wasm32"), inline(always))]
pub fn round(v: &mut [u32x16; 16], m: [u32x16; 16], r: usize) {
    v[0] += m[SIGMA[r][0] as usize];
    v[1] += m[SIGMA[r][2] as usize];
    v[2] += m[SIGMA[r][4] as usize];
    v[3] += m[SIGMA[r][6] as usize];
    v[0] += v[4];
    v[1] += v[5];
    v[2] += v[6];
    v[3] += v[7];
    v[12] ^= v[0];
    v[13] ^= v[1];
    v[14] ^= v[2];
    v[15] ^= v[3];
    v[12] = rotate::<16>(v[12]);
    v[13] = rotate::<16>(v[13]);
    v[14] = rotate::<16>(v[14]);
    v[15] = rotate::<16>(v[15]);
    v[8] += v[12];
    v[9] += v[13];
    v[10] += v[14];
    v[11] += v[15];
    v[4] ^= v[8];
    v[5] ^= v[9];
    v[6] ^= v[10];
    v[7] ^= v[11];
    v[4] = rotate::<12>(v[4]);
    v[5] = rotate::<12>(v[5]);
    v[6] = rotate::<12>(v[6]);
    v[7] = rotate::<12>(v[7]);
    v[0] += m[SIGMA[r][1] as usize];
    v[1] += m[SIGMA[r][3] as usize];
    v[2] += m[SIGMA[r][5] as usize];
    v[3] += m[SIGMA[r][7] as usize];
    v[0] += v[4];
    v[1] += v[5];
    v[2] += v[6];
    v[3] += v[7];
    v[12] ^= v[0];
    v[13] ^= v[1];
    v[14] ^= v[2];
    v[15] ^= v[3];
    v[12] = rotate::<8>(v[12]);
    v[13] = rotate::<8>(v[13]);
    v[14] = rotate::<8>(v[14]);
    v[15] = rotate::<8>(v[15]);
    v[8] += v[12];
    v[9] += v[13];
    v[10] += v[14];
    v[11] += v[15];
    v[4] ^= v[8];
    v[5] ^= v[9];
    v[6] ^= v[10];
    v[7] ^= v[11];
    v[4] = rotate::<7>(v[4]);
    v[5] = rotate::<7>(v[5]);
    v[6] = rotate::<7>(v[6]);
    v[7] = rotate::<7>(v[7]);

    v[0] += m[SIGMA[r][8] as usize];
    v[1] += m[SIGMA[r][10] as usize];
    v[2] += m[SIGMA[r][12] as usize];
    v[3] += m[SIGMA[r][14] as usize];
    v[0] += v[5];
    v[1] += v[6];
    v[2] += v[7];
    v[3] += v[4];
    v[15] ^= v[0];
    v[12] ^= v[1];
    v[13] ^= v[2];
    v[14] ^= v[3];
    v[15] = rotate::<16>(v[15]);
    v[12] = rotate::<16>(v[12]);
    v[13] = rotate::<16>(v[13]);
    v[14] = rotate::<16>(v[14]);
    v[10] += v[15];
    v[11] += v[12];
    v[8] += v[13];
    v[9] += v[14];
    v[5] ^= v[10];
    v[6] ^= v[11];
    v[7] ^= v[8];
    v[4] ^= v[9];
    v[5] = rotate::<12>(v[5]);
    v[6] = rotate::<12>(v[6]);
    v[7] = rotate::<12>(v[7]);
    v[4] = rotate::<12>(v[4]);
    v[0] += m[SIGMA[r][9] as usize];
    v[1] += m[SIGMA[r][11] as usize];
    v[2] += m[SIGMA[r][13] as usize];
    v[3] += m[SIGMA[r][15] as usize];
    v[0] += v[5];
    v[1] += v[6];
    v[2] += v[7];
    v[3] += v[4];
    v[15] ^= v[0];
    v[12] ^= v[1];
    v[13] ^= v[2];
    v[14] ^= v[3];
    v[15] = rotate::<8>(v[15]);
    v[12] = rotate::<8>(v[12]);
    v[13] = rotate::<8>(v[13]);
    v[14] = rotate::<8>(v[14]);
    v[10] += v[15];
    v[11] += v[12];
    v[8] += v[13];
    v[9] += v[14];
    v[5] ^= v[10];
    v[6] ^= v[11];
    v[7] ^= v[8];
    v[4] ^= v[9];
    v[5] = rotate::<7>(v[5]);
    v[6] = rotate::<7>(v[6]);
    v[7] = rotate::<7>(v[7]);
    v[4] = rotate::<7>(v[4]);
}

/// Transposes input chunks (16 chunks of 16 `u32`s each), to get 16 `u32x16`, each
/// representing 16 packed instances of a message word.
pub fn transpose_msgs(mut data: [u32x16; 16]) -> [u32x16; 16] {
    // Index abcd:xyzw, refers to a specific word in data as follows:
    //   abcd - chunk index (in base 2)
    //   xyzw - word offset (in base 2)
    // Transpose by applying 4 times the index permutation:
    //   abcd:xyzw => wabc:dxyz
    // In other words, rotate the index to the right by 1.
    for _ in 0..4 {
        let (d0, d8) = data[0].deinterleave(data[1]);
        let (d1, d9) = data[2].deinterleave(data[3]);
        let (d2, d10) = data[4].deinterleave(data[5]);
        let (d3, d11) = data[6].deinterleave(data[7]);
        let (d4, d12) = data[8].deinterleave(data[9]);
        let (d5, d13) = data[10].deinterleave(data[11]);
        let (d6, d14) = data[12].deinterleave(data[13]);
        let (d7, d15) = data[14].deinterleave(data[15]);
        data = [
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15,
        ];
    }

    data
}

pub fn untranspose_states(mut states: [u32x16; 8]) -> [u32x16; 8] {
    // Index abc:xyzw, refers to a specific word in data as follows:
    //   abc - chunk index (in base 2)
    //   xyzw - word offset (in base 2)
    // Transpose by applying 3 times the index permutation:
    //   abc:xyzw => bcx:yzwa
    // In other words, rotate the index to the left by 1.
    for _ in 0..3 {
        let (d0, d1) = states[0].interleave(states[4]);
        let (d2, d3) = states[1].interleave(states[5]);
        let (d4, d5) = states[2].interleave(states[6]);
        let (d6, d7) = states[3].interleave(states[7]);
        states = [d0, d1, d2, d3, d4, d5, d6, d7];
    }
    states
}

/// Compresses 16 blake2s instances.
pub fn compress16(
    h_vecs: [u32x16; 8],
    msg_vecs: [u32x16; 16],
    count_low: u32x16,
    count_high: u32x16,
    lastblock: u32x16,
    lastnode: u32x16,
) -> [u32x16; 8] {
    let mut v = [
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        u32x16::splat(IV[0]),
        u32x16::splat(IV[1]),
        u32x16::splat(IV[2]),
        u32x16::splat(IV[3]),
        u32x16::splat(IV[4]) ^ count_low,
        u32x16::splat(IV[5]) ^ count_high,
        u32x16::splat(IV[6]) ^ lastblock,
        u32x16::splat(IV[7]) ^ lastnode,
    ];

    round(&mut v, msg_vecs, 0);
    round(&mut v, msg_vecs, 1);
    round(&mut v, msg_vecs, 2);
    round(&mut v, msg_vecs, 3);
    round(&mut v, msg_vecs, 4);
    round(&mut v, msg_vecs, 5);
    round(&mut v, msg_vecs, 6);
    round(&mut v, msg_vecs, 7);
    round(&mut v, msg_vecs, 8);
    round(&mut v, msg_vecs, 9);

    [
        h_vecs[0] ^ v[0] ^ v[8],
        h_vecs[1] ^ v[1] ^ v[9],
        h_vecs[2] ^ v[2] ^ v[10],
        h_vecs[3] ^ v[3] ^ v[11],
        h_vecs[4] ^ v[4] ^ v[12],
        h_vecs[5] ^ v[5] ^ v[13],
        h_vecs[6] ^ v[6] ^ v[14],
        h_vecs[7] ^ v[7] ^ v[15],
    ]
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::mem::transmute;
    use std::simd::u32x16;

    use aligned::{Aligned, A64};
    use bytemuck::cast_slice;
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{compress16, hash_16, transpose_msgs, untranspose_states};
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::prover::backend::simd::blake2s_ref::{self, compress};

    #[test]
    fn compress16_works() {
        let states: Aligned<A64, [[u32; 8]; 16]> =
            Aligned(array::from_fn(|i| array::from_fn(|j| (i + j) as u32)));
        let msgs: Aligned<A64, [[u32; 16]; 16]> =
            Aligned(array::from_fn(|i| array::from_fn(|j| (i + j + 20) as u32)));
        let count_low = 1;
        let count_high = 2;
        let lastblock = 3;
        let lastnode = 4;
        let res_unvectorized = array::from_fn(|i| {
            compress(
                states[i], msgs[i], count_low, count_high, lastblock, lastnode,
            )
        });

        let res_vectorized: [[u32; 8]; 16] = unsafe {
            transmute(untranspose_states(compress16(
                transpose_states(transmute::<Aligned<A64, [[u32; 8]; 16]>, [u32x16; 8]>(
                    states,
                )),
                transpose_msgs(transmute::<Aligned<A64, [[u32; 16]; 16]>, [u32x16; 16]>(
                    msgs,
                )),
                u32x16::splat(count_low),
                u32x16::splat(count_high),
                u32x16::splat(lastblock),
                u32x16::splat(lastnode),
            )))
        };

        assert_eq!(res_vectorized, res_unvectorized);
    }

    #[test]
    fn untranspose_states_is_transpose_states_inverse() {
        let states = array::from_fn(|i| u32x16::from(array::from_fn(|j| (i + j) as u32)));
        let transposed_states = transpose_states(states);

        let untrasponsed_transposed_states = untranspose_states(transposed_states);

        assert_eq!(untrasponsed_transposed_states, states)
    }

    #[test]
    fn hash_16_works() {
        let mut rng = SmallRng::seed_from_u64(1055);
        let msgs: [[u32; 16]; 16] = array::from_fn(|_| rng.gen::<[u32; 16]>());
        let expected: [[u8; 32]; 16] = array::from_fn(|i| {
            let state = msgs[i];
            let mut hasher = Blake2sHasher::new();
            hasher.update(cast_slice(&state));
            hasher.finalize().0
        });
        let transposed_msgs: [u32x16; 16] =
            array::from_fn(|i| u32x16::from_array(array::from_fn(|j| msgs[j][i])));

        let res = hash_16(transposed_msgs, 64);

        let res: [[u8; 32]; 16] = unsafe { transmute(untranspose_states(res)) };
        assert_eq!(res, expected);
    }

    /// Transposes states, from 8 packed words, to get 16 results, each of size 32B.
    fn transpose_states(mut states: [u32x16; 8]) -> [u32x16; 8] {
        // Index abc:xyzw, refers to a specific word in data as follows:
        //   abc - chunk index (in base 2)
        //   xyzw - word offset (in base 2)
        // Transpose by applying 3 times the index permutation:
        //   abc:xyzw => wab:cxyz
        // In other words, rotate the index to the right by 1.
        for _ in 0..3 {
            let (s0, s4) = states[0].deinterleave(states[1]);
            let (s1, s5) = states[2].deinterleave(states[3]);
            let (s2, s6) = states[4].deinterleave(states[5]);
            let (s3, s7) = states[6].deinterleave(states[7]);
            states = [s0, s1, s2, s3, s4, s5, s6, s7];
        }

        states
    }

    fn prefix_as_32(prefix: &[u8; 64]) -> [u32; 16] {
        prefix
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes: &[u8; 4]| u32::from_le_bytes(*bytes))
            .collect_vec()
            .try_into()
            .unwrap()
    }

    #[test]
    fn test_leaf_initial_state() {
        let prefix: [u32; 16] = prefix_as_32(&crate::core::vcs::blake2_merkle::LEAF_PREFIX);
        let mut state = blake2s_ref::IV;
        // |Key| = 0x00, |HashLength| = 0x20.
        state[0] ^= 0x01010020;
        let res = compress(state, prefix, 64, 0, 0, 0);

        assert_eq!(res, super::LEAF_INITIAL_STATE);
    }

    #[test]
    fn test_node_initial_state() {
        let prefix: [u32; 16] = prefix_as_32(&crate::core::vcs::blake2_merkle::NODE_PREFIX);
        let mut state = blake2s_ref::IV;
        // |Key| = 0x00, |HashLength| = 0x20.
        state[0] ^= 0x01010020;

        let res = compress(state, prefix, 64, 0, 0, 0);

        assert_eq!(res, super::NODE_INITIAL_STATE);
    }
}
