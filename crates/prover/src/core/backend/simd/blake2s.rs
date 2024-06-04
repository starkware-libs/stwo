//! A SIMD implementation of the BLAKE2s compression function.
//! Based on <https://github.com/oconnor663/blake2_simd/blob/master/blake2s/src/avx2.rs>.

use std::array;
use std::iter::repeat;
use std::mem::transmute;
use std::simd::u32x16;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::m31::{LOG_N_LANES, N_LANES};
use super::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const SIGMA: [[u8; 16]; 10] = [
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
        if log_size < N_LANES as u32 {
            #[cfg(not(feature = "parallel"))]
            let iter = 0..1 << log_size;

            #[cfg(feature = "parallel")]
            let iter = (0..1 << log_size).into_par_iter();

            return iter
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

        let zeros = u32x16::splat(0);

        // Commit to columns.
        let mut res = vec![Blake2sHash::default(); 1 << log_size];
        #[cfg(not(feature = "parallel"))]
        let iter = res.chunks_mut(1 << LOG_N_LANES);

        #[cfg(feature = "parallel")]
        let iter = res.par_chunks_mut(1 << LOG_N_LANES);

        iter.enumerate().for_each(|(i, chunk)| {
            let mut state: [u32x16; 8] = unsafe { std::mem::zeroed() };
            // Hash prev_layer, if exists.
            if let Some(prev_layer) = prev_layer {
                let ptr = prev_layer[(i << 5)..((i + 1) << 5)].as_ptr() as *const u32x16;
                let msgs: [u32x16; 16] = array::from_fn(|j| unsafe { *ptr.add(j) });
                state = compress16(state, transpose_msgs(msgs), zeros, zeros, zeros, zeros);
            }

            // Hash columns in chunks of 16.
            let mut col_chunk_iter = columns.array_chunks();
            for col_chunk in &mut col_chunk_iter {
                let msgs = col_chunk.map(|column| column.data[i].into_simd());
                state = compress16(state, msgs, zeros, zeros, zeros, zeros);
            }

            // Hash remaining columns.
            let remainder = col_chunk_iter.remainder();
            if !remainder.is_empty() {
                let msgs = remainder
                    .iter()
                    .map(|column| column.data[i].into_simd())
                    .chain(repeat(zeros))
                    .take(N_LANES)
                    .collect_vec()
                    .try_into()
                    .unwrap();
                state = compress16(state, msgs, zeros, zeros, zeros, zeros);
            }
            let state: [Blake2sHash; 16] = unsafe { transmute(untranspose_states(state)) };
            chunk.copy_from_slice(&state);
        });
        res
    }
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
fn round(v: &mut [u32x16; 16], m: [u32x16; 16], r: usize) {
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
fn transpose_msgs(mut data: [u32x16; 16]) -> [u32x16; 16] {
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

fn untranspose_states(mut states: [u32x16; 8]) -> [u32x16; 8] {
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
fn compress16(
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

    use super::{compress16, transpose_msgs, untranspose_states};
    use crate::core::vcs::blake2s_ref::compress;

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
                transpose_states(transmute(states)),
                transpose_msgs(transmute(msgs)),
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
}
