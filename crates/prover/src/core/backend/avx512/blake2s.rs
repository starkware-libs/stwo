use std::arch::x86_64::{__m512i, _mm512_loadu_si512};

use itertools::Itertools;
use stwo_verifier::core::fields::m31::BaseField;

use super::blake2s_avx::{compress16, set1, transpose_msgs, untranspose_states};
use super::{AVX512Backend, VECS_LOG_SIZE};
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};

impl ColumnOps<Blake2sHash> for AVX512Backend {
    type Column = Vec<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for AVX512Backend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Col<AVX512Backend, BaseField>],
    ) -> Vec<Blake2sHash> {
        // Pad prev_layer if too small.
        if log_size < VECS_LOG_SIZE as u32 {
            return (0..(1 << log_size))
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
        let mut res = Vec::with_capacity(1 << log_size);
        for i in 0..(1 << (log_size - VECS_LOG_SIZE as u32)) {
            let mut state: [__m512i; 8] = unsafe { std::mem::zeroed() };
            // Hash prev_layer, if exists.
            if let Some(prev_layer) = prev_layer {
                let ptr = prev_layer[(i << 5)..((i + 1) << 5)].as_ptr() as *const __m512i;
                let msgs: [__m512i; 16] = std::array::from_fn(|j| unsafe {
                    _mm512_loadu_si512(ptr.add(j) as *const i32)
                });
                state = unsafe {
                    compress16(
                        state,
                        transpose_msgs(msgs),
                        set1(0),
                        set1(0),
                        set1(0),
                        set1(0),
                    )
                };
            }

            // Hash columns in chunks of 16.
            let mut col_chunk_iter = columns.array_chunks();
            for col_chunk in &mut col_chunk_iter {
                let msgs = col_chunk.map(|column| column.data[i].0);
                state = unsafe { compress16(state, msgs, set1(0), set1(0), set1(0), set1(0)) };
            }

            // Hash remaining columns.
            let remainder = col_chunk_iter.remainder();
            if !remainder.is_empty() {
                let msgs = remainder
                    .iter()
                    .map(|column| column.data[i].0)
                    .chain(std::iter::repeat(unsafe { set1(0) }))
                    .take(16)
                    .collect_vec()
                    .try_into()
                    .unwrap();
                state = unsafe { compress16(state, msgs, set1(0), set1(0), set1(0), set1(0)) };
            }
            let state: [Blake2sHash; 16] =
                unsafe { std::mem::transmute(untranspose_states(state)) };
            res.extend_from_slice(&state);
        }
        res
    }
}
