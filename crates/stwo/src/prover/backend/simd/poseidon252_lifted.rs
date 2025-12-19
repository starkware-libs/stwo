use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use starknet_ff::FieldElement as FieldElement252;

use crate::core::fields::m31::{BaseField, M31};
use crate::core::vcs::poseidon252_merkle::{construct_felt252_from_m31s, ELEMENTS_IN_BLOCK};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::poseidon252_merkle::{
    poseidon_finalize, poseidon_update, Poseidon252MerkleHasher, ELEMENTS_IN_BUFFER,
};
use crate::prover::backend::simd::m31::N_LANES;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Col, Column, CpuBackend};
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

/// TODO(Leo): the implementation below is not really vectorized because there is no poseidon hash
/// implementation in simd yet.
impl MerkleOpsLifted<Poseidon252MerkleHasher> for SimdBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Poseidon252MerkleHasher as MerkleHasherLifted>::Hash> {
        if columns.is_empty() {
            return vec![<Poseidon252MerkleHasher as MerkleHasherLifted>::Hash::default()];
        }
        if columns.first().unwrap().len() < N_LANES {
            let cpu_cols = columns.iter().map(|column| column.to_cpu()).collect_vec();
            return <CpuBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_leaves(
                &cpu_cols.iter().collect_vec(),
            );
        }
        let max_log_size: u32 = columns.last().unwrap().len().ilog2();
        let mut col_chunk_iter = columns.chunks(ELEMENTS_IN_BUFFER);
        let last_chunk = unsafe { col_chunk_iter.next_back().unwrap_unchecked() };

        // Preallocate working memory.
        // For every chunk of column, we go over all the rows, read from `prev_layer_states`, write
        // to `next_layer_states`, and then we swap them for the next chunk.
        let mut prev_layer_states: Vec<[FieldElement252; 3]> =
            vec![[FieldElement252::default(); 3]; 1 << (max_log_size)];
        let mut next_layer_states: Vec<[FieldElement252; 3]> =
            vec![[FieldElement252::default(); 3]; 1 << (max_log_size)];

        let mut prev_chunk_max_log_size = 1;
        for chunk_columns in &mut col_chunk_iter {
            let chunk_max_log_size: u32 = chunk_columns.iter().last().unwrap().len().ilog2();
            let next_layer_state_slice = &mut next_layer_states[0..1 << chunk_max_log_size];
            // Compute the new states of the current layer.
            #[cfg(not(feature = "parallel"))]
            let iter_states = next_layer_state_slice.iter_mut();
            #[cfg(feature = "parallel")]
            let iter_states = next_layer_state_slice.par_iter_mut();

            iter_states.enumerate().for_each(|(i, curr_state)| {
                let log_ratio = chunk_max_log_size - prev_chunk_max_log_size;
                let mut prev_state: [FieldElement252; 3] =
                    prev_layer_states[(i >> (log_ratio + 1) << 1) + (i & 1)];
                let mut msgs: [M31; ELEMENTS_IN_BUFFER] = unsafe { std::mem::zeroed() };
                for (j, column) in chunk_columns.iter().enumerate() {
                    let log_size = column.len().ilog2();
                    let log_ratio = chunk_max_log_size - log_size;
                    msgs[j] = column.at((i >> (log_ratio + 1) << 1) + (i & 1));
                }
                poseidon_update_m31s(&msgs, &mut prev_state);
                *curr_state = prev_state;
            });
            std::mem::swap(&mut prev_layer_states, &mut next_layer_states);
            prev_chunk_max_log_size = chunk_max_log_size;
        }

        #[cfg(not(feature = "parallel"))]
        let iter_states = next_layer_states.iter_mut();
        #[cfg(feature = "parallel")]
        let iter_states = next_layer_states.par_iter_mut();

        iter_states.enumerate().for_each(|(i, curr_state)| {
            let log_ratio = max_log_size - prev_chunk_max_log_size;
            let prev_state: [FieldElement252; 3] =
                prev_layer_states[(i >> (log_ratio + 1) << 1) + (i & 1)];
            let mut msgs: [M31; ELEMENTS_IN_BUFFER] = unsafe { std::mem::zeroed() };
            for (j, column) in last_chunk.iter().enumerate() {
                let log_size = column.len().ilog2();
                let log_ratio = max_log_size - log_size;
                msgs[j] = column.at((i >> (log_ratio + 1) << 1) + (i & 1));
            }
            *curr_state = poseidon_finalize_m31s(&msgs[..last_chunk.len()], prev_state);
        });
        next_layer_states.iter().map(|[fin, ..]| *fin).collect()
    }

    fn build_next_layer(
        prev_layer: &Col<Self, <Poseidon252MerkleHasher as MerkleHasherLifted>::Hash>,
    ) -> Col<Self, <Poseidon252MerkleHasher as MerkleHasherLifted>::Hash> {
        <CpuBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_next_layer(prev_layer)
    }
}

fn poseidon_update_m31s(msgs: &[M31; ELEMENTS_IN_BUFFER], prev_state: &mut [FieldElement252; 3]) {
    let field_elements: [FieldElement252; 2] = std::array::from_fn(|i| {
        construct_felt252_from_m31s(&msgs[i * ELEMENTS_IN_BLOCK..(i + 1) * ELEMENTS_IN_BLOCK])
    });
    poseidon_update(&field_elements, prev_state);
}

fn poseidon_finalize_m31s(msgs: &[M31], prev_state: [FieldElement252; 3]) -> [FieldElement252; 3] {
    let field_elements: Vec<FieldElement252> = msgs
        .chunks(ELEMENTS_IN_BLOCK)
        .map(construct_felt252_from_m31s)
        .collect();
    poseidon_finalize(&field_elements, prev_state)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::FieldElement252;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::CpuBackend;
    use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
    use crate::prover::vcs_lifted::prover::MerkleProverLifted;

    #[test]
    fn test_build_next_layer() {
        const LOG_SIZE: u32 = 6;
        let layer: Vec<FieldElement252> = (0u32..1 << (LOG_SIZE + 1))
            .map(FieldElement252::from)
            .collect();
        assert_eq!(
            <CpuBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_next_layer(&layer),
            <SimdBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_next_layer(&layer)
        );
    }

    fn prepare_poseidon_merkle_commit() -> (FieldElement252, FieldElement252) {
        const MAX_LOG_N_ROWS: u32 = 9;
        const N_COLS: u32 = 95;
        let mut cols: Vec<Vec<BaseField>> = (0..N_COLS)
            .map(|i| {
                (0..1 << MAX_LOG_N_ROWS)
                    .map(|j| M31::from(100 * i + j))
                    .collect_vec()
            })
            .collect();

        // Make the first two columns smaller to test a non-uniform sized trace.
        (0..20).for_each(|i| {
            cols[i] = (0..1 << (MAX_LOG_N_ROWS - 4))
                .map(M31::from_u32_unchecked)
                .collect_vec()
        });
        (20..40).for_each(|i| {
            cols[i] = (0..1 << (MAX_LOG_N_ROWS - 3))
                .map(M31::from_u32_unchecked)
                .collect_vec()
        });
        let cols_simd: Vec<BaseColumn> = cols
            .iter()
            .map(|c| BaseColumn::from_cpu(c.clone()))
            .collect();

        (
            MerkleProverLifted::<CpuBackend, Poseidon252MerkleHasher>::commit(
                cols.iter().collect(),
            )
            .root(),
            MerkleProverLifted::<SimdBackend, Poseidon252MerkleHasher>::commit(
                cols_simd.iter().collect(),
            )
            .root(),
        )
    }

    #[test]
    fn test_poseidon_merkle_commit() {
        let (cpu_root, simd_root) = prepare_poseidon_merkle_commit();
        assert_eq!(cpu_root, simd_root);
    }
    #[test]
    fn test_small_columns_leaves() {
        for log_size in 2..9 {
            const N_COLS: usize = 2;
            let cols: Vec<Vec<BaseField>> = (0..N_COLS)
                .map(|i| {
                    (0..1 << log_size)
                        .map(|j| M31::from(100 * i + j))
                        .collect_vec()
                })
                .collect();
            let cols_simd: Vec<BaseColumn> = cols
                .iter()
                .map(|c| BaseColumn::from_cpu(c.clone()))
                .collect();

            assert_eq!(
                <CpuBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_leaves(
                    &cols.iter().collect::<Vec<_>>()
                ),
                <SimdBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_leaves(
                    &cols_simd.iter().collect::<Vec<_>>()
                )
            );
        }
    }
}
