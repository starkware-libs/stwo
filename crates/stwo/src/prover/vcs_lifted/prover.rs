use hashbrown::HashMap;
use itertools::Itertools;
use tracing::{span, Level};

use super::ops::MerkleOpsLifted;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{
    ExtendedMerkleDecommitmentLifted, MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux,
};
use crate::core::ColumnVec;
use crate::prover::backend::{Col, Column};

/// Represents the prover side of a Merkle commitment scheme.
#[derive(Debug)]
pub struct MerkleProverLifted<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    /// Layers of the Merkle tree, sorted by increasing length.
    /// The first layer is a column of length 1, containing the root commitment.
    pub layers: Vec<Col<B, H::Hash>>,
}

impl<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> MerkleProverLifted<B, H> {
    /// Commits to columns.
    /// Columns must be of power of 2 sizes, not necessarily sorted by length.
    ///
    /// # Arguments
    ///
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A new instance of `MerkleProverLifted` with the committed layers.
    pub fn commit(
        columns: Vec<&Col<B, BaseField>>,
        lifting_log_size: u32,
        log_rows_per_leaf: u32,
    ) -> Self {
        let _span = span!(Level::TRACE, "Merkle", class = "MerkleCommitment").entered();
        if columns.is_empty() {
            return Self {
                layers: vec![B::build_leaves(&[], lifting_log_size)],
            };
        }

        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();
        // We enter this branch only during FRI commit phase, in which we commit 4 columns of the
        // same size. In particular, we don't need to sort the columns by size.
        if log_rows_per_leaf > 0 {
            // TODO(Leo): add support for higher log_rows_per_leaf sizes.
            assert_eq!(
                log_rows_per_leaf, 2,
                "Leaf packing is only supported for log_rows_per_leaf = 2."
            );
            let columns: [&Col<B, BaseField>; SECURE_EXTENSION_DEGREE] =
                columns.try_into().unwrap();
            let packed_columns = B::pack_leaves_input(&columns);
            let max_log_size = packed_columns[0].len().ilog2();
            assert!(lifting_log_size >= max_log_size);
            layers.push(B::build_leaves(
                &packed_columns.iter().collect_vec(),
                lifting_log_size,
            ));
        } else {
            let sorted_columns = columns.into_iter().sorted_by_key(|c| c.len()).collect_vec();
            let max_log_size = sorted_columns.last().unwrap().len().ilog2();
            assert!(lifting_log_size >= max_log_size);
            layers.push(B::build_leaves(&sorted_columns, lifting_log_size));
        }

        (0..lifting_log_size).for_each(|_| {
            layers.push(B::build_next_layer(layers.last().unwrap()));
        });
        layers.reverse();

        Self { layers }
    }

    /// Decommits to columns on the given queries.
    /// Queries are given as indices to the largest column.
    ///
    /// # Arguments
    ///
    /// * `query_positions` - Vector containing the positions of the queries. Note that the
    ///   positions are not necessarily sorted and may contain duplicates.
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A vector `v` of queried values, where `v[col_idx][pos_idx]` contains the evaluation of the
    ///   `col_idx`-th column at the `query_position[pos_idx]` position.
    /// * A `MerkleDecommitment` containing the hash witness.
    pub fn decommit(
        &self,
        query_positions: &[usize],
        columns: Vec<&Col<B, BaseField>>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        ExtendedMerkleDecommitmentLifted<H>,
    ) {
        // Prepare output buffers.
        let mut queried_values: ColumnVec<Vec<BaseField>> = vec![];
        let mut decommitment = MerkleDecommitmentLifted::<H>::default();
        let mut all_node_values: Vec<HashMap<usize, <H as MerkleHasherLifted>::Hash>> = vec![];

        // Compute the queried values.
        let max_log_size = self.layers.len() - 1;
        for col in columns.iter() {
            let log_size = col.len().ilog2() as usize;
            let shift = max_log_size - log_size;
            // Map each query position to its column index (unchanged formula), then gather all in a
            // single batched read. `batch_at` returns values in the SAME order as these indices, so
            // `res[k]` equals the previous `col.at(idx_k)` for every `k` — byte-identical.
            let indices: Vec<usize> = query_positions
                .iter()
                .map(|pos| (pos >> (shift + 1) << 1) + (pos & 1))
                .collect();
            queried_values.push(col.batch_at(&indices));
        }

        let mut prev_layer_queries: Vec<usize> =
            query_positions.iter().copied().sorted().dedup().collect();
        // The largest log size of a layer is equal to `self.layers.len() - 1`. We start iterating
        // from the layer of log size `self.layers.len() - 2` so that we always have a previous
        // layer available for the computation.
        for layer_log_size in (0..self.layers.len() - 1).rev() {
            let mut all_node_values_for_layer =
                HashMap::<usize, <H as MerkleHasherLifted>::Hash>::new();
            // Prepare write buffer for queries to the current layer. This will propagate to the
            // next layer.
            let mut curr_layer_queries: Vec<usize> = vec![];

            // Each layer node is a hash of column values as previous layer hashes.
            // Prepare the previous layer hashes to read from.
            let prev_layer_hashes = self.layers.get(layer_log_size + 1).unwrap();
            // All chunks have either length 1 (only one child is present) or 2 (both children are
            // present).
            let chunks: Vec<&[usize]> = prev_layer_queries
                .as_slice()
                .chunk_by(|a, b| a ^ 1 == *b)
                .collect();
            // PASS 1: collect the hash indices to read, in the EXACT order the original per-element
            // loop read them — per chunk: optionally `first ^ 1` (the witness sibling, only for a
            // length-1 chunk), then `2 * curr_index`, then `2 * curr_index + 1`. A single batched
            // read then replaces one device->host copy per element with one per layer.
            let mut read_indices: Vec<usize> = Vec::new();
            for queries_chunk in &chunks {
                let first = queries_chunk[0];
                if queries_chunk.len() == 1 {
                    read_indices.push(first ^ 1);
                }
                let curr_index = first >> 1;
                read_indices.push(2 * curr_index);
                read_indices.push(2 * curr_index + 1);
            }
            let hashes = prev_layer_hashes.batch_at(&read_indices);
            // PASS 2: replay the original loop, consuming `hashes` in the same order they were
            // requested. `hash_witness` push order and the `all_node_values` (key -> hash) contents
            // are therefore identical to the per-element path (the map is order-insensitive, and
            // the witness pushes happen in the same chunk order with the same values).
            let mut hashes_iter = hashes.into_iter();
            for queries_chunk in &chunks {
                let first = queries_chunk[0];
                // If the brother of `first` was not queried before, add its hash to the witness.
                if queries_chunk.len() == 1 {
                    decommitment.hash_witness.push(hashes_iter.next().unwrap())
                }
                let curr_index = first >> 1;
                curr_layer_queries.push(curr_index);

                // Add the previous layer hashes to all_node_values.
                all_node_values_for_layer.insert(2 * curr_index, hashes_iter.next().unwrap());
                all_node_values_for_layer.insert(2 * curr_index + 1, hashes_iter.next().unwrap());
            }
            // Propagate queries to the next layer.
            prev_layer_queries = curr_layer_queries;

            all_node_values.push(all_node_values_for_layer);
        }
        (
            queried_values,
            ExtendedMerkleDecommitmentLifted {
                decommitment,
                aux: MerkleDecommitmentLiftedAux { all_node_values },
            },
        )
    }

    pub fn root(&self) -> H::Hash {
        // Option B: the root is read then immediately mixed into the channel to draw the next
        // challenge (serial commit/FRI chain), so this single read is latency-critical. Use the
        // pinned minimal-latency read (Cuda overrides it; CPU/SIMD default to `at`,
        // byte-identical).
        self.layers.first().unwrap().at_root_pinned(0)
    }
}

#[cfg(test)]
mod test {
    use num_traits::Zero;

    use super::*;
    use crate::core::fields::m31::M31;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher as Blake2sMerkleHasherCurrent;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs_lifted::test_utils::lift_poly;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::CpuBackend;
    use crate::prover::vcs::prover::MerkleProver;

    #[test]
    fn test_empty_cols() {
        // Check Merkle commitment on empty columns.
        let mixed_degree_merkle_prover =
            MerkleProver::<CpuBackend, Blake2sMerkleHasherCurrent>::commit(vec![]);
        let lifted_merkle_prover =
            MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(vec![], 0, 0);
        assert_eq!(
            mixed_degree_merkle_prover.layers,
            lifted_merkle_prover.layers
        );
    }

    fn prepare_merkle() -> (
        Vec<Vec<BaseField>>,
        MerkleProverLifted<CpuBackend, Blake2sHasher>,
    ) {
        let max_log_size = 4;
        let columns: Vec<Vec<BaseField>> = (2..=max_log_size)
            .map(|i| (0..1 << i).map(M31::from_u32_unchecked).collect())
            .collect();
        let merkle_prover = MerkleProverLifted::<CpuBackend, Blake2sHasher>::commit(
            columns.iter().collect(),
            max_log_size,
            0,
        );
        (columns, merkle_prover)
    }

    #[test]
    fn test_lifted_merkle_leaves() {
        let (_, merkle_prover) = prepare_merkle();
        let leaves = &merkle_prover.layers.last().unwrap();

        // Compute the expected first leaf.
        let mut hasher = Blake2sHasher::default();
        let data = [0u8; 12];
        hasher.update(&data);
        assert_eq!(hasher.finalize(), leaves[0]);

        // Compute the expected fifth leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = Vec::new();
        data.extend(0_u32.to_le_bytes());
        data.extend(2_u32.to_le_bytes());
        data.extend(4_u32.to_le_bytes());
        hasher.update(&data);
        assert_eq!(hasher.finalize(), leaves[4]);

        // Compute the expected last leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = Vec::new();
        data.extend(3_u32.to_le_bytes());
        data.extend(7_u32.to_le_bytes());
        data.extend(15_u32.to_le_bytes());
        hasher.update(&data);

        assert_eq!(hasher.finalize(), *leaves.last().unwrap());
    }

    #[test]
    fn test_lifted_decommitted_values() {
        let (cols, merkle_prover) = prepare_merkle();
        // Test decommits at position 0.
        let queried_values = merkle_prover.decommit(&[0], cols.iter().collect_vec()).0;

        let expected_values = vec![vec![BaseField::zero()]; 3];
        assert_eq!(expected_values, queried_values);

        // Test decommits at position 4.
        let queried_values = merkle_prover.decommit(&[4], cols.iter().collect_vec()).0;
        let expected_values = vec![
            vec![BaseField::from_u32_unchecked(0)],
            vec![BaseField::from_u32_unchecked(2)],
            vec![BaseField::from_u32_unchecked(4)],
        ];
        assert_eq!(expected_values, queried_values);

        // Test decommits at position 15.
        let queried_values = merkle_prover.decommit(&[15], cols.iter().collect_vec()).0;
        let expected_values = vec![
            vec![BaseField::from_u32_unchecked(3)],
            vec![BaseField::from_u32_unchecked(7)],
            vec![BaseField::from_u32_unchecked(15)],
        ];
        assert_eq!(expected_values, queried_values);
    }

    /// See the docs of `[crate::prover::backend::cpu::blake2s_lifted::build_leaves]`.
    #[test]
    fn test_bit_reverse_lifted_merkle_cpu() {
        const LOG_SIZE: u32 = 3;
        const LIFTED_LOG_SIZE: u32 = 9;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let poly = CpuCirclePoly::new((0..1 << LOG_SIZE).map(BaseField::from).collect());
        let lifted_evaluation = lift_poly(&poly, LIFTED_LOG_SIZE);

        let last_column: Col<CpuBackend, BaseField> =
            (0..1 << LIFTED_LOG_SIZE).map(|_| M31::zero()).collect_vec();

        let mixed_degree_merkle_prover =
            MerkleProver::<CpuBackend, Blake2sMerkleHasherCurrent>::commit(vec![
                &lifted_evaluation.values,
                &last_column,
            ]);
        let lifted_merkle_prover_1 = MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(
            vec![&lifted_evaluation.values, &last_column],
            LIFTED_LOG_SIZE,
            0,
        );
        let lifted_merkle_prover_2 = MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(
            vec![&poly.evaluate(domain), &last_column],
            LIFTED_LOG_SIZE,
            0,
        );

        assert_eq!(lifted_merkle_prover_1.root(), lifted_merkle_prover_2.root());
        assert_eq!(
            mixed_degree_merkle_prover.root(),
            lifted_merkle_prover_1.root()
        );
    }

    #[test]
    fn test_decommitment_aux() {
        let (columns, merkle_prover) = prepare_merkle();
        let (
            _,
            ExtendedMerkleDecommitmentLifted {
                decommitment: _,
                aux,
            },
        ) = merkle_prover.decommit(&[1], columns.iter().collect_vec());

        let mut expected: Vec<HashMap<usize, Blake2sHash>> = vec![];
        merkle_prover
            .layers
            .iter()
            .skip(1)
            .rev()
            .for_each(|layer| expected.push(HashMap::from_iter([(0, layer[0]), (1, layer[1])])));
        assert_eq!(expected, aux.all_node_values);
    }

    // ---------------------------------------------------------------------------------------------
    // Host-staging recovery (route b) support tests.
    //
    // These pin down the *recovery* half of the streaming-Merkle host-staging design: given a
    // query position `pos` (an index into the LARGEST column), which element of a smaller staged
    // column provides that query's opening. This is the exact mapping `decommit` uses at
    // prover.rs:114 — `col.at((pos >> (shift + 1) << 1) + (pos & 1))` — where
    // `shift = max_log_size - col_log_size`. Route b will read this index out of a HOST buffer (a
    // D2H-staged copy of the column) instead of the resident device column, so getting the map
    // byte-exact is load-bearing. The tests are pure CPU and run on the laptop with no cuda.
    // ---------------------------------------------------------------------------------------------

    /// The query-position -> staged-buffer index map used by `decommit`.
    /// `pos` is an index into the largest column (log size `max_log_size`); `col_log_size` is the
    /// log size of the (possibly smaller) column whose opening we want. Identical to the inline
    /// expression at the top of `decommit`.
    fn query_to_buffer_index(pos: usize, max_log_size: usize, col_log_size: usize) -> usize {
        let shift = max_log_size - col_log_size;
        (pos >> (shift + 1) << 1) + (pos & 1)
    }

    #[test]
    fn test_query_to_buffer_index_map() {
        // (a) Equal sizes (shift = 0): the map is the identity. This is the main-tree case where
        // all eval columns are at the lifting size, so it must be exact.
        for pos in 0..16usize {
            assert_eq!(
                query_to_buffer_index(pos, 4, 4),
                pos,
                "shift=0 must be identity"
            );
        }

        // (b) shift = 1 (column is half the largest size). The map keeps the parity bit (pos & 1)
        // and drops one of the middle index bits: index = (pos >> 2 << 1) + (pos & 1).
        // Hand-computed table for max_log_size=4, col_log_size=3 over pos = 0..8:
        //   pos: 0 1 2 3 4 5 6 7
        //   idx: 0 1 0 1 2 3 2 3
        let expected_shift1 = [0usize, 1, 0, 1, 2, 3, 2, 3];
        for (pos, &want) in expected_shift1.iter().enumerate() {
            assert_eq!(
                query_to_buffer_index(pos, 4, 3),
                want,
                "shift=1 mismatch at pos={pos}"
            );
        }

        // (c) shift = 2 (column is a quarter of the largest size). max_log_size=4, col_log_size=2.
        //   index = (pos >> 3 << 1) + (pos & 1):
        //   pos:  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        //   idx:  0 1 0 1 0 1 0 1 2 3  2  3  2  3  2  3
        let expected_shift2 = [0usize, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3];
        for (pos, &want) in expected_shift2.iter().enumerate() {
            assert_eq!(
                query_to_buffer_index(pos, 4, 2),
                want,
                "shift=2 mismatch at pos={pos}"
            );
        }

        // (d) Every produced index is in-bounds for the staged column [0, 2^col_log_size).
        // Columns always have len >= 2 (asserted in `build_leaves`), i.e. col_log_size >= 1; the
        // map relies on this (for col_log_size = 0 it would return parity bit 1, out of a len-1
        // buffer). Range starts at 1 to match the real committed-column invariant.
        for col_log_size in 1..6usize {
            let max_log_size = 6usize;
            for pos in 0..(1usize << max_log_size) {
                let idx = query_to_buffer_index(pos, max_log_size, col_log_size);
                assert!(
                    idx < (1usize << col_log_size),
                    "index {idx} out of bounds for col_log_size={col_log_size} at pos={pos}"
                );
            }
        }
    }

    #[test]
    fn test_host_stage_recovery_roundtrip_matches_decommit() {
        // Route b round-trip: stage each committed column to a HOST buffer (here: clone the column
        // values into a plain Vec, modelling the D2H copy + device free), then RECOVER the queried
        // openings from the host buffer using the same query->buffer index map, and assert the
        // recovered values are byte-for-byte identical to what `decommit` returns from the resident
        // columns. This is the equivalence property route b must preserve (the proof is unchanged;
        // only where the opening bytes are read from changes).
        let max_log_size = 4usize;
        let columns: Vec<Vec<BaseField>> = (2..=max_log_size as u32)
            .map(|i| (0..1u32 << i).map(M31::from_u32_unchecked).collect())
            .collect();
        let merkle_prover = MerkleProverLifted::<CpuBackend, Blake2sHasher>::commit(
            columns.iter().collect(),
            max_log_size as u32,
            0,
        );

        // A spread of query positions into the largest column, incl. duplicates and both parities.
        let query_positions = [0usize, 1, 4, 7, 7, 15, 8, 3];

        // Reference: the resident-column decommit path.
        let (decommit_values, _) =
            merkle_prover.decommit(&query_positions, columns.iter().collect_vec());

        // Stage to host (model the D2H copy that frees the device column).
        let host_buffers: Vec<Vec<BaseField>> = columns.clone();

        // Recover openings purely from the host buffers via the query->buffer index map.
        let recovered: Vec<Vec<BaseField>> = host_buffers
            .iter()
            .map(|buf| {
                let col_log_size = buf.len().ilog2() as usize;
                query_positions
                    .iter()
                    .map(|&pos| buf[query_to_buffer_index(pos, max_log_size, col_log_size)])
                    .collect()
            })
            .collect();

        assert_eq!(
            recovered, decommit_values,
            "host-staged recovery must be byte-identical to resident decommit"
        );
    }
}
