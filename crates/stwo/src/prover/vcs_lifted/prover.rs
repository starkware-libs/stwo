use itertools::Itertools;
use tracing::{span, Level};

use super::ops::MerkleOpsLifted;
use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleDecommitmentLifted;
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
    pub fn commit(columns: Vec<&Col<B, BaseField>>) -> Self {
        let _span = span!(Level::TRACE, "Merkle", class = "MerkleCommitment").entered();
        if columns.is_empty() {
            return Self {
                layers: vec![B::build_leaves(&[])],
            };
        }

        let columns = &mut columns.into_iter().sorted_by_key(|c| c.len()).collect_vec();

        let max_log_size = columns.last().unwrap().len().ilog2();
        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();
        layers.push(B::build_leaves(columns));

        (0..max_log_size).for_each(|_| {
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
    /// * `queries_position` - Vector containing the positions of the queries, in increasing order.
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A vector of queried values. For each query position, the queried values are column values
    ///   corresponding to the query position, sorted increasingly by column length.
    /// * A `MerkleDecommitment` containing the hash witness.
    pub fn decommit(
        &self,
        queries_position: Vec<usize>,
        columns: Vec<&Col<B, BaseField>>,
    ) -> (Vec<BaseField>, MerkleDecommitmentLifted<H>) {
        // Prepare output buffers.
        let mut queried_values: Vec<BaseField> = vec![];
        let mut decommitment = MerkleDecommitmentLifted::<H>::default();

        let columns_sorted = columns.iter().sorted_by_key(|c| c.len()).collect_vec();

        // Compute the queried values.
        let max_log_size = self.layers.len() - 1;
        for pos in queries_position.iter() {
            let values = columns_sorted.iter().map(|col| {
                let log_size = col.len().ilog2() as usize;
                let shift = max_log_size - log_size;
                col.at((pos >> (shift + 1) << 1) + (pos & 1))
            });
            queried_values.extend(values);
        }

        let mut prev_layer_queries = queries_position;
        // The largest log size of a layer is equal to `self.layers.len() - 1`. We start iterating
        // from the layer of log size `self.layers.len() - 2` so that we always have a previous
        // layer available for the computation.
        for layer_log_size in (0..self.layers.len() - 1).rev() {
            // Prepare write buffer for queries to the current layer. This will propagate to the
            // next layer.
            let mut curr_layer_queries: Vec<usize> = vec![];

            // Each layer node is a hash of column values as previous layer hashes.
            // Prepare the previous layer hashes to read from.
            let prev_layer_hashes = self.layers.get(layer_log_size + 1).unwrap();
            // All chunks have either length 1 (only one child is present) or 2 (both children are
            // present).
            for queries_chunk in prev_layer_queries.as_slice().chunk_by(|a, b| a ^ 1 == *b) {
                let first = queries_chunk[0];
                // If the brother of `first` was not queried before, add its hash to the witness.
                if queries_chunk.len() == 1 {
                    decommitment
                        .hash_witness
                        .push(prev_layer_hashes.at(first ^ 1))
                }
                curr_layer_queries.push(first >> 1);
            }
            // Propagate queries to the next layer.
            prev_layer_queries = curr_layer_queries;
        }
        (queried_values, decommitment)
    }

    pub fn root(&self) -> H::Hash {
        self.layers.first().unwrap().at(0)
    }
}

#[cfg(test)]
mod test {
    use num_traits::Zero;

    use super::*;
    use crate::core::fields::m31::M31;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher as Blake2sMerkleHasherCurrent;
    use crate::core::vcs_lifted::blake2_merkle::{Blake2sMerkleHasher, LEAF_PREFIX};
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::{ColumnOps, CpuBackend};
    use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::vcs::prover::MerkleProver;

    #[test]
    fn test_empty_cols() {
        // Check Merkle commitment on empty columns.
        let mixed_degree_merkle_prover =
            MerkleProver::<CpuBackend, Blake2sMerkleHasherCurrent>::commit(vec![]);
        let lifted_merkle_prover =
            MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(vec![]);
        assert_eq!(
            mixed_degree_merkle_prover.layers,
            lifted_merkle_prover.layers
        );
    }

    fn prepare_merkle() -> (
        Vec<Vec<BaseField>>,
        MerkleProverLifted<CpuBackend, Blake2sHasher>,
    ) {
        let columns: Vec<Vec<BaseField>> = (2..5)
            .map(|i| (0..1 << i).map(M31::from_u32_unchecked).collect())
            .collect();
        let merkle_prover =
            MerkleProverLifted::<CpuBackend, Blake2sHasher>::commit(columns.iter().collect());
        (columns, merkle_prover)
    }

    #[test]
    fn test_lifted_merkle_leaves() {
        let (_, merkle_prover) = prepare_merkle();
        let leaves = &merkle_prover.layers.last().unwrap();

        // Compute the expected first leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = LEAF_PREFIX.to_vec();
        data.extend([0u8; 12]);
        hasher.update(&data);
        assert_eq!(hasher.finalize(), leaves[0]);

        // Compute the expected fifth leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = LEAF_PREFIX.to_vec();
        data.extend(0_u32.to_le_bytes());
        data.extend(2_u32.to_le_bytes());
        data.extend(4_u32.to_le_bytes());
        hasher.update(&data);
        assert_eq!(hasher.finalize(), leaves[4]);

        // Compute the expected last leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = LEAF_PREFIX.to_vec();
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
        let queried_values = merkle_prover.decommit(vec![0], cols.iter().collect_vec()).0;

        let expected_values = vec![BaseField::zero(); 3];
        assert_eq!(expected_values, queried_values);

        // Test decommits at position 4.
        let queried_values = merkle_prover.decommit(vec![4], cols.iter().collect_vec()).0;
        let expected_values = vec![
            BaseField::from_u32_unchecked(0),
            BaseField::from_u32_unchecked(2),
            BaseField::from_u32_unchecked(4),
        ];
        assert_eq!(expected_values, queried_values);

        // Test decommits at position 15.
        let queried_values = merkle_prover
            .decommit(vec![15], cols.iter().collect_vec())
            .0;
        let expected_values = vec![
            BaseField::from_u32_unchecked(3),
            BaseField::from_u32_unchecked(7),
            BaseField::from_u32_unchecked(15),
        ];
        assert_eq!(expected_values, queried_values);
    }

    fn lift_poly<B: ColumnOps<BaseField> + PolyOps>(
        poly: &CircleCoefficients<B>,
        lifted_log_size: u32,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
        let lifted_domain = CanonicCoset::new(lifted_log_size).circle_domain();
        let mut lifted_evaluation: Col<B, BaseField> = lifted_domain
            .iter()
            .map(|point| {
                poly.eval_at_point(
                    point
                        .repeated_double(lifted_log_size - poly.log_size())
                        .into_ef(),
                )
                .to_m31_array()[0]
            })
            .collect();
        <B as ColumnOps<BaseField>>::bit_reverse_column(&mut lifted_evaluation);
        CircleEvaluation::new(lifted_domain, lifted_evaluation)
    }

    /// See the docs of `[crate::prover::backend::cpu::blake2s_lifted::build_leaves]`.
    #[test]
    fn test_bit_reverse_lifted_merkle_cpu() {
        const LOG_SIZE: u32 = 3;
        const LIFTED_LOG_SIZE: u32 = 8;

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
        let lifted_merkle_prover_1 =
            MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(vec![
                &lifted_evaluation.values,
                &last_column,
            ]);
        let lifted_merkle_prover_2 =
            MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(vec![
                &poly.evaluate(domain),
                &last_column,
            ]);

        assert_eq!(lifted_merkle_prover_1.root(), lifted_merkle_prover_2.root());
        assert_eq!(
            mixed_degree_merkle_prover.root(),
            lifted_merkle_prover_1.root()
        );
    }
}
