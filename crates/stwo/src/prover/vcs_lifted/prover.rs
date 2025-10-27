use itertools::Itertools;
use tracing::{span, Level};

use super::ops::MerkleOpsLifted;
use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleDecommitmentLifted;
use crate::prover::backend::{Col, Column};

/// The MerkleProverLifted struct represents a prover for a Merkle commitment scheme.
#[derive(Debug)]
pub struct MerkleProverLifted<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    /// Layers of the Merkle tree, sorted by increasing length.
    /// The first layer is a column of length 1, containing the root commitment.
    pub layers: Vec<Col<B, H::Hash>>,
}

/// It is generic over the types `B` and `H`, which represent the Merkle operations and Merkle
/// hasher respectively.
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
                layers: vec![B::commit_on_first_layer(&[])],
            };
        }

        let columns = &mut columns.into_iter().sorted_by_key(|c| c.len()).collect_vec();

        let max_log_size = columns.last().unwrap().len().ilog2();
        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();
        layers.push(B::commit_on_first_layer(columns));

        (0..max_log_size).rev().for_each(|_| {
            layers.push(B::commit_on_inner_layer(layers.last().unwrap()));
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
        assert!(*queries_position.last().unwrap() < columns_sorted.last().unwrap().len());

        for pos in queries_position.iter() {
            queried_values.extend(columns_sorted.iter().map(|c| c.at(pos % c.len())))
        }

        let mut last_layer_queries = queries_position;

        // The largest log size of a layer is equal to `self.layers.len() - 1`. We start iterating
        // from the layer of log size `self.layers.len() - 2` so that we always have a previous
        // layer available for the computation.
        for layer_log_size in (0..self.layers.len() - 1).rev() {
            // Prepare write buffer for queries to the current layer. This will propagate to the
            // next layer.
            let mut layer_total_queries = vec![];

            // Each layer node is a hash of column values as previous layer hashes.
            // Prepare the relevant columns and previous layer hashes to read from.
            let previous_layer_hashes = self.layers.get(layer_log_size + 1).unwrap();

            // Queries to this layer come from queried node in the previous layer.
            let mut prev_layer_queries = last_layer_queries.into_iter().peekable();

            while let Some(node_index) = prev_layer_queries.peek().map(|q| q / 2) {
                // If the left child was not computed, add it to the witness.
                if prev_layer_queries.next_if_eq(&(2 * node_index)).is_none() {
                    decommitment
                        .hash_witness
                        .push(previous_layer_hashes.at(2 * node_index));
                }
                // If the right child was not computed, add it to the witness.
                if prev_layer_queries
                    .next_if_eq(&(2 * node_index + 1))
                    .is_none()
                {
                    decommitment
                        .hash_witness
                        .push(previous_layer_hashes.at(2 * node_index + 1));
                }

                layer_total_queries.push(node_index);
            }

            // Propagate queries to the next layer.
            last_layer_queries = layer_total_queries;
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
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs::blake2_merkle::LEAF_PREFIX;
    use crate::prover::backend::CpuBackend;

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
        let queried_values = merkle_prover.decommit(vec![0], cols.iter().collect_vec()).0;
        // Build the expected queried values at position 0.
        let expected_values = vec![BaseField::zero(); 3];
        assert_eq!(expected_values, queried_values);

        // 15 is the largest size of a column.
        let queried_values = merkle_prover
            .decommit(vec![15], cols.iter().collect_vec())
            .0;
        // Build the expected queried values at position 15.
        let expected_values = vec![
            BaseField::from_u32_unchecked(3),
            BaseField::from_u32_unchecked(7),
            BaseField::from_u32_unchecked(15),
        ];
        assert_eq!(expected_values, queried_values);
    }
}
