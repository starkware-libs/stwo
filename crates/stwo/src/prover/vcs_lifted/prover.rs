use itertools::Itertools;
use tracing::{span, Level};

use super::ops::MerkleOpsLifted;
use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleDecommitmentLifted;
use crate::prover::backend::{Col, Column};

#[derive(Debug)]
pub struct MerkleProver<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    /// Layers of the Merkle tree.
    /// The first layer is the root layer.
    /// The last layer is the largest layer.
    pub layers: Vec<Col<B, H::Hash>>,
}
/// The MerkleProver struct represents a prover for a Merkle commitment scheme.
/// It is generic over the types `B` and `H`, which represent the Merkle operations and Merkle
/// hasher respectively.
impl<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> MerkleProver<B, H> {
    /// Commits to columns.
    /// Columns must be of power of 2 sizes.
    ///
    /// # Arguments
    ///
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A new instance of `MerkleProver` with the committed layers.
    pub fn commit(columns: Vec<&Col<B, BaseField>>) -> Self {
        let _span = span!(Level::TRACE, "Merkle", class = "MerkleCommitment").entered();
        if columns.is_empty() {
            return Self {
                layers: vec![B::commit_on_first_layer(0, &[])],
            };
        }

        let columns = &mut columns.into_iter().sorted_by_key(|c| c.len()).collect_vec();

        let max_log_size = columns.last().unwrap().len().ilog2();
        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();
        layers.push(B::commit_on_first_layer(max_log_size, columns));

        for log_size in (0..max_log_size).rev() {
            layers.push(B::commit_on_layer(log_size, layers.last().unwrap()));
        }
        layers.reverse();

        Self { layers }
    }

    /// Decommits to columns on the given queries.
    /// Queries are given as indices to the largest column.
    ///
    /// # Arguments
    ///
    /// * `queries_position` - Vector containing the positions of the queries.
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A vector of queried values sorted by the order they were queried from the smallest layer
    ///   to the smallest.
    /// * A `MerkleDecommitment` containing the hash and witness.
    ///
    /// TODO(Leo): document assumptions on queries_position
    pub fn decommit(
        &self,
        queries_position: Vec<usize>,
        columns: Vec<&Col<B, BaseField>>,
    ) -> (Vec<BaseField>, MerkleDecommitmentLifted<H>) {
        // Prepare output buffers.
        let mut queried_values: Vec<BaseField> = vec![];
        let mut decommitment = MerkleDecommitmentLifted::<H>::empty();

        let columns_sorted = columns.iter().sorted_by_key(|c| c.len()).collect_vec();

        for pos in queries_position.iter() {
            queried_values.extend(columns_sorted.iter().map(|c| c.at(pos % c.len())))
        }

        let mut last_layer_queries = queries_position;

        // TODO(Leo): do better indexing and/or document it.
        for layer_log_size in (0..self.layers.len() as u32 - 1).rev() {
            // Prepare write buffer for queries to the current layer. This will propagate to the
            // next layer.
            let mut layer_total_queries = vec![];

            // Each layer node is a hash of column values as previous layer hashes.
            // Prepare the relevant columns and previous layer hashes to read from.
            let previous_layer_hashes = self.layers.get(layer_log_size as usize + 1).unwrap();

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
    use super::*;
    use crate::core::fields::m31::M31;
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs_lifted::verifier::MerkleVerifierLifted;
    use crate::prover::backend::CpuBackend;

    fn prepare_merkle() -> (Vec<Vec<BaseField>>, MerkleProver<CpuBackend, Blake2sHasher>) {
        // TODO(Leo): write better.
        // | 0 .. 3 | 0 .. 7 | 0 .. 15 |
        let columns: Vec<Vec<BaseField>> = (0..3)
            .map(|i| (0..1 << (i + 2)).map(M31::from_u32_unchecked).collect())
            .collect();
        let merkle_prover =
            MerkleProver::<CpuBackend, Blake2sHasher>::commit(columns.iter().collect());
        (columns, merkle_prover)
    }


    #[test]
    fn test_lifted_merkle_leaves() {
        let (_, merkle_prover) = prepare_merkle();
        let leaves = &merkle_prover.layers.last().unwrap();
        let mut hasher = Blake2sHasher::default();
        hasher.update(&[0u8; 12]);
        assert_eq!(hasher.finalize(), leaves[0]);

        let mut hasher = Blake2sHasher::default();
        let mut data = vec![];
        data.extend(3_u32.to_le_bytes());
        data.extend(7_u32.to_le_bytes());
        data.extend(15_u32.to_le_bytes());
        hasher.update(&data);
        assert_eq!(hasher.finalize(), *leaves.last().unwrap());
    }

    #[test]
    fn test_lifted_decommit() {
        let (columns, merkle_prover) = prepare_merkle();
        let queries_position: Vec<usize> = vec![0, 2, 15];
        let (queried_values, decommitment) =
            merkle_prover.decommit(queries_position.clone(), columns.iter().collect_vec());
        let verifier = MerkleVerifierLifted::new(
            merkle_prover.root(),
            columns.len(),
            columns.last().unwrap().len().ilog2() as u32,
        );
        verifier
            .verify(queries_position, queried_values, decommitment)
            .unwrap();
    }
}
