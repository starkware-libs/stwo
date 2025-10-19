use std::collections::BTreeMap;

use itertools::Itertools;
use tracing::{span, Level};

use super::ops::MerkleOpsLifted;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::verifier::MerkleDecommitment;
use crate::core::vcs::MerkleHasher;
use crate::prover::backend::{Col, Column};

#[derive(Debug)]
pub struct MerkleProver<B: MerkleOpsLifted<H>, H: MerkleHasher> {
    /// Layers of the Merkle tree.
    /// The first layer is the root layer.
    /// The last layer is the largest layer.
    /// See [MerkleOps::commit_on_layer] for more details.
    pub layers: Vec<Col<B, H::Hash>>,
}
/// The MerkleProver struct represents a prover for a Merkle commitment scheme.
/// It is generic over the types `B` and `H`, which represent the Merkle operations and Merkle
/// hasher respectively.
impl<B: MerkleOpsLifted<H>, H: MerkleHasher> MerkleProver<B, H> {
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

        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();

        let max_log_size = columns.last().unwrap().len().ilog2();
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
    /// * `queries_per_log_size` - Maps a log_size to a vector of queries for columns of that size.
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A vector queried values sorted by the order they were queried from the largest layer to
    ///   the smallest.
    /// * A `MerkleDecommitment` containing the hash and column witnesses.
    pub fn decommit(
        &self,
        _queries_per_log_size: &BTreeMap<u32, Vec<usize>>,
        _columns: Vec<&Col<B, BaseField>>,
    ) -> (Vec<BaseField>, MerkleDecommitment<H>) {
        unimplemented!() 
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
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use crate::prover::backend::CpuBackend;

    #[test]
    fn test_lifted_merkle_leaves() {
        // | 0 .. 3 | 0 .. 7 | 0 .. 15 |
        let columns: Vec<Vec<BaseField>> = (0..3)
            .map(|i| (0..1 << (i + 2)).map(M31::from_u32_unchecked).collect())
            .collect();
        let merkle_prover =
            MerkleProver::<CpuBackend, Blake2sMerkleHasher>::commit(columns.iter().collect());
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
}
