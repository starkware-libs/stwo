use std::collections::BTreeMap;

use hashbrown::HashMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std_shims::{vec, Vec};
use thiserror::Error;

use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Default)]
pub struct MerkleDecommitmentLifted<H: MerkleHasherLifted> {
    /// Hash values that the verifier needs but cannot deduce from previous computations, in the
    /// order they are needed.
    pub hash_witness: Vec<H::Hash>,
    // TODO(Leo): delete after e2e flow passes.
    pub column_witness: Vec<H::Hash>,
}

impl<H: MerkleHasherLifted> MerkleDecommitmentLifted<H> {
    pub const fn empty() -> Self {
        Self {
            hash_witness: Vec::new(),
            column_witness: Vec::new(),
        }
    }
}

/// Auxiliary data for Merkle decommitment.
#[derive(Clone, Debug)]
pub struct MerkleDecommitmentLiftedAux<H: MerkleHasherLifted> {
    /// For each layer, a map from node index to its hash value.
    pub all_node_values: Vec<HashMap<usize, H::Hash>>,
}

pub struct ExtendedMerkleDecommitmentLifted<H: MerkleHasherLifted> {
    pub decommitment: MerkleDecommitmentLifted<H>,
    pub aux: MerkleDecommitmentLiftedAux<H>,
}

pub struct MerkleVerifierLifted<H: MerkleHasherLifted> {
    /// The commitment value.
    pub root: H::Hash,
    // /// The number of columns committed.
    // pub n_columns: usize,
    // /// The largest log size of a committed column.
    // pub max_log_size: u32,
    pub column_log_sizes: Vec<u32>,
    pub n_columns_per_log_size: BTreeMap<u32, usize>,
}

impl<H: MerkleHasherLifted> MerkleVerifierLifted<H> {
    pub fn new(root: H::Hash, column_log_sizes: Vec<u32>) -> Self {
        let mut n_columns_per_log_size = BTreeMap::new();
        for log_size in &column_log_sizes {
            *n_columns_per_log_size.entry(*log_size).or_insert(0) += 1;
        }

        Self {
            root,
            column_log_sizes,
            n_columns_per_log_size,
        }
    }

    /// Verifies the decommitment of the columns.
    ///
    /// Returns `Ok(())` if the decommitment is successfully verified.
    ///
    /// # Arguments
    ///
    /// * `queries_positions` - Indices of the query positions (in range `[0,
    ///   2^self.max_log_size)`), in increasing order. Note that both the ordering and the value
    ///   bounds are not checked in this function.
    /// * `queried_values` - A vector of queried values according to the order in
    ///   [`MerkleProver::decommit()`].
    /// * `decommitment` - The decommitment object containing the hash witness.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the following conditions are met:
    ///
    /// * The witness is too long (not fully consumed).
    /// * The witness is too short (missing values).
    /// * The computed root does not match the expected root.
    ///
    /// # Note
    ///
    /// In the current implementation, the Merkle verifier expects a full row of values for each
    /// query. This means that the vector of queried values will contain redundancies: whenever
    /// two query positions map to the same index in a smaller column in the trace, the value at
    /// that index is sent twice.
    pub fn verify(
        &self,
        queries_position: &Vec<usize>,
        queried_values: Vec<BaseField>,
        decommitment: MerkleDecommitmentLifted<H>,
    ) -> Result<(), MerkleVerificationError> {
        let Some(max_log_size) = self.column_log_sizes.iter().max() else {
            return Ok(());
        };

        let n_columns = self.column_log_sizes.len();
        let mut prev_layer_hashes: Vec<(usize, H::Hash)> = queries_position
            .iter()
            .zip_eq(queried_values.chunks_exact(n_columns))
            .map(|(idx, column_values)| {
                let mut hasher = H::default_with_initial_state();
                hasher.update_leaf(column_values);
                (*idx, hasher.finalize())
            })
            .collect();

        let mut hash_witness = decommitment.hash_witness.into_iter();

        // Verify inner layers
        for _ in 0..*max_log_size {
            let mut curr_layer_hashes: Vec<(usize, H::Hash)> = vec![];
            for chunk in prev_layer_hashes.as_slice().chunk_by(|a, b| a.0 ^ 1 == b.0) {
                // If `chunk` has length 1, we need to fetch the brother of `chunk[0].1` from the
                // witness.
                let children = if chunk.len() == 1 {
                    let witness = hash_witness
                        .next()
                        .ok_or(MerkleVerificationError::WitnessTooShort)?;
                    match chunk[0].0 & 1 {
                        0 => (chunk[0].1, witness),
                        1 => (witness, chunk[0].1),
                        _ => unreachable!(),
                    }
                } else {
                    (chunk[0].1, chunk[1].1)
                };

                curr_layer_hashes.push((chunk[0].0 >> 1, H::hash_children(children)));
            }
            prev_layer_hashes = curr_layer_hashes;
        }
        // Check that the witness has been consumed.
        if hash_witness.next().is_some() {
            return Err(MerkleVerificationError::WitnessTooLong);
        }

        let [(_, computed_root)] = prev_layer_hashes.try_into().unwrap();
        if computed_root != self.root {
            return Err(MerkleVerificationError::RootMismatch);
        }

        Ok(())
    }
}

// TODO(ilya): Make error messages consistent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MerkleVerificationError {
    #[error("Witness is too short")]
    WitnessTooShort,
    #[error("Witness is too long.")]
    WitnessTooLong,
    #[error("Root mismatch.")]
    RootMismatch,
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use num_traits::Zero;

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::blake2_hash::Blake2sHash;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs_lifted::test_utils::prepare_merkle;
    use crate::core::vcs_lifted::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness[4] = Blake2sHash::default();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness.push(Blake2sHash::default());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }
}
