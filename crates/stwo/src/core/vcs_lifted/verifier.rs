use hashbrown::hash_map::Entry;
use hashbrown::HashMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std_shims::{vec, Vec};
use thiserror::Error;

use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::ColumnVec;

/// The log number of consecutive QM31s packed into a single Merkle leaf.
///
/// When FRI layers are skipped, adjacent evaluation points are grouped into chunks of
/// `2^LOG_PACKED_LEAF_SIZE` and hashed together, reducing the number of Merkle tree leaves.
pub const LOG_PACKED_LEAF_SIZE: u32 = 2;
pub const PACKED_LEAF_SIZE: usize = 1 << LOG_PACKED_LEAF_SIZE;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Default)]
pub struct MerkleDecommitmentLifted<H: MerkleHasherLifted> {
    /// Hash values that the verifier needs but cannot deduce from previous computations, in the
    /// order they are needed.
    pub hash_witness: Vec<H::Hash>,
}

impl<H: MerkleHasherLifted> MerkleDecommitmentLifted<H> {
    pub const fn empty() -> Self {
        Self {
            hash_witness: Vec::new(),
        }
    }
}

/// Auxiliary data for Merkle decommitment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleDecommitmentLiftedAux<H: MerkleHasherLifted> {
    /// For each layer, a map from node index to its hash value.
    pub all_node_values: Vec<HashMap<usize, H::Hash>>,
}

pub struct ExtendedMerkleDecommitmentLifted<H: MerkleHasherLifted> {
    pub decommitment: MerkleDecommitmentLifted<H>,
    pub aux: MerkleDecommitmentLiftedAux<H>,
}

/// The verifier part of the vector commitment scheme.
// TODO(Leo): the fields column_log_sizes contains more information than needed for implementing a
// merkle verifier (knowing the max and length of column_log_sizes is enough). However, this info is
// needed by the pcs and storing it here makes integration easier. Consider refactoring the API.
pub struct MerkleVerifierLifted<H: MerkleHasherLifted> {
    /// The commitment value.
    pub root: H::Hash,
    /// A vector containing the log sizes of the columns that were committed, in the order they
    /// were sent to the MerkleProver (i.e. before sorting).
    pub column_log_sizes: Vec<u32>,
    /// The height of the Merkle tree. Note that it can be different than the largest committed
    /// column if the verifier has set a larger lifting size.
    pub height: u32,
}

impl<H: MerkleHasherLifted> MerkleVerifierLifted<H> {
    pub fn new(root: H::Hash, column_log_sizes: Vec<u32>, min_lifting_log_size: u32) -> Self {
        let max_column_log_size = column_log_sizes.iter().copied().max().unwrap_or_default();
        let height = min_lifting_log_size.max(max_column_log_size);
        Self {
            root,
            column_log_sizes,
            height,
        }
    }

    /// Verifies the decommitment of the columns.
    ///
    /// Returns `Ok(())` if the decommitment is successfully verified (including
    /// the case in which no columns were committed).
    ///
    /// # Arguments
    ///
    /// * `query_positions` - Indices of the query positions (assumed to be in range `[0,
    ///   2^max_log_size)` where max_log_size is the log size of the largest column). Note that the
    ///   positions are not necessarily sorted and may contain duplicates.
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
        query_positions: &[usize],
        queried_values: ColumnVec<Vec<BaseField>>,
        decommitment: MerkleDecommitmentLifted<H>,
    ) -> Result<(), MerkleVerificationError> {
        if self.height == 0 {
            return Ok(());
        };
        // Check that the structure of the input is consistent with the configurations.
        assert_eq!(queried_values.len(), self.column_log_sizes.len());
        for values in queried_values.iter() {
            assert_eq!(values.len(), query_positions.len());
        }

        // Sort the queried values in ascending order by column log size.
        let queried_values_sorted_by_log_size = queried_values
            .into_iter()
            .zip_eq(self.column_log_sizes.iter())
            .sorted_by_key(|(_, col_log_size)| *col_log_size)
            .map(|(vals, _)| vals)
            .collect_vec();

        // Dedup the query positions and the associated queried values.
        let mut positions_and_values: HashMap<usize, Vec<BaseField>> = Default::default();
        for (idx, pos) in query_positions.iter().enumerate() {
            let values = queried_values_sorted_by_log_size
                .iter()
                .map(|col_vals| col_vals[idx])
                .collect_vec();
            match positions_and_values.entry(*pos) {
                // If we encounter `pos` for the first time, we insert the queried values.
                Entry::Vacant(entry) => {
                    entry.insert(values);
                }
                // Otherwise, we check that the queried values are equal to what's already in the
                // map.
                Entry::Occupied(entry) => {
                    let old_values = entry.get();
                    assert_eq!(old_values, &values);
                }
            }
        }

        // Build the vector of pairs `(query_position, leaf_hash)`, sorted in ascending order by
        // `query_position`. Note that the query positions are already deduplicated by the hash map.
        let mut prev_layer_hashes: Vec<(usize, H::Hash)> = positions_and_values
            .into_iter()
            .sorted_by_key(|(pos, _)| *pos)
            .map(|(pos, values)| {
                let mut hasher = H::default();
                hasher.update_leaf(&values);
                (pos, hasher.finalize())
            })
            .collect();

        let mut hash_witness = decommitment.hash_witness.into_iter();
        // Verify inner layers
        for _ in 0..self.height {
            let mut curr_layer_hashes: Vec<(usize, H::Hash)> = vec![];
            // Chunk the previous layer by siblings.
            for chunk in prev_layer_hashes.as_slice().chunk_by(|a, b| a.0 ^ 1 == b.0) {
                // If `chunk` has length 1, we need to fetch the brother of `hash_0` from the
                // witness.
                let (idx_0, hash_0) = chunk[0];
                let children = if chunk.len() == 1 {
                    let witness = hash_witness
                        .next()
                        .ok_or(MerkleVerificationError::WitnessTooShort)?;
                    match idx_0 & 1 {
                        0 => (hash_0, witness),
                        1 => (witness, hash_0),
                        _ => unreachable!(),
                    }
                } else {
                    let (_, hash_1) = chunk[1];
                    (hash_0, hash_1)
                };

                curr_layer_hashes.push((idx_0 >> 1, H::hash_children(children)));
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

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MerkleVerificationError {
    #[error("Witness is too short.")]
    WitnessTooShort,
    #[error("Witness is too long.")]
    WitnessTooLong,
    #[error("Root mismatch.")]
    RootMismatch,
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::blake2_hash::Blake2sHash;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs_lifted::test_utils::prepare_merkle;
    use crate::core::vcs_lifted::verifier::{MerkleVerificationError, MerkleVerifierLifted};
    use crate::prover::backend::CpuBackend;
    use crate::prover::vcs_lifted::prover::MerkleProverLifted;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_merkle_success_poseidon() {
        use crate::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher;
        let (queries, decommitment, values, verifier) = prepare_merkle::<Poseidon252MerkleHasher>();
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
    fn test_merkle_invalid_witness_poseidon() {
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
        values[0][2] = BaseField::zero();

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

    #[test]
    fn test_merkle_unsorted_and_duplicate_query_positions() {
        let mut rng = SmallRng::seed_from_u64(42);
        let log_sizes = vec![3, 4, 3];
        let cols: Vec<Vec<BaseField>> = log_sizes
            .iter()
            .map(|&log_size| {
                (0..(1 << log_size))
                    .map(|_| BaseField::from(rng.gen_range(1..(1u32 << 30))))
                    .collect()
            })
            .collect();
        let max_log_size = *log_sizes.iter().max().unwrap();

        let merkle = MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(
            cols.iter().collect(),
            max_log_size,
            0,
        );

        // Queries given out of order with a duplicate.
        let queries = vec![13, 3, 7, 3, 1];
        let (values, decommitment) = merkle.decommit(&queries, cols.iter().collect());
        let verifier = MerkleVerifierLifted::new(merkle.root(), log_sizes, 0);
        verifier
            .verify(&queries, values, decommitment.decommitment)
            .unwrap();
    }
}
