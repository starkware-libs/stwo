use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std_shims::{vec, Vec};
use thiserror::Error;

use crate::core::fields::m31::BaseField;
use crate::core::utils::PeekableExt;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd)]
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
/// TODO(Leo): document requirements on n_columns and log_size.
pub struct MerkleVerifierLifted<H: MerkleHasherLifted> {
    pub root: H::Hash,
    pub n_columns: usize,
    pub log_size: u32,
}

impl<H: MerkleHasherLifted> MerkleVerifierLifted<H> {
    pub fn new(root: H::Hash, n_columns: usize, log_size: u32) -> Self {
        Self {
            root,
            n_columns,
            log_size,
        }
    }
    /// Verifies the decommitment of the columns.
    ///
    /// Returns `Ok(())` if the decommitment is successfully verified.
    ///
    /// # Arguments
    ///
    /// * `queries_per_log_size` - A map from log_size to a vector of queries for columns of that
    ///   log_size.
    /// * `queried_values` - A vector of queried values according to the order in
    ///   [`MerkleProver::decommit()`].
    /// * `decommitment` - The decommitment object containing the witness and column values.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the following conditions are met:
    ///
    /// * The witness is too long (not fully consumed).
    /// * The witness is too short (missing values).
    /// * Too many queried values (not fully consumed).
    /// * Too few queried values (missing values).
    /// * The computed root does not match the expected root.
    ///
    /// [`MerkleProver::decommit()`]: crate::core::...::MerkleProver::decommit
    ///
    /// TODO(Leo): document assumptions on query positions.
    pub fn verify(
        &self,
        queries_position: Vec<usize>,
        queried_values: Vec<BaseField>,
        decommitment: MerkleDecommitmentLifted<H>,
    ) -> Result<(), MerkleVerificationError> {
        // Compute the leaf layer. Panics if queried_values.len() != queries_position * n_columns.
        // There may be an optimiziation to do here: not give repeated values of small
        // columns.
        let mut last_layer_hashes: Vec<(usize, H::Hash)> = queries_position
            .iter()
            .zip_eq(queried_values.chunks_exact(self.n_columns))
            .map(|(idx, column_values)| {
                let hasher = H::default();
                let hash = hasher.finalize_leaf_slice(column_values);
                (*idx, hash)
            })
            .collect();

        let mut hash_witness = decommitment.hash_witness.into_iter();

        // Verify inner layers
        for _ in (0..self.log_size).rev() {
            let mut layer_total_queries = vec![];

            let mut prev_layer_queries = last_layer_hashes
                .iter()
                .map(|(q, _)| *q)
                .collect_vec()
                .into_iter()
                .peekable();

            let mut prev_layer_hashes = last_layer_hashes.iter().peekable();

            while let Some(node_index) = prev_layer_queries.peek().map(|q| q / 2) {
                prev_layer_queries
                    .peek_take_while(|q| q / 2 == node_index)
                    .for_each(drop);

                // If the left child was not computed, read it from the witness.
                let left_hash = prev_layer_hashes
                    .next_if(|(index, _)| *index == 2 * node_index)
                    .map(|(_, hash)| Ok(*hash))
                    .unwrap_or_else(|| {
                        hash_witness
                            .next()
                            .ok_or(MerkleVerificationError::WitnessTooShort)
                    })?;

                // If the right child was not computed, read it to from the witness.
                let right_hash = prev_layer_hashes
                    .next_if(|(index, _)| *index == 2 * node_index + 1)
                    .map(|(_, hash)| Ok(*hash))
                    .unwrap_or_else(|| {
                        hash_witness
                            .next()
                            .ok_or(MerkleVerificationError::WitnessTooShort)
                    })?;
                let node_hashes = (left_hash, right_hash);
                layer_total_queries.push((node_index, H::hash_children(node_hashes)));
            }
            last_layer_hashes = layer_total_queries;
        }
        // Check that all witnesses and values have been consumed.
        if hash_witness.next().is_some() {
            return Err(MerkleVerificationError::WitnessTooLong);
        }

        let [(_, computed_root)] = last_layer_hashes.try_into().unwrap();
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
