use std::collections::BTreeMap;

use itertools::Itertools;
use thiserror::Error;

use super::ops::MerkleHasher;
use super::prover::MerkleDecommitment;
use super::utils::{next_decommitment_node, option_flatten_peekable};
use crate::core::fields::m31::BaseField;
use crate::core::utils::PeekableExt;

pub struct MerkleVerifier<H: MerkleHasher> {
    pub root: H::Hash,
    pub column_log_sizes: Vec<u32>,
    pub n_columns_per_log_size: BTreeMap<u32, usize>,
}
impl<H: MerkleHasher> MerkleVerifier<H> {
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
    pub fn verify(
        &self,
        queries_per_log_size: &BTreeMap<u32, Vec<usize>>,
        queried_values: Vec<BaseField>,
        decommitment: MerkleDecommitment<H>,
    ) -> Result<(), MerkleVerificationError> {
        let Some(max_log_size) = self.column_log_sizes.iter().max() else {
            return Ok(());
        };

        let mut queried_values = queried_values.into_iter();

        // Prepare read buffers.

        let mut hash_witness = decommitment.hash_witness.into_iter();
        let mut column_witness = decommitment.column_witness.into_iter();

        let mut last_layer_hashes: Option<Vec<(usize, H::Hash)>> = None;
        for layer_log_size in (0..=*max_log_size).rev() {
            let n_columns_in_layer = *self
                .n_columns_per_log_size
                .get(&layer_log_size)
                .unwrap_or(&0);

            // Prepare write buffer for queries to the current layer. This will propagate to the
            // next layer.
            let mut layer_total_queries = vec![];

            // Queries to this layer come from queried node in the previous layer and queried
            // columns in this one.
            let mut prev_layer_queries = last_layer_hashes
                .iter()
                .flatten()
                .map(|(q, _)| *q)
                .collect_vec()
                .into_iter()
                .peekable();
            let mut prev_layer_hashes = last_layer_hashes.as_ref().map(|x| x.iter().peekable());
            let mut layer_column_queries =
                option_flatten_peekable(queries_per_log_size.get(&layer_log_size));

            // Merge previous layer queries and column queries.
            while let Some(node_index) =
                next_decommitment_node(&mut prev_layer_queries, &mut layer_column_queries)
            {
                prev_layer_queries
                    .peek_take_while(|q| q / 2 == node_index)
                    .for_each(drop);

                let node_hashes = prev_layer_hashes
                    .as_mut()
                    .map(|prev_layer_hashes| {
                        {
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
                            Ok((left_hash, right_hash))
                        }
                    })
                    .transpose()?;

                // If the column values were queried, read them from `queried_value`.
                let (err, node_values_iter) = match layer_column_queries.next_if_eq(&node_index) {
                    Some(_) => (
                        MerkleVerificationError::TooFewQueriedValues,
                        &mut queried_values,
                    ),
                    // Otherwise, read them from the witness.
                    None => (
                        MerkleVerificationError::WitnessTooShort,
                        &mut column_witness,
                    ),
                };

                let node_values = node_values_iter.take(n_columns_in_layer).collect_vec();
                if node_values.len() != n_columns_in_layer {
                    return Err(err);
                }

                layer_total_queries.push((node_index, H::hash_node(node_hashes, &node_values)));
            }

            last_layer_hashes = Some(layer_total_queries);
        }

        // Check that all witnesses and values have been consumed.
        if !hash_witness.is_empty() {
            return Err(MerkleVerificationError::WitnessTooLong);
        }
        if !queried_values.is_empty() {
            return Err(MerkleVerificationError::TooManyQueriedValues);
        }
        if !column_witness.is_empty() {
            return Err(MerkleVerificationError::WitnessTooLong);
        }

        let [(_, computed_root)] = last_layer_hashes.unwrap().try_into().unwrap();
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
    #[error("too many Queried values")]
    TooManyQueriedValues,
    #[error("too few queried values")]
    TooFewQueriedValues,
    #[error("Root mismatch.")]
    RootMismatch,
}
