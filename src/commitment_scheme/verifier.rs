use std::cmp::Reverse;
use std::collections::BTreeMap;

use itertools::Itertools;
use thiserror::Error;

use super::ops::MerkleHasher;
use super::prover::MerkleDecommitment;
use crate::core::fields::m31::BaseField;
use crate::core::utils::PeekableExt;

// TODO(spapini): This struct is not necessary. Make it a function on decommitment?
pub struct MerkleVerifier<H: MerkleHasher> {
    pub root: H::Hash,
    pub column_log_sizes: Vec<u32>,
}
impl<H: MerkleHasher> MerkleVerifier<H> {
    /// Verifies the decommitment of the columns.
    ///
    /// # Arguments
    ///
    /// * `queries` - A vector of indices representing the queries to the largest column.
    ///  Note: This is sufficient for bit reversed STARK commitments.
    ///     It could be extended to support queries to any column.
    /// * `values` - A vector of pairs containing the log_size of the column and the decommitted
    ///   values of the column. Must be given in the same order as the columns were committed.
    /// * `decommitment` - The decommitment object containing the witness and column values.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the following conditions are met:
    ///
    /// * The witness is too long (not fully consumed).
    /// * The witness is too short (missing values).
    /// * The column values are too long (not fully consumed).
    /// * The column values are too short (missing values).
    /// * The computed root does not match the expected root.
    ///
    /// # Panics
    ///
    /// This function will panic if the `values` vector is not sorted in descending order based on
    /// the `log_size` of the columns.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the decommitment is successfully verified.
    pub fn verify(
        &self,
        queries_per_log_size: BTreeMap<u32, Vec<usize>>,
        queried_values: Vec<Vec<BaseField>>,
        decommitment: MerkleDecommitment<H>,
    ) -> Result<(), MerkleVerificationError> {
        let max_log_size = self.column_log_sizes.iter().max().copied().unwrap_or(0);

        // Prepare read buffers.
        let mut queried_values_by_layer = self
            .column_log_sizes
            .iter()
            .copied()
            .zip(
                queried_values
                    .into_iter()
                    .map(|column_values| column_values.into_iter()),
            )
            .sorted_by_key(|(log_size, _)| Reverse(*log_size))
            .peekable();
        let mut hash_witness = decommitment.hash_witness.into_iter();
        let mut column_witness = decommitment.column_witness.into_iter();

        let mut last_layer_hashes: Option<Vec<(usize, H::Hash)>> = None;
        for layer_log_size in (0..=max_log_size).rev() {
            // Prepare read buffer for queried values to the current layer.
            let mut layer_queried_values = queried_values_by_layer
                .peek_take_while(|(log_size, _)| *log_size == layer_log_size)
                .collect_vec();
            let n_cols = layer_queried_values.len();

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
            let mut layer_column_queries = queries_per_log_size
                .get(&layer_log_size)
                .into_iter()
                .flatten()
                .copied()
                .peekable();

            // Merge previous layer queries and column queries.
            while let Some(node_index) = prev_layer_queries
                .peek()
                .map(|q| *q / 2)
                .into_iter()
                .chain(layer_column_queries.peek().into_iter().copied())
                .min()
            {
                prev_layer_queries
                    .peek_take_while(|q| q / 2 == node_index)
                    .for_each(drop);
                // TODO: Handle first layer.
                let node_hashes = prev_layer_hashes
                    .as_mut()
                    .map(|prev_layer_hashes| {
                        {
                            // If the left child was not computed, read it from the witness.
                            let left_hash = prev_layer_hashes
                                .next_if(|(index, _)| *index == 2 * node_index)
                                .map(|(_, hash)| Ok(hash.clone()))
                                .unwrap_or_else(|| {
                                    hash_witness
                                        .next()
                                        .ok_or(MerkleVerificationError::WitnessTooShort)
                                })?;

                            // If the right child was not computed, read it to from the witness.
                            let right_hash = prev_layer_hashes
                                .next_if(|(index, _)| *index == 2 * node_index + 1)
                                .map(|(_, hash)| Ok(hash.clone()))
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
                let node_values = if layer_column_queries.next_if_eq(&node_index).is_some() {
                    layer_queried_values
                        .iter_mut()
                        .map(|(_, ref mut column_queries)| {
                            column_queries
                                .next()
                                .ok_or(MerkleVerificationError::ColumnValuesTooShort)
                        })
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    // Otherwise, read them from the witness.
                    (&mut column_witness).take(n_cols).collect_vec()
                };
                if node_values.len() != n_cols {
                    return Err(MerkleVerificationError::WitnessTooShort);
                }

                layer_total_queries.push((node_index, H::hash_node(node_hashes, &node_values)));
            }

            if !layer_queried_values.iter().all(|(_, c)| c.is_empty()) {
                return Err(MerkleVerificationError::ColumnValuesTooLong);
            }
            last_layer_hashes = Some(layer_total_queries);
        }

        // Check that all witnesses and values have been consumed.
        if !hash_witness.is_empty() {
            return Err(MerkleVerificationError::WitnessTooLong);
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

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MerkleVerificationError {
    #[error("Witness is too short.")]
    WitnessTooShort,
    #[error("Witness is too long.")]
    WitnessTooLong,
    #[error("Column values are too long.")]
    ColumnValuesTooLong,
    #[error("Column values are too short.")]
    ColumnValuesTooShort,
    #[error("Root mismatch.")]
    RootMismatch,
}
