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
        let mut n_columns_per_log_size = BTreeMap::<u32, usize>::new();
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
    /// # Arguments
    ///
    /// * `queries_per_log_size` - A map from log_size to a vector of queries for columns of that
    ///  log_size.
    /// * `queried_values` - A vector of vectors of queried values. For each column, there is a
    /// vector of queried values to that column.
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
        queries_per_log_size: &BTreeMap<u32, Vec<usize>>,
        queried_values_by_layer: Vec<Vec<BaseField>>,
        decommitment: MerkleDecommitment<H>,
    ) -> Result<(), MerkleVerificationError> {
        let Some(max_log_size) = self.column_log_sizes.iter().max() else {
            return Ok(());
        };

        let mut queried_values_by_layer_iter = queried_values_by_layer.into_iter().rev();

        // Prepare read buffers.

        let mut hash_witness = decommitment.hash_witness.into_iter();
        let mut column_witness = decommitment.column_witness.into_iter();

        let mut last_layer_hashes: Option<Vec<(usize, H::Hash)>> = None;
        for layer_log_size in (0..=*max_log_size).rev() {
            // Prepare read buffer for queried values to the current layer.
            let mut layer_queried_values = queried_values_by_layer_iter.next().unwrap().into_iter();

            let n_columns_in_layer = self
                .n_columns_per_log_size
                .get(&layer_log_size)
                .copied()
                .unwrap_or(0);

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
                let node_values = if layer_column_queries.next_if_eq(&node_index).is_some() {
                    (&mut layer_queried_values)
                        .take(n_columns_in_layer)
                        .collect_vec()
                } else {
                    // Otherwise, read them from the witness.
                    (&mut column_witness).take(n_columns_in_layer).collect_vec()
                };

                assert_eq!(node_values.len(), n_columns_in_layer);
                if node_values.len() != n_columns_in_layer {
                    return Err(MerkleVerificationError::WitnessTooShort);
                }

                layer_total_queries.push((node_index, H::hash_node(node_hashes, &node_values)));
            }

            if !layer_queried_values.is_empty() {
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
