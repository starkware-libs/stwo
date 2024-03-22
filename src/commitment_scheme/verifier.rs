use std::cmp::Reverse;
use std::iter::Peekable;

use itertools::Itertools;
use thiserror::Error;

use super::hasher::Hasher;
use super::ops::MerkleHasher;
use super::prover::MerkleDecommitment;
use crate::core::fields::m31::BaseField;

// TODO(spapini): This struct is not necessary. Make it a function on decommitment?
pub struct MerkleTreeVerifier<H: MerkleHasher> {
    pub root: H::Hash,
}
impl<H: MerkleHasher> MerkleTreeVerifier<H> {
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
        queries: Vec<usize>,
        values: Vec<(u32, Vec<BaseField>)>,
        decommitment: MerkleDecommitment<H>,
    ) -> Result<(), MerkleVerificationError> {
        // Check that columns are of descending order.
        assert!(values.is_sorted_by_key(|(log_size, _)| Reverse(log_size)));

        // Compute root from decommitment.
        let mut verifier = MerkleVerifier::<H> {
            witness: decommitment.witness.into_iter(),
            column_values: values.into_iter().peekable(),
            layer_column_values: Vec::new(),
        };
        let computed_root = verifier.compute_root_from_decommitment(queries)?;

        // Check that all witnesses and values have been consumed.
        if !verifier.witness.is_empty() {
            return Err(MerkleVerificationError::WitnessTooLong);
        }
        if !verifier.column_values.is_empty() {
            return Err(MerkleVerificationError::ColumnValuesTooLong);
        }

        // Check that the computed root matches the expected root.
        if computed_root != self.root {
            return Err(MerkleVerificationError::RootMismatch);
        }

        Ok(())
    }
}

/// A helper struct for verifying a [MerkleDecommitment].
struct MerkleVerifier<H: MerkleHasher> {
    /// A queue for consuming the next hash witness from the decommitment.
    witness: std::vec::IntoIter<<H as Hasher>::Hash>,
    /// A queue for consuming the next claimed values for each column.
    column_values: Peekable<std::vec::IntoIter<(u32, Vec<BaseField>)>>,
    /// A queue for consuming the next claimed values for each column in the current layer.
    layer_column_values: Vec<std::vec::IntoIter<BaseField>>,
}
impl<H: MerkleHasher> MerkleVerifier<H> {
    /// Computes the root hash of a Merkle tree from the decommitment information.
    ///
    /// # Arguments
    ///
    /// * `queries` - A vector of query indices to the largest column.
    ///
    /// # Returns
    ///
    /// Returns the computed root hash of the Merkle tree.
    ///
    /// # Errors
    ///
    /// Returns a `MerkleVerificationError` if there is an error during the computation.
    pub fn compute_root_from_decommitment(
        &mut self,
        queries: Vec<usize>,
    ) -> Result<H::Hash, MerkleVerificationError> {
        let max_log_size = self.column_values.peek().unwrap().0;
        assert!(*queries.iter().max().unwrap() < 1 << max_log_size);

        // A sequence of queries to the current layer.
        // Each query is a pair of the query index and the known hashes of the children, if any.
        // The known hashes are represented as ChildrenHashesAtQuery.
        // None on the largest layer, or a pair of Option<Hash>, for the known hashes of the left
        // and right children.
        let mut queries = queries.into_iter().map(|query| (query, None)).collect_vec();

        for layer_log_size in (0..=max_log_size).rev() {
            // Take values for columns of the current log_size.
            self.layer_column_values = (&mut self.column_values)
                .take_while(|(log_size, _)| *log_size == layer_log_size)
                .map(|(_, values)| values.into_iter())
                .collect();

            // Compute node hashes for the current layer.
            let mut hashes_at_layer = queries
                .into_iter()
                .map(|(index, children_hashes)| (index, self.compute_node_hash(children_hashes)))
                .peekable();

            // Propagate queries and hashes to the next layer.
            let mut next_queries = Vec::new();
            while let Some((index, node_hash)) = hashes_at_layer.next() {
                // If the sibling hash is known, propagate it to the next layer.
                if let Some((_, sibling_hash)) =
                    hashes_at_layer.next_if(|(next_index, _)| *next_index == index ^ 1)
                {
                    next_queries.push((index / 2, Some((Some(node_hash?), Some(sibling_hash?)))));
                    continue;
                }
                // Otherwise, propagate the node hash to the next layer, in the correct direction.
                if index & 1 == 0 {
                    next_queries.push((index / 2, Some((Some(node_hash?), None))));
                } else {
                    next_queries.push((index / 2, Some((None, Some(node_hash?)))));
                }
            }
            queries = next_queries;

            // Check that all layer_column_values have been consumed.
            if self
                .layer_column_values
                .iter_mut()
                .any(|values| values.next().is_some())
            {
                return Err(MerkleVerificationError::ColumnValuesTooLong);
            }
        }

        assert_eq!(queries.len(), 1);
        Ok(queries.pop().unwrap().1.unwrap().0.unwrap())
    }

    fn compute_node_hash(
        &mut self,
        children_hashes: ChildrenHashesAtQuery<H>,
    ) -> Result<H::Hash, MerkleVerificationError> {
        // For each child with an unknown hash, fill it from the witness queue.
        let hashes_part = children_hashes
            .map(|(l, r)| {
                let l = l
                    .or_else(|| self.witness.next())
                    .ok_or(MerkleVerificationError::WitnessTooShort)?;
                let r = r
                    .or_else(|| self.witness.next())
                    .ok_or(MerkleVerificationError::WitnessTooShort)?;
                Ok((l, r))
            })
            .transpose()?;
        // Fill the column values from the layer_column_values queue.
        let column_values = self
            .layer_column_values
            .iter_mut()
            .map(|values| {
                values
                    .next()
                    .ok_or(MerkleVerificationError::ColumnValuesTooShort)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Hash the node.
        Ok(H::hash_node(hashes_part, &column_values))
    }
}

type ChildrenHashesAtQuery<H> = Option<(Option<<H as Hasher>::Hash>, Option<<H as Hasher>::Hash>)>;

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
