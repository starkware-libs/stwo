use std::cmp::Reverse;

use itertools::Itertools;

use super::ops::{MerkleHasher, MerkleOps};
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;

pub struct MerkleProver<B: MerkleOps<H>, H: MerkleHasher> {
    /// Layers of the Merkle tree.
    /// The first layer is the largest column.
    /// The last layer is the root.
    /// Layer n holds the 2^n hashes of the previous layer if exists, and all the columns of size
    /// 2^n.
    pub layers: Vec<Col<B, H::Hash>>,
}
/// The MerkleProver struct represents a prover for a Merkle commitment scheme.
/// It is generic over the types `B` and `H`, which represent the Merkle operations and Merkle
/// hasher respectively.
impl<B: MerkleOps<H>, H: MerkleHasher> MerkleProver<B, H> {
    /// Commits to columns.
    /// Columns must be of power of 2 sizes and sorted in descending order.
    ///
    /// # Arguments
    ///
    /// * `columns` - A vector of references to columns.
    ///
    /// # Panics
    ///
    /// This function will panic if the columns are not sorted in descending order or if the columns
    /// vector is empty.
    ///
    /// # Returns
    ///
    /// A new instance of `MerkleProver` with the committed layers.
    pub fn commit(columns: Vec<&Col<B, BaseField>>) -> Self {
        // Check that columns are of descending order.
        assert!(!columns.is_empty());
        assert!(columns.is_sorted_by_key(|c| Reverse(c.len())));

        let mut columns = &mut columns.into_iter().peekable();
        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();

        let max_log_size = columns.peek().unwrap().len().ilog2();
        for log_size in (0..=max_log_size).rev() {
            // Take columns of the current log_size.
            let layer_columns = (&mut columns)
                .take_while(|column| column.len().ilog2() == log_size)
                .collect_vec();

            layers.push(B::commit_on_layer(log_size, layers.last(), &layer_columns));
        }
        Self { layers }
    }

    /// Decommits to columns on the given queries.
    /// Queries are given as indices to the largest column.
    ///
    /// # Arguments
    ///
    /// * `queries` - A vector of query indices to the largest column.
    ///
    /// # Returns
    ///
    /// A `Decommitment` struct containing the witness.
    pub fn decommit(&self, mut queries: Vec<usize>) -> Decommitment<H> {
        let mut witness = Vec::new();
        for layer in &self.layers[..self.layers.len() - 1] {
            let mut queries_iter = queries.into_iter().peekable();

            // Propagate queries and hashes to the next layer.
            let mut next_queries = Vec::new();
            while let Some(query) = queries_iter.next() {
                next_queries.push(query / 2);
                if queries_iter.next_if_eq(&(query ^ 1)).is_some() {
                    continue;
                }
                witness.push(layer.at(query ^ 1));
            }
            queries = next_queries;
        }
        Decommitment { witness }
    }

    pub fn root(&self) -> H::Hash {
        self.layers.last().unwrap().at(0)
    }
}

#[derive(Debug)]
pub struct Decommitment<H: MerkleHasher> {
    pub witness: Vec<H::Hash>,
}
