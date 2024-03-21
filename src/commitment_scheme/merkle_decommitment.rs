use std::fmt::{self, Display};
use std::iter::Peekable;

use itertools::Itertools;

use super::hasher::Hasher;
use crate::core::fields::IntoSlice;

/// Merkle proof of queried indices.
/// Used for storing a merkle proof of a given tree and a set of queries.
/// # Attributes
/// * `leaf_blocks` - The blocks of the bottom layer of the tree.
/// * `layers` - Internal nodes(hashes) of a specific layer in the tree. nodes that are not in a
///   queried path, or nodes with both children in the queried path are excluded.
/// * `n_rows_in_leaf_block` - The number of trace-rows packed in each leaf block.
// TODO(Ohad): derive Debug.
#[derive(Default, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        deserialize = "<H as Hasher>::Hash: serde::Deserialize<'de>, T: serde::Deserialize<'de>",
        serialize = "<H as Hasher>::Hash: serde::Serialize, T: serde::Serialize"
    ))
)]
pub struct MerkleDecommitment<T: Sized + Display, H: Hasher> {
    pub leaf_blocks: Vec<Vec<T>>,
    pub layers: Vec<Vec<H::Hash>>,
    pub n_rows_in_leaf_block: usize,
    queries: Vec<usize>,
}

impl<T: Sized + Display + Copy, H: Hasher> MerkleDecommitment<T, H>
where
    T: IntoSlice<H::NativeType>,
{
    pub fn new(
        leaf_blocks: Vec<Vec<T>>,
        layers: Vec<Vec<H::Hash>>,
        n_rows_in_leaf_block: usize,
        queries: Vec<usize>,
    ) -> Self {
        Self {
            leaf_blocks,
            layers,
            n_rows_in_leaf_block,
            queries,
        }
    }

    pub fn height(&self) -> usize {
        self.layers.len() + 1
    }

    // TODO(Ohad): Implement more verbose error handling.
    /// Verifies the decommitment against a given root. Queries are assumed to be sorted.
    pub fn verify(&self, root: H::Hash, queries: &[usize]) -> bool {
        let leaf_block_queries = queries
            .iter()
            .map(|q| q / self.n_rows_in_leaf_block)
            .dedup()
            .collect::<Vec<usize>>();
        assert_eq!(self.leaf_blocks.len(), leaf_block_queries.len());

        let mut curr_hashes = self
            .leaf_blocks
            .iter()
            .map(|leaf_block| H::hash(<T as IntoSlice<H::NativeType>>::into_slice(leaf_block)))
            .collect::<Vec<H::Hash>>();

        let mut layer_queries = leaf_block_queries.clone();
        for layer in self.layers.iter() {
            let mut next_layer_hashes = Vec::<H::Hash>::new();
            let mut query_iter = layer_queries.iter().enumerate().peekable();
            let mut layer_iter = layer.iter();

            while let Some((i, q)) = query_iter.next() {
                let mut f = || -> Option<_> {
                    if *q % 2 != 0 {
                        // Right child.
                        return Some(H::concat_and_hash(layer_iter.next()?, curr_hashes.get(i)?));
                    }
                    match query_iter.peek() {
                        Some((_, next_q)) if *q + 1 == **next_q => {
                            query_iter.next();
                            Some(H::concat_and_hash(
                                curr_hashes.get(i)?,
                                curr_hashes.get(i + 1)?,
                            ))
                        }
                        _ => Some(H::concat_and_hash(curr_hashes.get(i)?, layer_iter.next()?)),
                    }
                };
                next_layer_hashes.push(f().expect("Error verifying proof!"));
            }
            assert!(layer_iter.next().is_none());
            curr_hashes = next_layer_hashes;
            layer_queries = layer_queries
                .iter()
                .map(|q| q / 2)
                .dedup()
                .collect::<Vec<usize>>();
        }
        assert_eq!(
            layer_queries.into_iter().collect::<Vec<usize>>(),
            vec![0_usize]
        );
        assert_eq!(curr_hashes.len(), 1);
        curr_hashes[0].into() == root.into()
    }

    pub fn values(&self) -> impl Iterator<Item = Vec<T>> + '_ {
        QueriedValuesIterator {
            query_iterator: self.queries.iter(),
            leaf_block_iterator: self.leaf_blocks.iter().peekable(),
            current_leaf_block_index: self.queries[0] / self.n_rows_in_leaf_block,
            n_elements_in_row: self.leaf_blocks[0].len() / self.n_rows_in_leaf_block,
            n_rows_in_leaf_block: self.n_rows_in_leaf_block,
        }
    }
}

pub struct QueriedValuesIterator<'a, T: Sized + Display> {
    query_iterator: std::slice::Iter<'a, usize>,
    leaf_block_iterator: Peekable<std::slice::Iter<'a, Vec<T>>>,
    current_leaf_block_index: usize,
    n_elements_in_row: usize,
    n_rows_in_leaf_block: usize,
}

impl<'a, T: Sized + Display + Clone> Iterator for QueriedValuesIterator<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.query_iterator.next() {
            Some(query) => {
                let leaf_block_index = self.get_leaf_block_index(*query);
                if leaf_block_index != self.current_leaf_block_index {
                    self.leaf_block_iterator.next();
                    self.current_leaf_block_index = leaf_block_index;
                }
                let row_start_index = (query % self.n_rows_in_leaf_block) * self.n_elements_in_row;
                let row_end_index = row_start_index + self.n_elements_in_row;
                Some(
                    self.leaf_block_iterator.peek().unwrap().to_vec()
                        [row_start_index..row_end_index]
                        .to_vec(),
                )
            }
            None => None,
        }
    }
}

impl<'a, T: Sized + Display> QueriedValuesIterator<'a, T> {
    pub fn get_leaf_block_index(&self, query: usize) -> usize {
        query / self.n_rows_in_leaf_block
    }
}

impl<T: Sized + Display, H: Hasher> fmt::Display for MerkleDecommitment<T, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layers.last() {
            Some(_) => {
                self.leaf_blocks.iter().enumerate().for_each(|(i, leaf)| {
                    f.write_str(&std::format!("\nLeaf #[{:}]: ", i)).unwrap();
                    leaf.iter()
                        .for_each(|node| f.write_str(&std::format!("{} ", node)).unwrap());
                });
                for (i, layer) in self.layers.iter().enumerate().take(self.layers.len()) {
                    f.write_str(&std::format!("\nLayer #[{}]:", i))?;
                    for (j, node) in layer.iter().enumerate() {
                        f.write_str(&std::format!("\n\tNode #[{}]: {}", j, node))?;
                    }
                }
            }
            None => f.write_str("Empty Path!")?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_tree::MerkleTree;
    use crate::commitment_scheme::utils::tests::generate_test_queries;
    use crate::commitment_scheme::utils::ColumnArray;
    use crate::core::fields::m31::M31;

    #[test]
    pub fn verify_test() {
        let trace: ColumnArray<M31> = vec![(0..4096).map(M31::from_u32_unchecked).collect(); 7];
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(trace);
        let queries = generate_test_queries(100, 4096);
        let decommitment = tree.generate_decommitment(queries.clone());

        assert!(decommitment.verify(tree.root(), &queries));
    }

    #[test]
    pub fn verify_false_proof_test() {
        let trace_column_length = 1 << 12;
        let trace: ColumnArray<M31> = vec![
            (0..trace_column_length)
                .map(M31::from_u32_unchecked)
                .collect();
            4
        ];
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(trace);
        let queries = generate_test_queries(10, trace_column_length as usize);
        let mut wrong_internal_node_decommitment = tree.generate_decommitment(queries.clone());
        let mut wrong_leaf_block_decommitment = tree.generate_decommitment(queries.clone());

        wrong_internal_node_decommitment.layers[0][0] = Blake3Hasher::hash(&[0]);
        wrong_leaf_block_decommitment.leaf_blocks[0][0] += M31::from_u32_unchecked(1);

        assert!(
            !wrong_internal_node_decommitment.verify(tree.root(), &queries),
            "Wrong internal node decommitment passed!"
        );
        assert!(
            !wrong_leaf_block_decommitment.verify(tree.root(), &queries),
            "Wrong leaf block decommitment passed!"
        );
    }

    #[test]
    fn values_test() {
        let trace_column_length = 1 << 6;
        let trace_column = (0..trace_column_length)
            .map(M31::from_u32_unchecked)
            .collect::<Vec<M31>>();
        let reversed_trace_column = trace_column.iter().rev().cloned().collect::<Vec<M31>>();
        let trace: ColumnArray<M31> = vec![trace_column, reversed_trace_column];
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(trace.clone());
        let random_queries = generate_test_queries(10, trace_column_length as usize);
        let test_skip_queries = vec![17, 50];
        let random_query_decommitment = tree.generate_decommitment(random_queries.clone());
        let test_skip_decommitment = tree.generate_decommitment(test_skip_queries.clone());

        assert!(random_queries
            .iter()
            .zip(random_query_decommitment.values())
            .all(|(q, v)| v == vec![trace[0][*q], trace[1][*q]]));
        assert!(test_skip_queries
            .iter()
            .zip(test_skip_decommitment.values())
            .all(|(q, v)| v == vec![trace[0][*q], trace[1][*q]]));
    }
}
