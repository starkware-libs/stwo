use std::collections::BTreeSet;
use std::fmt::{self, Display};

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
#[derive(Default)]
pub struct MerkleDecommitment<T: Sized + Display, H: Hasher> {
    pub leaf_blocks: Vec<Vec<T>>,
    pub layers: Vec<Vec<H::Hash>>,
    pub n_rows_in_leaf_block: usize,
    queried_leaf_block_indices: BTreeSet<usize>,
}

impl<T: Sized + Display + Copy, H: Hasher> MerkleDecommitment<T, H>
where
    T: IntoSlice<H::NativeType>,
{
    pub fn new(
        leaf_blocks: Vec<Vec<T>>,
        layers: Vec<Vec<H::Hash>>,
        n_rows_in_leaf_block: usize,
        queried_block_indices: BTreeSet<usize>,
    ) -> Self {
        Self {
            leaf_blocks,
            layers,
            n_rows_in_leaf_block,
            queried_leaf_block_indices: queried_block_indices,
        }
    }

    pub fn height(&self) -> usize {
        self.layers.len() + 1
    }

    // TODO(Ohad): Implement more verbose error handling.
    pub fn verify(&self, root: H::Hash, queries: BTreeSet<usize>) -> bool {
        let leaf_block_queries = queries
            .iter()
            .map(|q| q / self.n_rows_in_leaf_block)
            .collect::<BTreeSet<usize>>();
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
                .collect::<BTreeSet<usize>>();
        }
        assert_eq!(
            layer_queries.into_iter().collect::<Vec<usize>>(),
            vec![0_usize]
        );
        assert_eq!(curr_hashes.len(), 1);
        curr_hashes[0].into() == root.into()
    }

    pub fn get_values_at(&self, query: usize) -> Vec<T> {
        // Get containing block.
        let leaf_block_index = query / self.n_rows_in_leaf_block;
        assert!(
            self.queried_leaf_block_indices.contains(&leaf_block_index),
            "Query {} not in decommitment!",
            query
        );
        let index_in_decommitment = self
            .queried_leaf_block_indices
            .iter()
            .position(|&i| i == leaf_block_index)
            .unwrap();
        let leaf_block = self.leaf_blocks.get(index_in_decommitment).unwrap();

        // Extract values.
        let row_length = leaf_block.len() / self.n_rows_in_leaf_block;
        let row_start_index = (query % self.n_rows_in_leaf_block) * row_length;
        leaf_block[row_start_index..row_start_index + row_length].to_vec()
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
    use std::collections::BTreeSet;

    use rand::{thread_rng, Rng};

    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_tree::MerkleTree;
    use crate::commitment_scheme::utils::ColumnArray;
    use crate::core::fields::m31::M31;

    #[test]
    pub fn verify_test() {
        let trace: ColumnArray<M31> = vec![(0..4096).map(M31::from_u32_unchecked).collect(); 7];
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(trace);
        let queries: BTreeSet<usize> = (0..100).map(|_| thread_rng().gen_range(0..4096)).collect();
        let decommitment = tree.generate_decommitment(queries.clone());

        assert!(decommitment.verify(tree.root(), queries));
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
        let queries: BTreeSet<usize> = (0..10)
            .map(|_| thread_rng().gen_range(0..trace_column_length as usize))
            .collect();
        let mut wrong_internal_node_decommitment = tree.generate_decommitment(queries.clone());
        let mut wrong_leaf_block_decommitment = tree.generate_decommitment(queries.clone());

        wrong_internal_node_decommitment.layers[0][0] = Blake3Hasher::hash(&[0]);
        wrong_leaf_block_decommitment.leaf_blocks[0][0] += M31::from_u32_unchecked(1);

        assert!(
            !wrong_internal_node_decommitment.verify(tree.root(), queries.clone(),),
            "Wrong internal node decommitment passed!"
        );
        assert!(
            !wrong_leaf_block_decommitment.verify(tree.root(), queries,),
            "Wrong leaf block decommitment passed!"
        );
    }

    #[test]
    fn get_values_at_test() {
        let trace_column_length = 1 << 6;
        let trace: ColumnArray<M31> = vec![
            (0..trace_column_length)
                .map(M31::from_u32_unchecked)
                .collect();
            4
        ];
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(trace.clone());
        let queries: BTreeSet<usize> = (0..10)
            .map(|_| thread_rng().gen_range(0..trace_column_length as usize))
            .collect();
        let decommitment = tree.generate_decommitment(queries.clone());
        let first_query = *queries.iter().next().unwrap();

        let values_at_first_query = decommitment.get_values_at(first_query);

        assert_eq!(
            values_at_first_query,
            vec![M31::from_u32_unchecked(first_query as u32); 4]
        );
    }

    #[test]
    #[should_panic]
    fn get_values_at_unincluded_blocks_test() {
        let trace_column_length = 1 << 6;
        let trace: ColumnArray<M31> = vec![
            (0..trace_column_length)
                .map(M31::from_u32_unchecked)
                .collect();
            4
        ];
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(trace.clone());
        let queries: BTreeSet<usize> = (0..10).collect();
        let decommitment = tree.generate_decommitment(queries.clone());

        decommitment.get_values_at(63);
    }
}
