use std::collections::BTreeSet;
use std::fmt::{self, Display};

use super::hasher::Hasher;
use crate::commitment_scheme::utils::to_byte_slice;

/// Merkle authentication path.
/// Used for storing a merkle proof of a given tree and a set of queries.
// TODO(Ohad): write verify function.
// TODO(Ohad): derive Debug.
#[derive(Default)]
pub struct MerkleDecommitment<T: Sized + Display, H: Hasher> {
    pub leaves: Vec<Vec<T>>,
    pub layers: Vec<Vec<H::Hash>>,
}

impl<T: Sized + Display, H: Hasher> MerkleDecommitment<T, H> {
    pub fn new() -> Self {
        Self {
            leaves: Vec::new(),
            layers: Vec::new(),
        }
    }

    pub fn height(&self) -> usize {
        self.layers.len() + 1
    }

    pub fn verify(
        &self,
        root: H::Hash,
        queries: BTreeSet<usize>,
        n_rows_in_leaf_block: usize,
    ) -> bool {
        let leaf_block_queries = queries
            .iter()
            .map(|q| q / n_rows_in_leaf_block)
            .collect::<BTreeSet<usize>>();
        assert_eq!(self.leaves.len(), leaf_block_queries.len());

        let mut curr_hashes = self
            .leaves
            .iter()
            .map(|leaf_block| H::hash(to_byte_slice(leaf_block)))
            .collect::<Vec<H::Hash>>();

        let mut layer_queries = leaf_block_queries.clone();
        for layer in self.layers.iter() {
            let mut next_layer_hashes = Vec::<H::Hash>::new();
            let mut query_iter = layer_queries.iter().enumerate().peekable();
            let mut layer_iter = layer.iter();
            while let Some((i, q)) = query_iter.next() {
                next_layer_hashes.push(if q % 2 != 0 {
                    H::concat_and_hash(
                        layer_iter.next().expect("Error advancing layer iterator!"),
                        &curr_hashes[i],
                    )
                } else {
                    match query_iter.peek() {
                        Some((_, next_q)) => {
                            if q ^ 1 == **next_q {
                                query_iter.next();
                                H::concat_and_hash(&curr_hashes[i], &curr_hashes[i + 1])
                            } else {
                                H::concat_and_hash(
                                    &curr_hashes[i],
                                    layer_iter.next().expect("Error advancing layer iterator!"),
                                )
                            }
                        }
                        None => H::concat_and_hash(
                            &curr_hashes[i],
                            layer_iter.next().expect("Error advancing layer iterator!"),
                        ),
                    }
                })
            }
            assert!(layer_iter.next().is_none());
            curr_hashes = next_layer_hashes;
            layer_queries = layer_queries
                .iter()
                .map(|q| q / 2)
                .collect::<BTreeSet<usize>>();
        }

        curr_hashes[0].into() == root.into()
    }
}

impl<T: Sized + Display, H: Hasher> fmt::Display for MerkleDecommitment<T, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layers.last() {
            Some(_) => {
                self.leaves.iter().enumerate().for_each(|(i, leaf)| {
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
    use crate::commitment_scheme::merkle_tree::MerkleTree;
    use crate::commitment_scheme::utils::ColumnArray;
    use crate::core::fields::m31::M31;

    #[test]
    pub fn verify_test() {
        let trace: ColumnArray<M31> = vec![(0..4096).map(M31::from_u32_unchecked).collect(); 7];
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(trace);
        let queries: BTreeSet<usize> = (0..100).map(|_| thread_rng().gen_range(0..4096)).collect();
        let decommitment = tree.generate_decommitment(queries.clone());

        assert!(decommitment.verify(tree.root(), queries, tree.bottom_layer_n_rows_in_node));
    }

    #[test]
    #[should_panic]
    pub fn verify_false_proof_test() {
        let trace: ColumnArray<M31> = vec![(0..4096).map(M31::from_u32_unchecked).collect(); 4];
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(trace);
        let queries: BTreeSet<usize> = (0..100).map(|_| thread_rng().gen_range(0..128)).collect();
        let mut decommitment = tree.generate_decommitment(queries.clone());

        decommitment.leaves[0][0] += M31::from_u32_unchecked(1);

        assert!(decommitment.verify(tree.root(), queries, tree.bottom_layer_n_rows_in_node));
    }
}
