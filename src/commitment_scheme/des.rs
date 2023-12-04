use std::fmt::Display;

use super::hasher::Hasher;
use super::merkle_decommitment::MerkleDecommitment;
use crate::core::fields::Field;

pub type ColumnRefArray<'a, F> = Vec<&'a Vec<F>>;
pub type ColumnsToInject<'a, F> = Vec<ColumnRefArray<'a, F>>;

/// Merkle tree.
/// F for now should be used as basefild, and extension fields should implement Into<F> of some
/// sort.
pub struct MerkleTree<'a, F: Field, H: Hasher> {
    _input: MerkleTreeInput<'a, F>,
    _hash_layers: Vec<SubTreeLayer<H>>,
}

impl<'a, F: Field + Display, H: Hasher> MerkleTree<'a, F, H>
where
    F: Into<Box<[H::NativeType]>>, 

{
    pub fn get(&self, layer_depth: usize, idx: usize) -> H::Hash {
        todo!()
    }

    pub fn commit(_config: MerkleTreeInput<'a, F>) -> Self {
        todo!()
    }
    pub fn decommit(&self) -> MerkleDecommitment<F, H> {
        todo!()
    }
    pub fn root(&self) -> H::Hash {
        todo!()
    }

    pub fn height(&self) -> usize {
        self._hash_layers.iter().fold(0, |acc, layer| acc + layer.width)
    }
}

// Post Order layer 
#[derive(Default)]
pub struct SubTreeLayer<H: Hasher> {
    pub data: Vec<H::Hash>,
    depth: usize,
    width: usize,
}

impl<H: Hasher> SubTreeLayer<H> {
    pub fn new(depth: usize, width: usize, n_trees: usize) -> Self {
        Self {
            data: Vec::with_capacity((1 << (depth + width)) -  (1<<depth) + 1),
            depth,
            width,
        }
    }

    pub fn get(&self, idx: usize) -> H::Hash {
        self.data[idx]
    }

    pub fn get_layer(&self,tree_idx:usize, layer_idx: usize) -> &[H::Hash] {
        let len = 1 << layer_idx;
        let idx = (self.data.capacity() * tree_idx / self.n_trees) + (len - 1); 
        &self.data[idx..idx+len]
    }


    pub fn get_mut_layer(&mut self,tree_idx:usize, layer_idx: usize) -> &mut [H::Hash] {
        let len = 1 << layer_idx;
        let idx = (self.data.capacity() * tree_idx / self.n_trees) + (len - 1); 
        &mut self.data[idx..idx+len]
    }
    
}

/// Merkle tree configuration.
/// A map from the depth of the tree requested to be injected to the to-be-injected columns.
/// Info about the tree's position in the containing tree for recursive hashing.
pub struct MerkleTreeInput<'a, F: Field>(ColumnsToInject<'a, F>);

impl<'a, F: Field> MerkleTreeInput<'a, F> {
    pub fn new(max_depth: usize) -> Self {
        Self(Vec::with_capacity(max_depth))
    }
    pub fn insert(&mut self, depth: usize, column: &'a Vec<F>) {
        assert_ne!(depth, 0, "Injectiing to layer 0 has no effect!");
        assert!(column.len().is_power_of_two());
        assert!(column.len() < 2usize.pow(depth as u32));
        assert!(depth <= self.0.len());

        self.0[depth].push(column);
    }

    pub fn get(&'a self, depth: usize) -> Option<&'a Vec<&'a Vec<F>>> {
        self.0.get(depth)
    }
    pub fn get_columns() {
        todo!()
    }
    pub fn verify() {
        todo!()
    }
    pub fn max_depth() {
        todo!()
    }
}

