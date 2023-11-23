use std::fmt::{self, Debug};

use super::hasher::Hasher;

/// Merkle authentication path.
pub struct MerkleDecommitment<T: Sized + Debug, H: Hasher, const LEAF_SIZE: usize> {
    leaves: Vec<[T; LEAF_SIZE]>,
    layers: Vec<Vec<H::Hash>>,
}

impl<T: Sized + Debug, H: Hasher, const LEAF_SIZE: usize> MerkleDecommitment<T, H, LEAF_SIZE> {
    pub fn push_layer(&mut self, layer: Vec<H::Hash>) {
        self.layers.push(layer);
    }

    pub fn push_leaf(&mut self, leaf: [T; LEAF_SIZE]) {
        self.leaves.push(leaf);
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            leaves: Vec::new(),
            layers: Vec::with_capacity(capacity),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty() && self.leaves.is_empty()
    }

    pub fn len(&self) -> usize {
        self.layers.len() + 1
    }
}

impl<T: Sized + Debug, H: Hasher, const LEAF_SIZE: usize> fmt::Debug
    for MerkleDecommitment<T, H, LEAF_SIZE>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layers.last() {
            Some(_) => {
                self.leaves.iter().enumerate().for_each(|(i, leaf)| {
                    f.write_str(&std::format!("\nLeaf #[{:}]: ", i)).unwrap();
                    leaf.iter()
                        .for_each(|node| f.write_str(&std::format!("{:?} ", node)).unwrap());
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
    use super::MerkleDecommitment;
    use crate::commitment_scheme::blake3_hash::{Blake3Hash, Blake3Hasher};
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::fields::m31::M31;

    #[test]
    pub fn merkle_path_struct_test() {
        const LEAF_SIZE: usize = Blake3Hasher::BLOCK_SIZE_IN_BYTES / std::mem::size_of::<M31>();

        let mut m_path = MerkleDecommitment::<M31, Blake3Hasher, LEAF_SIZE>::with_capacity(3);
        let leaf: [M31; 16] = (0..16)
            .map(M31::from_u32_unchecked)
            .collect::<Vec<M31>>()
            .try_into()
            .unwrap();
        m_path.push_leaf(leaf);
        (0..3).for_each(|i| {
            let mut m_layer = Vec::<Blake3Hash>::with_capacity(3);
            (0..3).for_each(|_| m_layer.push(Blake3Hasher::hash(&[i as u8; LEAF_SIZE])));
            m_path.push_layer(m_layer);
        });
        m_path.push_layer(Vec::<Blake3Hash>::new());

        assert_eq!(m_path.leaves[0], leaf);
        assert_eq!(m_path.len(), 5);
        m_path.layers.iter().enumerate().for_each(|(i, layer)| {
            layer
                .iter()
                .for_each(|node| assert_eq!(*node, Blake3Hasher::hash(&[i as u8; LEAF_SIZE])));
        });
    }
}
