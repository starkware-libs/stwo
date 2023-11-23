use std::fmt::{self, Display};

use super::hasher::Hasher;

/// Merkle authentication path.
/// Used for storing a merkle proof of a given tree and a set of queries.
// TODO(Ohad): write verify function.
// TODO(Ohad): derive Debug.
#[derive(Default)]
pub struct MerkleDecommitment<T: Sized + Display, H: Hasher, const LEAF_SIZE: usize> {
    pub leaves: Vec<[T; LEAF_SIZE]>,
    pub layers: Vec<Vec<H::Hash>>,
}

impl<T: Sized + Display, H: Hasher, const LEAF_SIZE: usize> MerkleDecommitment<T, H, LEAF_SIZE> {
    pub fn new() -> Self {
        Self {
            leaves: Vec::new(),
            layers: Vec::new(),
        }
    }

    pub fn height(&self) -> usize {
        self.layers.len() + 1
    }
}

impl<T: Sized + Display, H: Hasher, const LEAF_SIZE: usize> fmt::Display
    for MerkleDecommitment<T, H, LEAF_SIZE>
{
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
    use super::MerkleDecommitment;
    use crate::commitment_scheme::blake3_hash::{Blake3Hash, Blake3Hasher};
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::fields::m31::M31;

    #[test]
    pub fn merkle_path_struct_test() {
        const LEAF_SIZE: usize = Blake3Hasher::BLOCK_SIZE_IN_BYTES / std::mem::size_of::<M31>();

        let mut m_path = MerkleDecommitment::<M31, Blake3Hasher, LEAF_SIZE>::new();
        let leaf: [M31; 16] = (0..16)
            .map(M31::from_u32_unchecked)
            .collect::<Vec<M31>>()
            .try_into()
            .unwrap();
        m_path.leaves.push(leaf);
        (0..3).for_each(|i| {
            let mut m_layer = Vec::<Blake3Hash>::with_capacity(3);
            (0..3).for_each(|_| m_layer.push(Blake3Hasher::hash(&[i as u8; LEAF_SIZE])));
            m_path.layers.push(m_layer);
        });
        m_path.layers.push(Vec::<Blake3Hash>::new());

        assert_eq!(m_path.leaves[0], leaf);
        assert_eq!(m_path.height(), 5);
        m_path.layers.iter().enumerate().for_each(|(i, layer)| {
            layer
                .iter()
                .for_each(|node| assert_eq!(*node, Blake3Hasher::hash(&[i as u8; LEAF_SIZE])));
        });
    }
}
