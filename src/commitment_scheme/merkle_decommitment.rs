use std::fmt::{self, Display};

use super::hasher::Hasher;

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
