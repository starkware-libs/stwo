use std::fmt;

use super::hasher::Hasher;
use crate::core::fields::Field;

/// A Merkle proof of queried indices.
/// Used for storing a merkle proof of a given tree and a set of queries.
/// A correctly generated decommitment should hold all the information needed to generate the root
/// of the tree, proving the queried values.
pub struct MixedDecommitment<F: Field, H: Hasher> {
    pub decommitment_layers: Vec<Vec<DecommitmentNode<F, H>>>,
}

impl<F: Field, H: Hasher> MixedDecommitment<F, H> {
    pub fn new(decommitment_layers: Vec<Vec<DecommitmentNode<F, H>>>) -> Self {
        Self {
            decommitment_layers,
        }
    }
}

/// A node in a Merkle tree's decommitment path.
/// Holds the input to the hash function, excluding information that can be calculated by a verifier
/// given the entire decommitment. Namely, one of the children hashes (if not a leaf) and the
/// injected elements if exists.
///
/// # Attributes
///
/// * `hash` - Optional value, hash of one of the node's children.
/// * `injected_elements` - Elements injected to the node.
/// * `position_in_layer` - The position of the provided hash in its layer - for debugging purposes,
///   can be deducted by a verifier.
pub struct DecommitmentNode<F: Field, H: Hasher> {
    pub hash: Option<H::Hash>,
    pub injected_elements: Vec<F>,
    pub position_in_layer: usize,
}

impl<F: Field, H: Hasher> fmt::Display for MixedDecommitment<F, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.decommitment_layers.last() {
            Some(_) => {
                for (i, layer) in self.decommitment_layers.iter().enumerate() {
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

impl<F: Field, H: Hasher> fmt::Display for DecommitmentNode<F, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&std::format!(
            "Position in Layer: {}, ",
            self.position_in_layer
        ))?;
        if let Some(hash) = self.hash {
            f.write_str(&std::format!("Hash: {}, ", hash))?;
        }
        f.write_str(&std::format!(
            " Injected Elements: {:?}",
            self.injected_elements
        ))?;
        f.write_str("\n")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::fields::m31::M31;

    #[test]
    fn display_test() {
        let path = super::MixedDecommitment::<M31, Blake3Hasher>::new(vec![
            vec![super::DecommitmentNode::<M31, Blake3Hasher> {
                hash: Some(Blake3Hasher::hash(b"a")),
                injected_elements: (0..3).map(M31::from_u32_unchecked).collect(),
                position_in_layer: 0,
            }],
            vec![
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    hash: Some(Blake3Hasher::hash(b"b")),
                    injected_elements: (3..6).map(M31::from_u32_unchecked).collect(),
                    position_in_layer: 1,
                },
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    hash: Some(Blake3Hasher::hash(b"c")),
                    injected_elements: (6..9).map(M31::from_u32_unchecked).collect(),
                    position_in_layer: 0,
                },
            ],
            vec![
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    hash: Some(Blake3Hasher::hash(b"d")),
                    injected_elements: Vec::new(),
                    position_in_layer: 2,
                },
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    hash: Some(Blake3Hasher::hash(b"e")),
                    injected_elements: Vec::new(),
                    position_in_layer: 1,
                },
            ],
        ]);

        println!("{}", path)
    }
}
