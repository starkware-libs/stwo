use std::fmt;

use super::hasher::Hasher;
use crate::core::fields::Field;

/// A Merkle proof of queried indices.
/// Used for storing a all the paths from the query leaves to the root.
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
/// given the entire decommitment.
/// Hashes - one or less of the children hashes(zero happens at the leaves or at the joining point
/// of 2 coinciding paths). injected elements - if exists.
///
/// # Attributes
///
/// * `hash` - Optional value, hash of one of the node's children.
/// * `injected_elements` - Elements injected to the node.
/// * `position_in_layer` - The position of the provided hash in the layer - for debugging purposes,
///   can be deducted by a verifier.
pub struct DecommitmentNode<F: Field, H: Hasher> {
    pub left_hash: Option<H::Hash>,
    pub right_hash: Option<H::Hash>,
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
            "Node position in Layer: {}, ",
            self.position_in_layer
        ))?;
        if let Some(hash) = self.left_hash {
            f.write_str(&std::format!("Left hash: {}, ", hash))?;
        }
        if let Some(hash) = self.right_hash {
            f.write_str(&std::format!("Right hash: {}, ", hash))?;
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
                left_hash: Some(Blake3Hasher::hash(b"a")),
                right_hash: None,
                injected_elements: (0..3).map(M31::from_u32_unchecked).collect(),
                position_in_layer: 0,
            }],
            vec![
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    right_hash: Some(Blake3Hasher::hash(b"b")),
                    left_hash: None,
                    injected_elements: (3..6).map(M31::from_u32_unchecked).collect(),
                    position_in_layer: 1,
                },
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    left_hash: Some(Blake3Hasher::hash(b"c")),
                    right_hash: None,
                    injected_elements: (6..9).map(M31::from_u32_unchecked).collect(),
                    position_in_layer: 0,
                },
            ],
        ]);
        println!("{}", path)
    }
}
