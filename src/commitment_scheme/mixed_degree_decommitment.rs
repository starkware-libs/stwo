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
/// Hashes - two or less of the children hashes(zero happens at the leaves or at the joining point
/// of 2 coinciding paths, two happens when a node is directly queried and is not a parent).
/// injected elements - if exists.
///
/// # Attributes
///
/// * `left_hash`, 'right_hash' - Optional values, hash of one of the node's children.
/// * `witness_elements` - Elements injected to the node that are part of the witness.
/// * 'DebugInfo' - Debug information, only available in debug builds.
///     * `d.queried_values` - Elements injected to the node that are not part of the witness.
///     * `d.position_in_layer` - The position of the provided hash in the layer - for debugging
///       purposes,
pub struct DecommitmentNode<F: Field, H: Hasher> {
    pub left_hash: Option<H::Hash>,
    pub right_hash: Option<H::Hash>,
    pub witness_elements: Vec<F>,
    pub d: DebugInfo<F>,
}

#[cfg(debug_assertions)]
pub struct DebugInfo<F: Field> {
    pub queried_values: Vec<F>,
    pub position_in_layer: usize,
}

#[cfg(debug_assertions)]
impl<F: Field> DebugInfo<F> {
    pub fn new(queried_values: Vec<F>, position_in_layer: usize) -> Self {
        Self {
            queried_values,
            position_in_layer,
        }
    }

    pub fn queried_values(&self) -> &Vec<F> {
        &self.queried_values
    }

    pub fn position_in_layer(&self) -> usize {
        self.position_in_layer
    }

    pub fn set_queried_values(&mut self, queried_values: Vec<F>) {
        self.queried_values = queried_values;
    }

    pub fn set_position_in_layer(&mut self, position_in_layer: usize) {
        self.position_in_layer = position_in_layer;
    }
}

#[cfg(non_debug_assertions)]
pub struct DebugInfo<F> {
    pub _phantom: std::marker::PhantomData<F>,
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
        if let Some(hash) = self.left_hash {
            f.write_str(&std::format!("Left hash: {}, ", hash))?;
        }
        if let Some(hash) = self.right_hash {
            f.write_str(&std::format!("Right hash: {}, ", hash))?;
        }
        f.write_str(&std::format!(
            " Witness Elements: {:?}",
            self.witness_elements
        ))?;
        #[cfg(debug_assertions)]
        {
            f.write_str(&std::format!(
                " Queried Values: {:?}, Position in Layer: {}",
                self.d.queried_values(),
                self.d.position_in_layer()
            ))?;
        }
        f.write_str("\n")?;
        Ok(())
    }
}

#[cfg(debug_assertions)]
#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::mixed_degree_decommitment::DebugInfo;
    use crate::core::fields::m31::M31;

    #[test]
    fn display_test() {
        let path = super::MixedDecommitment::<M31, Blake3Hasher>::new(vec![
            vec![super::DecommitmentNode::<M31, Blake3Hasher> {
                left_hash: Some(Blake3Hasher::hash(b"a")),
                right_hash: None,
                witness_elements: (0..6).step_by(2).map(M31::from_u32_unchecked).collect(),
                d: DebugInfo::new((1..7).step_by(2).map(M31::from_u32_unchecked).collect(), 0),
            }],
            vec![
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    right_hash: Some(Blake3Hasher::hash(b"b")),
                    left_hash: None,
                    witness_elements: (3..6).step_by(2).map(M31::from_u32_unchecked).collect(),
                    d: DebugInfo::new((4..7).step_by(2).map(M31::from_u32_unchecked).collect(), 0),
                },
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    left_hash: Some(Blake3Hasher::hash(b"c")),
                    right_hash: None,
                    witness_elements: (6..9).step_by(2).map(M31::from_u32_unchecked).collect(),
                    d: DebugInfo::new((7..10).step_by(2).map(M31::from_u32_unchecked).collect(), 0),
                },
            ],
        ]);
        println!("{}", path)
    }
}
