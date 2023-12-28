use std::fmt;

use super::hasher::Hasher;
use crate::core::fields::Field;

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

pub struct DecommitmentNode<F: Field, H: Hasher> {
    pub hash: Option<H::Hash>,
    pub injected_elements: Vec<F>,
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
            }],
            vec![
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    hash: Some(Blake3Hasher::hash(b"b")),
                    injected_elements: (3..6).map(M31::from_u32_unchecked).collect(),
                },
                super::DecommitmentNode::<M31, Blake3Hasher> {
                    hash: Some(Blake3Hasher::hash(b"c")),
                    injected_elements: (6..9).map(M31::from_u32_unchecked).collect(),
                },
            ],
        ]);

        println!("{}", path)
    }
}
