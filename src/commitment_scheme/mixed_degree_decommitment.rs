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
