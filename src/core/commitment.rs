use blake3::Hash;
use blake3::Hasher;

use super::field::Field;
use super::poly::line::LineDomain;
use super::poly::line::LinePoly;

pub struct TableCommitmentProver {
    pub domain: LineDomain,
    pub cols_per_layer: Vec<Vec<usize>>,
    pub layers: Vec<Vec<Hash>>,
}
impl TableCommitmentProver {
    pub fn new(evals: &[&[Field]]) -> Self {
        let mut cols_per_layer = vec![];
        let mut evals: Vec<_> = evals.iter().enumerate().collect();
        let mut n_bits = evals
            .iter()
            .map(|e| e.1.len().ilog2())
            .max()
            .expect("At least one column must be given");
        let domain = LineDomain::canonic(n_bits as usize);
        let mut layers: Vec<Vec<Hash>> = vec![];
        loop {
            let mut new_layer = Vec::with_capacity(1 << n_bits);
            let layer_evals: Vec<_> = evals
                .drain(..)
                .filter(|x| x.1.len().ilog2() == n_bits)
                .collect();
            for i in 0..(1 << n_bits) {
                let mut hasher = Hasher::new();
                for layer in &layer_evals {
                    hasher.update(&layer.1[i].to_bytes());
                }
                if let Some(layer) = layers.last() {
                    hasher.update(layer[i << 1].as_bytes());
                    hasher.update(layer[(i << 1) | 1].as_bytes());
                }
                new_layer.push(hasher.finalize());
            }
            layers.push(new_layer);
            cols_per_layer.push(layer_evals.into_iter().map(|x| x.0).collect());
            if n_bits == 0 {
                break;
            }
            n_bits -= 1;
        }
        Self {
            domain,
            cols_per_layer,
            layers,
        }
    }
    pub fn decommit(&self, indices: Vec<usize>, oracles: &[LayerOracle<'_>]) -> Decommitment {
        let mut decommitment = Decommitment {
            values: vec![],
            hashes: vec![],
        };
        let mut domain = self.domain;
        for (layer, cols_at_layer) in self.layers.iter().zip(self.cols_per_layer.iter()) {
            let mut next_indices = vec![];
            let mut i = 0;
            while i < indices.len() {
                let index = indices[i];
                // Decommit leaves.
                for c in cols_at_layer.iter() {
                    decommitment.values.push(oracles[*c].sample(index));
                }

                if index & 1 == 0 {
                    if indices.get(i + 1) == Some(&(index + 1)) {
                        // Both children present. Nothing is needed.
                        i += 1;
                    } else {
                        // Left child is present. Add right child.
                        decommitment.hashes.push(layer[i + 1]);
                    }
                } else {
                    // Right child.
                    decommitment.hashes.push(layer[i - 1]);
                }
                next_indices.push(index >> 1);
                i += 1;
            }
            domain = domain.double();
        }
        decommitment
    }
}

pub struct LayerOracle<'a> {
    pub domain: LineDomain,
    pub poly: &'a LinePoly,
}
impl<'a> LayerOracle<'a> {
    fn sample(&self, index: usize) -> Field {
        self.poly.eval_at_point(self.domain.at(index))
    }
}
pub struct Decommitment {
    values: Vec<Field>,
    hashes: Vec<Hash>,
}
