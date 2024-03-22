use itertools::Itertools;
use num_traits::Zero;

use super::blake2s_ref::compress;
use super::ops::{MerkleHasher, MerkleOps};
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;

pub struct Blake2Hasher;
impl MerkleHasher for Blake2Hasher {
    type Hash = [u32; 8];

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        node_values: &[BaseField],
    ) -> Self::Hash {
        let mut state = [0; 8];
        if let Some((left, right)) = children_hashes {
            state = compress(
                state,
                unsafe { std::mem::transmute([left, right]) },
                0,
                0,
                0,
                0,
            );
        }
        let rem = 15 - ((node_values.len() + 15) % 16);
        let padded_values = node_values
            .iter()
            .copied()
            .chain(std::iter::repeat(BaseField::zero()).take(rem));
        for chunk in padded_values.array_chunks::<16>() {
            state = compress(state, unsafe { std::mem::transmute(chunk) }, 0, 0, 0, 0);
        }
        state
    }
}

impl MerkleOps<Blake2Hasher> for CPUBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<[u32; 8]>>,
        columns: &[&Vec<BaseField>],
    ) -> Vec<[u32; 8]> {
        (0..(1 << log_size))
            .map(|i| {
                Blake2Hasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column[i]).collect_vec(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::commitment_scheme::blake2_merkle::Blake2Hasher;
    use crate::commitment_scheme::prover::{Decommitment, MerkleProver};
    use crate::commitment_scheme::verifier::{MerkleTreeVerifier, MerkleVerificationError};
    use crate::core::backend::CPUBackend;
    use crate::core::fields::m31::BaseField;

    type TestData = (
        Vec<usize>,
        Decommitment<Blake2Hasher>,
        Vec<(u32, Vec<BaseField>)>,
        MerkleTreeVerifier<Blake2Hasher>,
    );
    fn prepare_merkle() -> TestData {
        const N_COLS: usize = 400;
        const N_QUERIES: usize = 7;

        let rng = &mut StdRng::seed_from_u64(0);
        let log_sizes = (0..N_COLS)
            .map(|_| rng.gen_range(6..9))
            .sorted()
            .rev()
            .collect_vec();
        let max_log_size = *log_sizes.iter().max().unwrap();
        let cols = log_sizes
            .iter()
            .map(|&log_size| {
                (0..(1 << log_size))
                    .map(|_| BaseField::from(rng.gen_range(0..(1 << 30))))
                    .collect_vec()
            })
            .collect_vec();
        let merkle = MerkleProver::<CPUBackend, Blake2Hasher>::commit(cols.iter().collect_vec());

        let queries = (0..N_QUERIES)
            .map(|_| rng.gen_range(0..(1 << max_log_size)))
            .sorted()
            .dedup()
            .collect_vec();
        let decommitment = merkle.decommit(queries.clone());
        let values = cols
            .iter()
            .map(|col| {
                let layer_queries = queries
                    .iter()
                    .map(|&q| q >> (max_log_size - col.len().ilog2()))
                    .dedup();
                layer_queries.map(|q| col[q]).collect_vec()
            })
            .collect_vec();
        let values = log_sizes.into_iter().zip(values).collect_vec();

        let verifier = MerkleTreeVerifier {
            root: merkle.root(),
        };
        (queries, decommitment, values, verifier)
    }

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle();

        verifier.verify(queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle();
        decommitment.witness[20] = [0; 8];

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle();
        values[3].1[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle();
        decommitment.witness.pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle();
        values[3].1.push(BaseField::zero());

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::ColumnValuesTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle();
        values[3].1.pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::ColumnValuesTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle();
        decommitment.witness.push([0; 8]);

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }
}
