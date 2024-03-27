use itertools::Itertools;
use num_traits::Zero;

use super::blake2s_ref::compress;
use super::ops::{MerkleHasher, MerkleOps};
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;

#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct Blake2sHash(pub [u32; 8]);
pub struct Blake2Hasher;
impl MerkleHasher for Blake2Hasher {
    type Hash = Blake2sHash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
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
        let rem = 15 - ((column_values.len() + 15) % 16);
        let padded_values = column_values
            .iter()
            .copied()
            .chain(std::iter::repeat(BaseField::zero()).take(rem));
        for chunk in padded_values.array_chunks::<16>() {
            state = compress(state, unsafe { std::mem::transmute(chunk) }, 0, 0, 0, 0);
        }
        Blake2sHash(state)
    }
}

impl std::fmt::Debug for Blake2sHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write as hex.
        for &byte in self.0.iter() {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl MerkleOps<Blake2Hasher> for CPUBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Vec<BaseField>],
    ) -> Vec<Blake2sHash> {
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
    use std::collections::BTreeMap;

    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::commitment_scheme::blake2_merkle::{Blake2Hasher, Blake2sHash};
    use crate::commitment_scheme::prover::{MerkleDecommitment, MerkleProver};
    use crate::commitment_scheme::verifier::{MerkleVerificationError, MerkleVerifier};
    use crate::core::backend::CPUBackend;
    use crate::core::fields::m31::BaseField;

    type TestData = (
        BTreeMap<u32, Vec<usize>>,
        MerkleDecommitment<Blake2Hasher>,
        Vec<Vec<BaseField>>,
        MerkleVerifier<Blake2Hasher>,
    );
    fn prepare_merkle() -> TestData {
        const N_COLS: usize = 400;
        const N_QUERIES: usize = 7;
        let log_size_range = 6..9;

        let rng = &mut StdRng::seed_from_u64(0);
        let log_sizes = (0..N_COLS)
            .map(|_| rng.gen_range(log_size_range.clone()))
            .collect_vec();
        let cols = log_sizes
            .iter()
            .map(|&log_size| {
                (0..(1 << log_size))
                    .map(|_| BaseField::from(rng.gen_range(0..(1 << 30))))
                    .collect_vec()
            })
            .collect_vec();
        let merkle = MerkleProver::<CPUBackend, Blake2Hasher>::commit(cols.iter().collect_vec());

        let mut queries = BTreeMap::<u32, Vec<usize>>::new();
        for log_size in log_size_range.rev() {
            let layer_queries = (0..N_QUERIES)
                .map(|_| rng.gen_range(0..(1 << log_size)))
                .sorted()
                .dedup()
                .collect_vec();
            queries.insert(log_size, layer_queries);
        }

        let (values, decommitment) = merkle.decommit(queries.clone(), cols.iter().collect_vec());

        let verifier = MerkleVerifier {
            root: merkle.root(),
            column_log_sizes: log_sizes,
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
        decommitment.hash_witness[20] = Blake2sHash([0; 8]);

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle();
        values[3][6] = BaseField::zero();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle();
        values[3].push(BaseField::zero());

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::ColumnValuesTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle();
        values[3].pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::ColumnValuesTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle();
        decommitment.hash_witness.push(Blake2sHash([0; 8]));

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }
}
