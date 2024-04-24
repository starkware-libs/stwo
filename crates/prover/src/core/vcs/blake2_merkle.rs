use num_traits::Zero;
use stwo_verifier::core::fields::m31::BaseField;

use super::blake2_hash::Blake2sHash;
use super::blake2s_ref::compress;
use super::ops::MerkleHasher;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Blake2sMerkleHasher;
impl MerkleHasher for Blake2sMerkleHasher {
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
        state.map(|x| x.to_le_bytes()).flatten().into()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use stwo_verifier::core::fields::m31::BaseField;

    use crate::core::backend::CPUBackend;
    use crate::core::vcs::blake2_merkle::{Blake2sHash, Blake2sMerkleHasher};
    use crate::core::vcs::prover::{MerkleDecommitment, MerkleProver};
    use crate::core::vcs::verifier::{MerkleVerificationError, MerkleVerifier};

    type TestData = (
        BTreeMap<u32, Vec<usize>>,
        MerkleDecommitment<Blake2sMerkleHasher>,
        Vec<Vec<BaseField>>,
        MerkleVerifier<Blake2sMerkleHasher>,
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
        let merkle =
            MerkleProver::<CPUBackend, Blake2sMerkleHasher>::commit(cols.iter().collect_vec());

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
        decommitment.hash_witness[20] = Blake2sHash::default();

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
        decommitment.hash_witness.push(Blake2sHash::default());

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }
}
