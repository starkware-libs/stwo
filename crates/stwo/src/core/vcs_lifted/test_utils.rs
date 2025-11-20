use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{MerkleDecommitmentLifted, MerkleVerifierLifted};
use crate::prover::backend::CpuBackend;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;

pub type TestData<H> = (
    Vec<usize>,
    MerkleDecommitmentLifted<H>,
    Vec<BaseField>,
    MerkleVerifierLifted<H>,
);

pub fn prepare_merkle<H: MerkleHasherLifted>() -> TestData<H>
where
    CpuBackend: MerkleOpsLifted<H>,
{
    const N_COLS: usize = 10;
    const N_QUERIES: usize = 4;
    let log_size_range = 3_u32..5;

    let mut rng = SmallRng::seed_from_u64(0);
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
    let merkle = MerkleProverLifted::<CpuBackend, H>::commit(cols.iter().collect_vec());

    let max_log_size: u32 = *log_sizes.iter().max().unwrap();
    let queries = (0..N_QUERIES)
        .map(|_| rng.gen_range(0..(1 << max_log_size)))
        .sorted()
        .dedup()
        .collect_vec();

    let (values, decommitment) = merkle.decommit(queries.clone(), cols.iter().collect_vec());

    let verifier = MerkleVerifierLifted::new(merkle.root(), log_sizes);
    (queries, decommitment.decommitment, values, verifier)
}
