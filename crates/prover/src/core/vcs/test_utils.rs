use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::ops::{MerkleHasher, MerkleOps};
use super::prover::MerkleDecommitment;
use super::verifier::MerkleVerifier;
use crate::collections::BTreeMap;
use crate::core::backend::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::prover::MerkleProver;
use crate::prelude::*;

pub type TestData<H> = (
    BTreeMap<u32, Vec<usize>>,
    MerkleDecommitment<H>,
    Vec<BaseField>,
    MerkleVerifier<H>,
);

pub fn prepare_merkle<H: MerkleHasher>() -> TestData<H>
where
    CpuBackend: MerkleOps<H>,
{
    const N_COLS: usize = 10;
    const N_QUERIES: usize = 3;
    let log_size_range = 3..5;

    let mut rng = SmallRng::seed_from_u64(0);
    let log_sizes = (0..N_COLS)
        .map(|_| rng.gen_range(log_size_range.clone()))
        .collect::<Vec<_>>();
    let cols = log_sizes
        .iter()
        .map(|&log_size| {
            (0..(1 << log_size))
                .map(|_| BaseField::from(rng.gen_range(0..(1 << 30))))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let merkle = MerkleProver::<CpuBackend, H>::commit(cols.iter().collect());

    let mut queries = BTreeMap::<u32, Vec<usize>>::new();
    for log_size in log_size_range.rev() {
        let layer_queries = (0..N_QUERIES)
            .map(|_| rng.gen_range(0..(1 << log_size)))
            .sorted()
            .dedup()
            .collect::<Vec<_>>();
        queries.insert(log_size, layer_queries);
    }

    let (values, decommitment) = merkle.decommit(&queries, cols.iter().collect::<Vec<_>>());

    let verifier = MerkleVerifier::new(merkle.root(), log_sizes);
    (queries, decommitment, values, verifier)
}
