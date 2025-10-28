#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use num_traits::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as Blake2sMerkleHasherLifted;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Col;
use stwo::prover::vcs::prover::MerkleProver;
use stwo::prover::vcs_lifted::prover::MerkleProverLifted;

fn generate_trace(log_size_to_n_cols: &Vec<(usize, usize)>) -> Vec<Col<SimdBackend, BaseField>> {
    log_size_to_n_cols
        .iter()
        .map(|(log_size, n_cols)| {
            (0..*n_cols).map(move |_| BaseColumn::from_cpu(vec![BaseField::zero(); 1 << log_size]))
        })
        .flatten()
        .collect_vec()
}

fn bench_merkle_commit(c: &mut Criterion, id: &str, log_size_to_n_cols: &Vec<(usize, usize)>) {
    let cols = generate_trace(log_size_to_n_cols);
    let mut group = c.benchmark_group("merkle_commit");
    let merkle_commit: Box<dyn Fn()> = match id {
        "mixed" => Box::new(|| {
            MerkleProver::<SimdBackend, Blake2sMerkleHasher>::commit(cols.iter().collect_vec());
        }),
        "lifted" => Box::new(|| {
            MerkleProverLifted::<SimdBackend, Blake2sMerkleHasherLifted>::commit(
                cols.iter().collect_vec(),
            );
        }),
        _ => unreachable!(),
    };
    group.bench_function(format!("{id} merkle commit: {log_size_to_n_cols:?}"), |b| {
        b.iter_with_large_drop(&merkle_commit)
    });
}

fn blake2s_merkle_commit(c: &mut Criterion) {
    let test_vectors = [
        vec![(23, 1000)],
        // vec![(18, 20), (19, 20), (20, 20), (21, 20)],
        // vec![(18, 100), (19, 100)],
    ];

    for vector in test_vectors.iter() {
        bench_merkle_commit(c, "mixed", &vector);
        bench_merkle_commit(c, "lifted", &vector);
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = blake2s_merkle_commit);
criterion_main!(benches);
