#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use num_traits::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as Blake2sMerkleHasherLifted;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Col;
use stwo::prover::vcs::prover::MerkleProver;
use stwo::prover::vcs_lifted::prover::MerkleProverLifted;

const LOG_N_ROWS: u32 = 20;

const LOG_N_COLS: u32 = 8;

fn generate_trace() -> Vec<Col<SimdBackend, BaseField>> {
    let col: Col<SimdBackend, BaseField> =
        (0..1 << LOG_N_ROWS).map(|_| BaseField::zero()).collect();
    let mut cols = (0..1 << LOG_N_COLS).map(|_| col.clone()).collect_vec();
    (0..(1 << LOG_N_COLS) - 1)
        .for_each(|i| cols[i] = (0..1 << 8).map(|_| BaseField::zero()).collect());
    cols
}

fn bench_merkle_commit(c: &mut Criterion, id: &str) {
    let cols = generate_trace();
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
    group.bench_function(format!("{id} merkle commit"), |b| {
        b.iter_with_large_drop(&merkle_commit)
    });
}

fn blake2s_merkle_commit(c: &mut Criterion) {
    bench_merkle_commit(c, "mixed");
    bench_merkle_commit(c, "lifted");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = blake2s_merkle_commit);
criterion_main!(benches);
