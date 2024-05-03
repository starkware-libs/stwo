#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use itertools::Itertools;
use num_traits::Zero;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::{CPUBackend, Col};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::MerkleOps;

const LOG_N_ROWS: u32 = 16;

const LOG_N_COLS: u32 = 8;

fn bench_blake2s_merkle<B: MerkleOps<Blake2sMerkleHasher>>(c: &mut Criterion, id: &str) {
    let col: Col<B, BaseField> = (0..1 << LOG_N_ROWS).map(|_| BaseField::zero()).collect();
    let cols = (0..1 << LOG_N_COLS).map(|_| col.clone()).collect_vec();
    let col_refs = cols.iter().collect_vec();
    let mut group = c.benchmark_group("merkle throughput");
    group.throughput(Throughput::Elements(1 << (LOG_N_COLS + LOG_N_ROWS)));
    group.throughput(Throughput::Bytes(4 << (LOG_N_COLS + LOG_N_ROWS)));
    group.bench_function(&format!("{id} merkle"), |b| {
        b.iter_with_large_drop(|| B::commit_on_layer(LOG_N_ROWS, None, &col_refs))
    });
}

fn blake2s_merkle_benches(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    if stwo_prover::platform::avx512_detected() {
        use stwo_prover::core::backend::avx512::AVX512Backend;
        bench_blake2s_merkle::<AVX512Backend>(c, "avx");
    }
    bench_blake2s_merkle::<SimdBackend>(c, "simd");
    bench_blake2s_merkle::<CPUBackend>(c, "cpu");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = blake2s_merkle_benches);
criterion_main!(benches);
