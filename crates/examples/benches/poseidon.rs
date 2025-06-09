use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stwo_prover::core::pcs::PcsConfig;
use stwo_prover::examples::poseidon::prove_poseidon;

pub fn simd_poseidon(c: &mut Criterion) {
    const LOG_N_INSTANCES: u32 = 18;
    let mut group = c.benchmark_group("poseidon2");
    group.throughput(Throughput::Elements(1u64 << LOG_N_INSTANCES));
    group.bench_function(format!("poseidon2 2^{} instances", LOG_N_INSTANCES), |b| {
        b.iter(|| prove_poseidon(LOG_N_INSTANCES, PcsConfig::default()));
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_poseidon);
criterion_main!(bit_rev);
