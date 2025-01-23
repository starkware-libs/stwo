use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stwo_prover::core::pcs::PcsConfig;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo_prover::core::vcs::poseidon31_merkle::Poseidon31MerkleChannel;
use stwo_prover::examples::poseidon::prove_poseidon;

pub fn simd_poseidon_blake2s(c: &mut Criterion) {
    const LOG_N_INSTANCES: u32 = 18;
    let mut group = c.benchmark_group("poseidon2 with blake2s");
    group.throughput(Throughput::Elements(1u64 << LOG_N_INSTANCES));
    group.bench_function(
        format!("poseidon2 with blake2s 2^{} instances", LOG_N_INSTANCES),
        |b| {
            b.iter(|| {
                prove_poseidon::<Blake2sMerkleChannel>(LOG_N_INSTANCES, PcsConfig::default())
            });
        },
    );
}

pub fn simd_poseidon_poseidon31(c: &mut Criterion) {
    const LOG_N_INSTANCES: u32 = 18;
    let mut group = c.benchmark_group("poseidon2 with poseidon31");
    group.throughput(Throughput::Elements(1u64 << LOG_N_INSTANCES));
    group.bench_function(
        format!("poseidon2 with poseidon31 2^{} instances", LOG_N_INSTANCES),
        |b| {
            b.iter(|| {
                prove_poseidon::<Poseidon31MerkleChannel>(LOG_N_INSTANCES, PcsConfig::default())
            });
        },
    );
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_poseidon_blake2s, simd_poseidon_poseidon31);
criterion_main!(bit_rev);
