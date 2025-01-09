use criterion::{criterion_group, criterion_main, Criterion};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::CpuBackend;
use stwo_prover::core::channel::{Blake2sChannel, Channel, Poseidon31Channel};
use stwo_prover::core::proof_of_work::GrindOps;

pub fn grind<C: Channel, B: GrindOps<C>>(name: impl ToString, c: &mut Criterion) {
    const POW_BITS: u32 = 20;
    let mut group = c.benchmark_group(format!("grind {}", name.to_string()));
    group.bench_function(
        format!("grind {} {} bits", name.to_string(), POW_BITS),
        |b| {
            b.iter(|| {
                let channel = C::default();
                <B as GrindOps<C>>::grind(&channel, POW_BITS)
            });
        },
    );
}

pub fn simd_grind_blake2s(c: &mut Criterion) {
    grind::<Blake2sChannel, SimdBackend>("blake2s simd", c);
}

pub fn simd_grind_poseidon31(c: &mut Criterion) {
    grind::<Poseidon31Channel, SimdBackend>("poseidon31 simd", c);
}

pub fn cpu_grind_blake2s(c: &mut Criterion) {
    grind::<Blake2sChannel, CpuBackend>("blake2s cpu", c);
}

pub fn cpu_grind_poseidon31(c: &mut Criterion) {
    grind::<Poseidon31Channel, CpuBackend>("poseidon31 cpu", c);
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = simd_grind_blake2s, cpu_grind_blake2s, simd_grind_poseidon31, cpu_grind_poseidon31);
criterion_main!(benches);
