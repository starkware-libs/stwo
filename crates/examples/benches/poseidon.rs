use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stwo::core::pcs::PcsConfig;
use stwo_examples::poseidon::prove_poseidon;

pub fn simd_poseidon(c: &mut Criterion) {
    if std::env::var_os("STWO_RUN_POSEIDON_PROOF_BENCH").is_none() {
        eprintln!(
            "skipping Poseidon proof bench by default; set \
             STWO_RUN_POSEIDON_PROOF_BENCH=1 to run it explicitly"
        );
        return;
    }

    const LOG_N_INSTANCES: u32 = 18;
    let mut group = c.benchmark_group("poseidon2");
    group.throughput(Throughput::Elements(1u64 << LOG_N_INSTANCES));
    group.bench_function(format!("poseidon2 2^{LOG_N_INSTANCES} instances"), |b| {
        b.iter(|| prove_poseidon(LOG_N_INSTANCES, PcsConfig::default()));
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_poseidon);
criterion_main!(bit_rev);
