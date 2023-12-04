use blake2::{Blake2s256, Digest};
use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
use prover_research::commitment_scheme::blake3_hash::Blake3Hasher;
use prover_research::commitment_scheme::hasher::{Hasher, Name};
use prover_research::commitment_scheme::merkle_tree::MerkleTree;
use prover_research::core::fields::m31::M31;

static N_BYTES_U32: usize = 4;

fn prepare_element_vector(size: usize) -> Vec<M31> {
    (0..size as u32).map(M31::from_u32_unchecked).collect()
}

fn merkle_bench<T: Hasher>(group: &mut BenchmarkGroup<'_, WallTime>, elems: &[M31]) {
    let size = elems.len();
    let elems = elems.to_vec();
    group.sample_size(10);
    group.throughput(Throughput::Bytes((size * N_BYTES_U32) as u64));
    group.bench_with_input(
        BenchmarkId::new(T::Hash::NAME, size),
        &size,
        |b: &mut criterion::Bencher<'_>, &_size| {
            b.iter_batched(
                || -> Vec<M31> { elems.clone() },
                |elems| {
                    MerkleTree::<M31, Blake3Hasher>::commit(vec![elems]);
                },
                BatchSize::LargeInput,
            )
        },
    );
}

fn merkle_blake3_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Blake3_Tree");
    let exponent = 20u32;
    // Set Up.
    let elems = prepare_element_vector(2usize.pow(exponent));

    // Benchmark Loop.
    merkle_bench::<Blake3Hasher>(&mut group, &elems);
    group.finish();
}

fn merkle_blake2s_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Blake3_Tree");
    let exponent = 20u32;
    // Set Up.
    let elems = prepare_element_vector(2usize.pow(exponent));

    // Benchmark Loop.
    merkle_bench::<Blake2sHasher>(&mut group, &elems);
    group.finish();
}

// Compare Blake2s256 w. Blake3.
fn compare_blakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison of hashing algorithms and caching overhead");
    let exponent = 20u32;
    // Set up.
    let elems: Vec<M31> = prepare_element_vector(2usize.pow(exponent));

    // Benchmark Loop.
    merkle_bench::<Blake2sHasher>(&mut group, &elems);
    merkle_bench::<Blake3Hasher>(&mut group, &elems);

    group.finish();
}

fn single_blake2s_hash_benchmark(c: &mut Criterion) {
    let input = [0u8; 1];
    c.bench_function("Single blake2s hash", |b| {
        b.iter_batched(
            || -> Blake2s256 { Blake2s256::new() },
            |mut h| {
                h.update(&input[..]);
                h.finalize()
            },
            BatchSize::SmallInput,
        )
    });
}

fn single_blake3_hash_benchmark(c: &mut Criterion) {
    let input = [0u8; 1];
    c.bench_function("Single blake3 hash", |b| b.iter(|| blake3::hash(&input)));
}

criterion_group!(
    merkle_benches,
    merkle_blake2s_benchmark,
    merkle_blake3_benchmark,
);

criterion_group!(comparisons, compare_blakes,);

criterion_group!(
    single_hash,
    single_blake2s_hash_benchmark,
    single_blake3_hash_benchmark,
);

criterion_main!(comparisons);
