use blake2::{Blake2s256, Digest};
use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use prover_research::commitment_scheme::{
    blake2_hash::Blake2sHasher,
    blake3_hash::Blake3Hasher,
    hasher::{Hasher, Name},
    merkle_tree::MerkleTree,
};

static N_BYTES_U32: usize = 4;

fn prepare_element_vector(size: usize) -> Vec<u32> {
    (0..size as u32).collect()
}

fn merkle_bench<T: Hasher>(group: &mut BenchmarkGroup<'_, WallTime>, elems: &[u32]) {
    let size = elems.len();
    group.sample_size(10);
    group.throughput(Throughput::Bytes((size * N_BYTES_U32) as u64));
    group.bench_with_input(
        BenchmarkId::new(T::Hash::NAME, size),
        &size,
        |b: &mut criterion::Bencher<'_>, &_size| {
            b.iter(|| {
                MerkleTree::<T>::commit(elems);
            })
        },
    );
}

fn merkle_blake3_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Blake3_Tree");
    for exp in 15u32..20u32 {
        // Set Up.
        let elems: Vec<u32> = prepare_element_vector(2usize.pow(exp));

        // Benchmark Loop.
        merkle_bench::<Blake3Hasher>(&mut group, &elems);
    }
    group.finish();
}

fn merkle_blake2s_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Blake2s_Tree");
    for exp in 15u32..20u32 {
        // Set up.
        let size = 2usize.pow(exp);
        let elems: Vec<u32> = (0..(size as u32)).collect();

        // Benchmark Loop.
        merkle_bench::<Blake2sHasher>(&mut group, &elems);
    }
    group.finish();
}

// Compare Blake2s256 w. Blake3.
fn compare_blakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison of hashing algorithms and caching overhead");
    for exp in 15u32..20u32 {
        // Set up.
        let size = 2usize.pow(exp);
        let elems: Vec<u32> = (0..(size as u32)).collect();

        // Benchmark Loop.
        merkle_bench::<Blake2sHasher>(&mut group, &elems);
        merkle_bench::<Blake3Hasher>(&mut group, &elems);
    }
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
