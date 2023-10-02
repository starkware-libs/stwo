use blake2::{Blake2s256, Digest};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
use prover_research::commitment_scheme::blake3_hash::Blake3Hasher;
use prover_research::commitment_scheme::merkle_tree::{self, MerkleTree};

fn merkle_blake3_in_place_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake3 in-place");
    for exp in 15u32..20u32 {
        // Prepare.
        let size = 2usize.pow(exp);
        let leaves: Vec<u32> = (0..(size as u32)).collect();
        // Paramaterize.
        group.sample_size(10);
        group.throughput(Throughput::Bytes((size * 8) as u64));
        // Bench.
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| merkle_tree::MerkleTree::<Blake3Hasher>::commit(&leaves))
        });
    }
    group.finish();
}

fn merkle_blake3_tree_build_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake3 Tree build");
    for exp in 15u32..20u32 {
        let size = 2usize.pow(exp);
        let leaves: Vec<u32> = (0..(size as u32)).collect();

        group.sample_size(10);
        group.throughput(Throughput::Bytes((size * 8) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| {
                MerkleTree::<Blake3Hasher>::commit(&leaves);
            })
        });
    }
    group.finish();
}

fn merkle_blake2s_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLake2s");
    for exp in 15u32..20u32 {
        let size = 2usize.pow(exp);
        let leaves: Vec<u32> = (0..(size as u32)).collect();

        group.sample_size(10);
        group.throughput(Throughput::Bytes((size * 8) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| merkle_tree::MerkleTree::<Blake2sHasher>::commit(&leaves))
        });
    }
    group.finish();
}

// Compare Blake2s256 w. Blake3
fn compare_blakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison of hashing algorithms and caching overhead");
    for exp in 15u32..20u32 {
        // Prepare
        let size = 2usize.pow(exp);
        let leaves: Vec<u32> = (0..(size as u32)).collect();
        group.sample_size(10);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("Blake2s256", size), &size, |b, &_size| {
            b.iter(|| merkle_tree::MerkleTree::<Blake2sHasher>::commit(&leaves))
        });
        group.bench_with_input(
            BenchmarkId::new("Blake3 in-place", size),
            &size,
            |b, &_size| b.iter(|| merkle_tree::MerkleTree::<Blake3Hasher>::commit(&leaves)),
        );
        group.bench_with_input(
            BenchmarkId::new("Blake3 with layer caching", size),
            &size,
            |b, &_size| b.iter(|| MerkleTree::<Blake3Hasher>::commit(&leaves)),
        );
    }
    group.finish();
}

fn single_blake2_hash_benchmark(c: &mut Criterion) {
    // let mut group = c.benchmark_group("Single Hash");
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
    // let mut group = c.benchmark_group("Single Hash");
    let input = [0u8; 1];
    c.bench_function("Single blake3 hash", |b| b.iter(|| blake3::hash(&input)));
}
criterion_group!(
    benches,
    merkle_blake2s_benchmark,
    merkle_blake3_in_place_benchmark,
    merkle_blake3_tree_build_benchmark,
);

criterion_group!(blake2, merkle_blake2s_benchmark,);

criterion_group!(comparisons, compare_blakes,);

criterion_group!(
    single_hash,
    single_blake2_hash_benchmark,
    single_blake3_hash_benchmark,
);

// The criterion-group to run after "$ cargo bench".
criterion_main!(single_hash);
