// TODO(Ohad): write better benchmarks. Reduce the variance in sample size.
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use stwo::commitment_scheme::blake2_hash::Blake2sHasher;
use stwo::commitment_scheme::blake3_hash::Blake3Hasher;
use stwo::commitment_scheme::hasher::{Hasher, Name};
use stwo::commitment_scheme::merkle_tree::MerkleTree;
use stwo::core::fields::m31::M31;
use stwo::core::fields::IntoSlice;

static N_BYTES_U32: usize = 4;

fn prepare_element_vector(size: usize) -> Vec<M31> {
    (0..size as u32).map(M31::from_u32_unchecked).collect()
}

fn merkle_bench<H: Hasher>(group: &mut BenchmarkGroup<'_, WallTime>, elems: &[M31])
where
    M31: IntoSlice<<H as Hasher>::NativeType>,
{
    let size = elems.len();
    const LOG_N_COLS: usize = 7;
    let cols: Vec<_> = elems
        .chunks(size >> LOG_N_COLS)
        .map(|chunk| chunk.to_vec())
        .collect();
    assert_eq!(cols.len(), 1 << LOG_N_COLS);
    group.sample_size(10);
    group.throughput(Throughput::Bytes((size * N_BYTES_U32) as u64));
    group.bench_function(BenchmarkId::new(H::Hash::NAME, size), |b| {
        b.iter_batched(
            || cols.clone(),
            |cols| {
                black_box(MerkleTree::<M31, H>::commit(black_box(cols)));
            },
            BatchSize::LargeInput,
        )
    });
}

fn merkle_blake3_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Blake3_Tree");
    for exp in 15u32..20u32 {
        // Set Up.
        let elems: Vec<M31> = prepare_element_vector(2usize.pow(exp));

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
        let elems: Vec<M31> = (0..(size as u32)).map(M31::from_u32_unchecked).collect();

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
        let elems: Vec<M31> = (0..(size as u32)).map(M31::from_u32_unchecked).collect();

        // Benchmark Loop.
        merkle_bench::<Blake2sHasher>(&mut group, &elems);
        merkle_bench::<Blake3Hasher>(&mut group, &elems);
    }
    group.finish();
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

criterion_group!(single_hash, single_blake3_hash_benchmark,);

criterion_main!(comparisons);
