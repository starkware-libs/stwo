use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stwo::core::fields::m31::BaseField;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, ColumnOps};
use stwo::prover::vcs::ops::MerkleOps;
use stwo::prover::vcs::prover::MerkleProver;
use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;
use stwo::prover::vcs_lifted::prover::MerkleProverLifted;

/// Given an iterator of (log_size, n_cols): (u32, usize), builds a trace with the prescribed shape.
/// For example, for sizes [(6, 10), (8, 2)], builds a trace whose first 10 columns have log size 6
/// and the last 2 columns have log size 8. Note that we assume that log sizes are in ascending
/// order.
fn prepare_trace<B: ColumnOps<BaseField>>(sizes: &[(u32, usize)]) -> ColumnVec<Col<B, BaseField>> {
    let mut res = vec![];
    for (log_size, n_cols) in sizes.iter() {
        res.extend(
            (0..*n_cols)
                .map(|_| Col::<B, BaseField>::from_iter((0..1 << log_size).map(BaseField::from))),
        );
    }
    res
}

fn bench_blake2s_merkle<B: MerkleOpsLifted<Blake2sMerkleHasher>>(
    c: &mut Criterion,
    sizes: &[(u32, usize)],
) {
    let cols = prepare_trace::<B>(sizes);
    let cols_ref: Vec<_> = cols.iter().collect();
    let mut group = c.benchmark_group("merkle");
    group.bench_function(format!("simd_merkle_lifted: {:?}", sizes), |b| {
        b.iter_with_large_drop(|| MerkleProverLifted::<B, _>::commit(black_box(cols_ref.clone())))
    });
}

fn bench_blake2s_merkle_old<B: MerkleOps<stwo::core::vcs::blake2_merkle::Blake2sMerkleHasher>>(
    c: &mut Criterion,
    sizes: &[(u32, usize)],
) {
    let cols = prepare_trace::<B>(sizes);
    let cols_ref: Vec<_> = cols.iter().collect();
    let mut group = c.benchmark_group("merkle");
    group.bench_function(format!("simd_merkle_mixed: {:?}", sizes), |b| {
        b.iter_with_large_drop(|| MerkleProver::<B, _>::commit(black_box(cols_ref.clone())))
    });
}

fn blake2s_merkle_benches_lifted(c: &mut Criterion) {
    let trace_sizes = [
        vec![(16, 100), (17, 100)],
        vec![
            (16, 100),
            (17, 100),
            (18, 100),
            (19, 100),
            (20, 100),
            (21, 100),
            (22, 200)
        ],
    ];
    for sizes in trace_sizes.iter() {
        bench_blake2s_merkle::<SimdBackend>(c, sizes);
        bench_blake2s_merkle_old::<SimdBackend>(c, sizes);
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = blake2s_merkle_benches_lifted);
criterion_main!(benches);
