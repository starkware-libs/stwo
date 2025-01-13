use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_prover::core::backend::simd::m31::PackedM31;
use stwo_prover::core::backend::simd::qm31::PackedQM31;
use stwo_prover::core::fields::{batch_inverse, batch_inverse_chunked, batch_inverse_chunked2};

pub const N_ELEMENTS: usize = 1 << 18;

pub fn m31_batch_inverse_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<PackedM31> = (0..N_ELEMENTS)
        .map(|_| PackedM31::from_array(std::array::from_fn(|_| rng.gen())))
        .collect();

    c.bench_function("M31 batch inverse not chunked", |b| {
        b.iter(|| {
            black_box(batch_inverse(&elements));
        })
    });

    c.bench_function("M31 chunked 32", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 32>(&elements));
        })
    });
    c.bench_function("M31 chunked2 32", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 32));
        })
    });

    c.bench_function("M31 chunked 64", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 64>(&elements));
        })
    });
    c.bench_function("M31 chunked2 64", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 64));
        })
    });

    c.bench_function("M31 chunked 128", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 128>(&elements));
        })
    });
    c.bench_function("M31 chunked2 128", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 128));
        })
    });

    c.bench_function("M31 chunked 256", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 256>(&elements));
        })
    });
    c.bench_function("M31 chunked2 256", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 256));
        })
    });

    c.bench_function("M31 chunked 512", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 512>(&elements));
        })
    });
    c.bench_function("M31 chunked2 512", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 512));
        })
    });

    c.bench_function("M31 chunked 1024", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 1024>(&elements));
        })
    });
    c.bench_function("M31 chunked2 1024", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 1024));
        })
    });

    c.bench_function("M31 chunked 2048", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 2048>(&elements));
        })
    });
    c.bench_function("M31 chunked2 2048", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 2048));
        })
    });

    c.bench_function("M31 chunked 4096", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 4096>(&elements));
        })
    });
    c.bench_function("M31 chunked2 4096", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 4096));
        })
    });

    c.bench_function("M31 chunked 8192", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 8192>(&elements));
        })
    });
    c.bench_function("M31 chunked2 8192", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked2(&elements, 8192));
        })
    });
}

pub fn qm31_batch_inverse_bench(c: &mut Criterion) {
    // QM31 benchmarks remain unchanged as they don't have chunked2 variants
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<PackedQM31> = (0..N_ELEMENTS)
        .map(|_| PackedQM31::from_array(std::array::from_fn(|_| rng.gen())))
        .collect();

    c.bench_function("QM31 batch inverse not chunked", |b| {
        b.iter(|| {
            black_box(batch_inverse(&elements));
        })
    });

    c.bench_function("QM31 chunked 32", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 32>(&elements));
        })
    });

    c.bench_function("QM31 chunked 64", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 64>(&elements));
        })
    });

    c.bench_function("QM31 chunked 128", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 128>(&elements));
        })
    });

    c.bench_function("QM31 chunked 256", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 256>(&elements));
        })
    });

    c.bench_function("QM31 chunked 512", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 512>(&elements));
        })
    });
    c.bench_function("QM31 chunked 1024", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 1024>(&elements));
        })
    });
    c.bench_function("QM31 chunked 2048", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 2048>(&elements));
        })
    });

    c.bench_function("QM31 chunked 4096", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 4096>(&elements));
        })
    });

    c.bench_function("QM31 chunked 8192", |b| {
        b.iter(|| {
            black_box(batch_inverse_chunked::<_, 8192>(&elements));
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = m31_batch_inverse_bench, qm31_batch_inverse_bench,);
criterion_main!(benches);
