#![feature(sync_unsafe_cell)]
#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, Criterion};
use prover_research::benches::commitment::{fft_bench, prepare};

fn criterion_benchmark(c: &mut Criterion) {
    let (values, twiddles) = prepare();
    c.bench_function("fft", |b| {
        b.iter(|| {
            fft_bench(&values, &twiddles);
        })
    });
    let val = values.into_inner();
    assert!(val[0].0[0] != val[1].0[0]);
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(30);
    targets=criterion_benchmark
);
criterion_main!(benches);
