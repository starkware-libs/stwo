use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pprof::{criterion::{Output, PProfProfiler}};

pub fn fibonacci_benchmark(c: &mut Criterion) {
    use stwo::core::fields::m31::M31;
    use stwo::fibonacci::Fibonacci;

    const FIB_LOG_SIZE: u32 = 18;
    let fib = Fibonacci::new(FIB_LOG_SIZE, M31::from_u32_unchecked(443693538));

    let mut group = c.benchmark_group("fibonacci");
    group.sample_size(10);
    group.bench_function("prove", |b| {
        b.iter(|| black_box(fib.prove()));
    });
}

criterion_group! {name = fibonacci_protobuf;
config = {
     Criterion::default().sample_size(10).with_profiler(PProfProfiler::new(100, Output::Protobuf))};
targets = fibonacci_benchmark}
criterion_group!(fibonacci_20, fibonacci_benchmark);
criterion_main!(fibonacci_protobuf);
