use criterion::{criterion_group, criterion_main, Criterion};
use stwo::core::fields::m31::BaseField;
use stwo::fibonacci::MultiFibonacci;

fn multi_fib_benchmark(c: &mut Criterion) {
    let multi_fib = MultiFibonacci::new(16, 5, BaseField::from_u32_unchecked(443693538));
    c.bench_function("prove", |b| {
        b.iter(|| {
            multi_fib.prove().unwrap();
        })
    });
}

criterion_group!(benches, multi_fib_benchmark);
criterion_main!(benches);
