use prover_research::core::field::m31::{M31, P};
use rand::Rng;

pub fn field_operations_bench(c: &mut criterion::Criterion) {
    let mut rng = rand::thread_rng();
    let mut values: Vec<M31> = Vec::new();
    for _ in 0..(1 << 20) {
        values.push(M31::from_u32_unchecked(rng.gen::<u32>() % P));
    }
    let mut x0 = M31::from_u32_unchecked(rng.gen::<u32>() % P);

    c.bench_function("mul", |b| {
        b.iter(|| {
            for elem in &values {
                for _ in 0..50 {
                    x0 *= *elem;
                }
            }
        })
    });

    c.bench_function("add", |b| {
        b.iter(|| {
            for elem in &values {
                for _ in 0..50 {
                    x0 += *elem;
                }
            }
        })
    });
}

criterion::criterion_group!(benches, field_operations_bench);
criterion::criterion_main!(benches);
