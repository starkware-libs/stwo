use criterion::{black_box, criterion_group, criterion_main, Criterion};
// use num_traits::One;
use prover_research::core::fields::m31::{M31, P};
use prover_research::math::matrix::{CircularMatrix, RowMajorMatrix, SquareMatrix};
use rand::Rng;

const MATRIX_SIZE: usize = 8;

fn matrix_mul(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    // Create row major matrix.
    let row_major_matrix = RowMajorMatrix::<M31, MATRIX_SIZE>::new(
        (0..MATRIX_SIZE.pow(2))
            .map(|_| M31::from_u32_unchecked(rng.gen::<u32>() % P))
            .collect::<Vec<M31>>(),
    );

    // Create circular matrix.
    let mut circular_matrix_values = vec![];
    circular_matrix_values.append(
        &mut (0..MATRIX_SIZE * 2)
            .map(|_| M31::from_u32_unchecked(rng.gen::<u32>() % P))
            .collect::<Vec<M31>>(),
    );
    let circlular_matrix = CircularMatrix::<M31, MATRIX_SIZE>::new(circular_matrix_values);

    // Create vector.
    let vec: [M31; MATRIX_SIZE] =
        [(); MATRIX_SIZE].map(|_| M31::from_u32_unchecked(rng.gen::<u32>() % P));


    // bench matrix multiplication.
    c.bench_function("RowMajorMatrix mul", |b| {
        b.iter(|| {
            black_box(row_major_matrix.mul(vec));
        })
    });
    c.bench_function("CircularMatrix mul", |b| {
        b.iter(|| {
            black_box(circlular_matrix.mul(vec));
        })
    });
}

criterion_group!(benches, matrix_mul);
criterion_main!(benches);
