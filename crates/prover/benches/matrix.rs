use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use stwo_prover::core::fields::m31::{M31, P};
use stwo_prover::core::fields::qm31::QM31;
use stwo_prover::math::matrix::{RowMajorMatrix, SquareMatrix};

const MATRIX_SIZE: usize = 24;
const QM31_MATRIX_SIZE: usize = 6;

// TODO(ShaharS): Share code with other benchmarks.
fn row_major_matrix_multiplication_bench(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let matrix_m31 = RowMajorMatrix::<M31, MATRIX_SIZE>::new(
        (0..MATRIX_SIZE.pow(2))
            .map(|_| M31::from_u32_unchecked(rng.gen::<u32>() % P))
            .collect::<Vec<M31>>(),
    );

    let matrix_qm31 = RowMajorMatrix::<QM31, QM31_MATRIX_SIZE>::new(
        (0..QM31_MATRIX_SIZE.pow(2))
            .map(|_| {
                QM31::from_u32_unchecked(
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
                )
            })
            .collect::<Vec<QM31>>(),
    );

    // Create vector M31.
    let vec: [M31; MATRIX_SIZE] =
        [(); MATRIX_SIZE].map(|_| M31::from_u32_unchecked(rng.gen::<u32>() % P));

    // Create vector QM31.
    let vec_qm31: [QM31; QM31_MATRIX_SIZE] = [(); QM31_MATRIX_SIZE].map(|_| {
        QM31::from_u32_unchecked(
            rng.gen::<u32>() % P,
            rng.gen::<u32>() % P,
            rng.gen::<u32>() % P,
            rng.gen::<u32>() % P,
        )
    });

    // bench matrix multiplication.
    c.bench_function(
        &format!("RowMajorMatrix M31 {size}x{size} mul", size = MATRIX_SIZE),
        |b| {
            b.iter(|| {
                black_box(matrix_m31.mul(vec));
            })
        },
    );
    c.bench_function(
        &format!(
            "QM31 RowMajorMatrix {size}x{size} mul",
            size = QM31_MATRIX_SIZE
        ),
        |b| {
            b.iter(|| {
                black_box(matrix_qm31.mul(vec_qm31));
            })
        },
    );
}

criterion_group!(benches, row_major_matrix_multiplication_bench);
criterion_main!(benches);
