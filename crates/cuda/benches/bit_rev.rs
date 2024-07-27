#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use stwo_cuda::{CudaBackend, CUDA_CTX};
use stwo_prover::core::backend::{Col, ColumnOps};
use stwo_prover::core::fields::m31::BaseField;

pub fn cuda_bit_rev(c: &mut Criterion) {
    const SIZE: usize = 1 << 26;
    let data: Col<CudaBackend, BaseField> = (0..SIZE).map(BaseField::from).collect();
    c.bench_function("cuda bit_rev 26bit", |b| {
        b.iter_batched(
            || data.clone(),
            |mut data| {
                <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut data);
                CUDA_CTX.synchronize().unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = cuda_bit_rev);
criterion_main!(bit_rev);
