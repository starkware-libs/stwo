#![feature(iter_array_chunks)]

use std::hint::black_box;
use std::mem::{size_of_val, transmute};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use itertools::Itertools;
use stwo_prover::core::backend::simd::column::BaseColumn;
use stwo_prover::core::backend::simd::fft::ifft::{
    get_itwiddle_dbls, ifft, ifft3_loop, ifft_vecwise_loop,
};
use stwo_prover::core::backend::simd::fft::rfft::{fft, get_twiddle_dbls};
use stwo_prover::core::backend::simd::fft::transpose_vecs;
use stwo_prover::core::backend::simd::m31::PackedBaseField;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::CanonicCoset;

pub fn simd_ifft(c: &mut Criterion) {
    let mut group = c.benchmark_group("iffts");

    for log_size in 16..=28 {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
        let twiddle_dbls_refs = twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec();
        let values: BaseColumn = (0..domain.size()).map(BaseField::from).collect();
        group.throughput(Throughput::Bytes(size_of_val(&*values.data) as u64));
        group.bench_function(BenchmarkId::new("simd ifft", log_size), |b| {
            b.iter_batched(
                || values.clone().data,
                |mut data| unsafe {
                    ifft(
                        transmute::<*mut PackedBaseField, *mut u32>(data.as_mut_ptr()),
                        black_box(&twiddle_dbls_refs),
                        black_box(log_size as usize),
                    );
                },
                BatchSize::LargeInput,
            )
        });
    }
}

pub fn simd_ifft_parts(c: &mut Criterion) {
    const LOG_SIZE: u32 = 14;

    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
    let twiddle_dbls_refs = twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec();
    let values: BaseColumn = (0..domain.size()).map(BaseField::from).collect();

    let mut group = c.benchmark_group("ifft parts");

    // Note: These benchmarks run only on 2^LOG_SIZE elements because of their parameters.
    // Increasing the figure above won't change the runtime of these benchmarks.
    group.throughput(Throughput::Bytes(4 << LOG_SIZE));
    group.bench_function(format!("simd ifft_vecwise_loop 2^{LOG_SIZE}"), |b| {
        b.iter_batched(
            || values.clone().data,
            |mut values| unsafe {
                ifft_vecwise_loop(
                    transmute::<*mut PackedBaseField, *mut u32>(values.as_mut_ptr()),
                    black_box(&twiddle_dbls_refs),
                    black_box(9),
                    black_box(0),
                )
            },
            BatchSize::LargeInput,
        );
    });
    group.bench_function(format!("simd ifft3_loop 2^{LOG_SIZE}"), |b| {
        b.iter_batched(
            || values.clone().data,
            |mut values| unsafe {
                ifft3_loop(
                    transmute::<*mut PackedBaseField, *mut u32>(values.as_mut_ptr()),
                    black_box(&twiddle_dbls_refs[3..]),
                    black_box(7),
                    black_box(4),
                    black_box(0),
                )
            },
            BatchSize::LargeInput,
        );
    });

    const TRANSPOSE_LOG_SIZE: u32 = 20;
    let transpose_values: BaseColumn = (0..1 << TRANSPOSE_LOG_SIZE).map(BaseField::from).collect();
    group.throughput(Throughput::Bytes(4 << TRANSPOSE_LOG_SIZE));
    group.bench_function(format!("simd transpose_vecs 2^{TRANSPOSE_LOG_SIZE}"), |b| {
        b.iter_batched(
            || transpose_values.clone().data,
            |mut values| unsafe {
                transpose_vecs(
                    transmute::<*mut PackedBaseField, *mut u32>(values.as_mut_ptr()),
                    black_box(TRANSPOSE_LOG_SIZE as usize - 4),
                )
            },
            BatchSize::LargeInput,
        );
    });
}

pub fn simd_rfft(c: &mut Criterion) {
    const LOG_SIZE: u32 = 20;

    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddle_dbls = get_twiddle_dbls(domain.half_coset);
    let twiddle_dbls_refs = twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec();
    let values: BaseColumn = (0..domain.size()).map(BaseField::from).collect();

    c.bench_function("simd rfft 20bit", |b| {
        b.iter_with_large_drop(|| unsafe {
            let mut target = Vec::<PackedBaseField>::with_capacity(values.data.len());
            #[allow(clippy::uninit_vec)]
            target.set_len(values.data.len());

            fft(
                black_box(transmute::<*const PackedBaseField, *const u32>(
                    values.data.as_ptr(),
                )),
                transmute::<*mut PackedBaseField, *mut u32>(target.as_mut_ptr()),
                black_box(&twiddle_dbls_refs),
                black_box(LOG_SIZE as usize),
            )
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = simd_ifft, simd_ifft_parts, simd_rfft);
criterion_main!(benches);
