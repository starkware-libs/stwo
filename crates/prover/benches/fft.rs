#![feature(iter_array_chunks)]

use std::hint::black_box;
use std::mem::{size_of_val, transmute};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use itertools::Itertools;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::CanonicCoset;

pub fn simd_ifft(c: &mut Criterion) {
    use stwo_prover::core::backend::simd::column::BaseFieldVec;
    use stwo_prover::core::backend::simd::fft::ifft::{get_itwiddle_dbls, ifft};

    let mut group = c.benchmark_group("iffts");

    for log_size in 16..=28 {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
        let twiddle_dbls_refs = twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec();
        let values: BaseFieldVec = (0..domain.size()).map(BaseField::from).collect();
        group.throughput(Throughput::Bytes(size_of_val(&*values.data) as u64));
        group.bench_function(BenchmarkId::new("simd ifft", log_size), |b| {
            b.iter_batched(
                || values.clone().data,
                |mut data| unsafe {
                    ifft(
                        transmute(data.as_mut_ptr()),
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
    use stwo_prover::core::backend::simd::column::BaseFieldVec;
    use stwo_prover::core::backend::simd::fft::ifft::{
        get_itwiddle_dbls, ifft3_loop, ifft_vecwise_loop,
    };
    use stwo_prover::core::backend::simd::fft::transpose_vecs;

    const LOG_SIZE: u32 = 14;

    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
    let twiddle_dbls_refs = twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec();
    let values: BaseFieldVec = (0..domain.size()).map(BaseField::from).collect();

    let mut group = c.benchmark_group("ifft parts");

    // Note: These benchmarks run only on 2^LOG_SIZE elements because of their parameters.
    // Increasing the figure above won't change the runtime of these benchmarks.
    group.throughput(Throughput::Bytes(4 << LOG_SIZE));
    group.bench_function(format!("simd ifft_vecwise_loop 2^{LOG_SIZE}"), |b| {
        b.iter_batched(
            || values.clone().data,
            |mut values| unsafe {
                ifft_vecwise_loop(
                    transmute(values.as_mut_ptr()),
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
                    transmute(values.as_mut_ptr()),
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
    let transpose_values: BaseFieldVec =
        (0..1 << TRANSPOSE_LOG_SIZE).map(BaseField::from).collect();
    group.throughput(Throughput::Bytes(4 << TRANSPOSE_LOG_SIZE));
    group.bench_function(format!("simd transpose_vecs 2^{TRANSPOSE_LOG_SIZE}"), |b| {
        b.iter_batched(
            || transpose_values.clone().data,
            |mut values| unsafe {
                transpose_vecs(
                    transmute(values.as_mut_ptr()),
                    black_box(TRANSPOSE_LOG_SIZE as usize - 4),
                )
            },
            BatchSize::LargeInput,
        );
    });
}

pub fn simd_rfft(c: &mut Criterion) {
    use stwo_prover::core::backend::simd::column::BaseFieldVec;
    use stwo_prover::core::backend::simd::fft::rfft::{fft, get_twiddle_dbls};
    use stwo_prover::core::backend::simd::m31::PackedBaseField;

    const LOG_SIZE: u32 = 20;

    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddle_dbls = get_twiddle_dbls(domain.half_coset);
    let twiddle_dbls_refs = twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec();
    let values: BaseFieldVec = (0..domain.size()).map(BaseField::from).collect();

    c.bench_function("simd rfft 20bit", |b| {
        b.iter_with_large_drop(|| unsafe {
            let mut target = Vec::<PackedBaseField>::with_capacity(values.data.len());
            #[allow(clippy::uninit_vec)]
            target.set_len(values.data.len());

            fft(
                black_box(transmute(values.data.as_ptr())),
                transmute(target.as_mut_ptr()),
                black_box(&twiddle_dbls_refs),
                black_box(LOG_SIZE as usize),
            )
        });
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_ifft(c: &mut criterion::Criterion) {
    use stwo_prover::core::backend::avx512::fft::ifft;
    use stwo_prover::platform;
    if !platform::avx512_detected() {
        return;
    }

    let mut group = c.benchmark_group("iffts");
    for log_size in 16..=28 {
        let (values, twiddle_dbls) = prepare_values(log_size);

        group.throughput(Throughput::Bytes(
            (std::mem::size_of::<BaseField>() as u64) << log_size,
        ));
        group.bench_function(BenchmarkId::new("avx ifft", log_size), |b| {
            b.iter_batched(
                || values.clone().data,
                |mut values| unsafe {
                    ifft::ifft(
                        std::mem::transmute(values.as_mut_ptr()),
                        &twiddle_dbls
                            .iter()
                            .map(|x| x.as_slice())
                            .collect::<Vec<_>>(),
                        log_size as usize,
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_ifft_parts(c: &mut criterion::Criterion) {
    use stwo_prover::core::backend::avx512::fft::{ifft, transpose_vecs};
    use stwo_prover::platform;
    if !platform::avx512_detected() {
        return;
    }

    let (values, twiddle_dbls) = prepare_values(14);
    let mut group = c.benchmark_group("ifft parts");

    // Note: These benchmarks run only on 2^14 elements ebcause of their parameters.
    // Increasing the figure above won't change the runtime of these benchmarks.
    group.throughput(Throughput::Bytes(4 << 14));
    group.bench_function("avx ifft_vecwise_loop 2^14", |b| {
        b.iter_batched(
            || values.clone().data,
            |mut values| unsafe {
                ifft::ifft_vecwise_loop(
                    std::mem::transmute(values.as_mut_ptr()),
                    &twiddle_dbls
                        .iter()
                        .map(|x| x.as_slice())
                        .collect::<Vec<_>>(),
                    9,
                    0,
                )
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("avx ifft3_loop 2^14", |b| {
        b.iter_batched(
            || values.clone().data,
            |mut values| unsafe {
                ifft::ifft3_loop(
                    std::mem::transmute(values.as_mut_ptr()),
                    &twiddle_dbls
                        .iter()
                        .skip(3)
                        .map(|x| x.as_slice())
                        .collect::<Vec<_>>(),
                    7,
                    4,
                    0,
                )
            },
            BatchSize::LargeInput,
        );
    });

    let (values, _twiddle_dbls) = prepare_values(20);
    group.throughput(Throughput::Bytes(4 << 20));
    group.bench_function("avx transpose_vecs 2^20", |b| {
        b.iter_batched(
            || values.clone().data,
            |mut values| unsafe {
                transpose_vecs(std::mem::transmute(values.as_mut_ptr()), (20 - 4) as usize);
            },
            BatchSize::LargeInput,
        );
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_rfft(c: &mut criterion::Criterion) {
    use stwo_prover::core::backend::avx512::fft::rfft;
    use stwo_prover::core::backend::avx512::PackedBaseField;
    use stwo_prover::platform;
    if !platform::avx512_detected() {
        return;
    }

    const LOG_SIZE: u32 = 20;
    let (values, twiddle_dbls) = prepare_values(LOG_SIZE);

    c.bench_function("avx rfft 20bit", |b| {
        b.iter_with_large_drop(|| unsafe {
            let mut target = Vec::<PackedBaseField>::with_capacity(values.data.len());
            #[allow(clippy::uninit_vec)]
            target.set_len(values.data.len());

            rfft::fft(
                black_box(std::mem::transmute(values.data.as_ptr())),
                std::mem::transmute(target.as_mut_ptr()),
                &twiddle_dbls
                    .iter()
                    .map(|x| x.as_slice())
                    .collect::<Vec<_>>(),
                LOG_SIZE as usize,
            );
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn prepare_values(
    log_size: u32,
) -> (
    stwo_prover::core::backend::avx512::BaseFieldVec,
    Vec<Vec<i32>>,
) {
    use stwo_prover::core::backend::avx512::fft::ifft::get_itwiddle_dbls;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let values = (0..domain.size())
        .map(|i| BaseField::from_u32_unchecked(i as u32))
        .collect::<Vec<_>>();
    let values = stwo_prover::core::backend::avx512::BaseFieldVec::from_iter(values);
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
    (values, twiddle_dbls)
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = avx512_ifft, avx512_ifft_parts, avx512_rfft, simd_ifft, simd_ifft_parts, simd_rfft);
#[cfg(not(target_arch = "x86_64"))]
criterion::criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = simd_ifft, simd_ifft_parts, simd_rfft);
criterion::criterion_main!(benches);
