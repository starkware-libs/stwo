#![feature(iter_array_chunks)]

use std::mem::{size_of_val, transmute};

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use itertools::Itertools;
use stwo::core::backend::cpu::CPUCirclePoly;
use stwo::core::backend::CPUBackend;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::{CanonicCoset, PolyOps};

pub fn cpu_rfft(c: &mut Criterion) {
    const LOG_SIZE: u32 = 20;

    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddles = CPUBackend::precompute_twiddles(domain.half_coset);
    let coeffs = (0..domain.size()).map(BaseField::from).collect();
    let poly = CPUCirclePoly::new(coeffs);

    c.bench_function("cpu rfft 20bit", |b| {
        b.iter_batched(
            || poly.clone(),
            |poly| poly.evaluate_with_twiddles(domain, &twiddles),
            BatchSize::LargeInput,
        );
    });
}

pub fn cpu_bit_rev(c: &mut Criterion) {
    use stwo::core::utils::bit_reverse;
    const SIZE: usize = 1 << 24;
    let mut data = (0..SIZE).map(BaseField::from).collect_vec();
    c.bench_function("cpu bit_rev 24bit", |b| {
        b.iter(|| bit_reverse(black_box(&mut data)))
    });
}

pub fn simd_ifft(c: &mut Criterion) {
    use stwo::core::backend::simd::column::BaseFieldVec;
    use stwo::core::backend::simd::fft::ifft::{get_itwiddle_dbls, ifft};

    let mut group = c.benchmark_group("iffts");

    for log_size in 16..=17 {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
        let mut values: BaseFieldVec = (0..domain.size()).map(BaseField::from).collect();
        group.throughput(Throughput::Bytes(size_of_val(&*values.data) as u64));
        group.bench_function(BenchmarkId::new("simd ifft", log_size), |b| {
            b.iter(|| unsafe {
                ifft(
                    black_box(transmute(values.data.as_mut_ptr())),
                    black_box(&twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec()),
                    black_box(log_size as usize),
                );
            });
        });
    }
}

pub fn simd_ifft_parts(c: &mut Criterion) {
    use stwo::core::backend::simd::column::BaseFieldVec;
    use stwo::core::backend::simd::fft::ifft::{get_itwiddle_dbls, ifft3_loop, ifft_vecwise_loop};
    use stwo::core::backend::simd::fft::transpose_vecs;

    const LOG_SIZE: u32 = 14;

    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
    let mut values: BaseFieldVec = (0..domain.size()).map(BaseField::from).collect();

    let mut group = c.benchmark_group("ifft parts");

    // Note: These benchmarks run only on 2^LOG_SIZE elements because of their parameters.
    // Increasing the figure above won't change the runtime of these benchmarks.
    group.throughput(Throughput::Bytes(4 << LOG_SIZE));
    group.bench_function(format!("simd ifft_vecwise_loop 2^{LOG_SIZE}"), |b| {
        b.iter(|| unsafe {
            ifft_vecwise_loop(
                black_box(transmute(values.data.as_mut_ptr())),
                black_box(&twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec()),
                black_box(9),
                black_box(0),
            )
        });
    });
    group.bench_function(format!("simd ifft3_loop 2^{LOG_SIZE}"), |b| {
        b.iter(|| unsafe {
            ifft3_loop(
                black_box(transmute(values.data.as_mut_ptr())),
                black_box(&twiddle_dbls[3..].iter().map(|x| x.as_slice()).collect_vec()),
                black_box(7),
                black_box(4),
                black_box(0),
            )
        });
    });

    const TRANSPOSE_LOG_SIZE: u32 = 20;
    let mut transpose_values: BaseFieldVec =
        (0..1 << TRANSPOSE_LOG_SIZE).map(BaseField::from).collect();
    group.throughput(Throughput::Bytes(4 << TRANSPOSE_LOG_SIZE));
    group.bench_function(format!("simd transpose_vecs 2^{TRANSPOSE_LOG_SIZE}"), |b| {
        b.iter(|| unsafe {
            transpose_vecs(
                black_box(transmute(transpose_values.data.as_mut_ptr())),
                black_box(TRANSPOSE_LOG_SIZE as usize - 4),
            )
        });
    });
}

pub fn simd_rfft(c: &mut Criterion) {
    use stwo::core::backend::simd::column::BaseFieldVec;
    use stwo::core::backend::simd::fft::rfft::{fft, get_twiddle_dbls};
    // use stwo::core::backend::simd::m31::PackedBaseField;

    const LOG_SIZE: u32 = 20;

    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddle_dbls = get_twiddle_dbls(domain.half_coset);
    let values: BaseFieldVec = (0..domain.size()).map(BaseField::from).collect();

    c.bench_function("simd rfft 20bit", |b| {
        b.iter_batched(
            || values.clone(),
            |mut values| unsafe {
                fft(
                    black_box(transmute(values.data.as_ptr())),
                    black_box(transmute(values.data.as_mut_ptr())),
                    black_box(&twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec()),
                    black_box(LOG_SIZE as usize),
                )
            },
            BatchSize::LargeInput,
        );

        // b.iter(|| unsafe {
        //     fft(
        //         black_box(transmute(values.data.as_ptr())),
        //         black_box(transmute(target.as_mut_ptr())),
        //         black_box(&twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec()),
        //         black_box(LOG_SIZE as usize),
        //     )
        // })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_ifft(c: &mut Criterion) {
    use stwo::core::backend::avx512::fft::ifft;
    use stwo::platform;
    if !platform::avx512_detected() {
        return;
    }

    let mut group = c.benchmark_group("iffts");
    for log_size in 16..=28 {
        let (mut values, twiddle_dbls) = prepare_values(log_size);

        group.throughput(Throughput::Bytes(
            (std::mem::size_of::<BaseField>() as u64) << log_size,
        ));
        group.bench_function(BenchmarkId::new("avx ifft", log_size), |b| {
            b.iter(|| unsafe {
                ifft::ifft(
                    transmute(values.data.as_mut_ptr()),
                    &twiddle_dbls
                        .iter()
                        .map(|x| x.as_slice())
                        .collect::<Vec<_>>(),
                    log_size as usize,
                );
            });
        });
    }
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_ifft_parts(c: &mut Criterion) {
    use stwo::core::backend::avx512::fft::ifft;
    use stwo::platform;
    if !platform::avx512_detected() {
        return;
    }

    let (mut values, twiddle_dbls) = prepare_values(14);
    let mut group = c.benchmark_group("ifft parts");

    // Note: These benchmarks run only on 2^14 elements ebcause of their parameters.
    // Increasing the figure above won't change the runtime of these benchmarks.
    group.throughput(Throughput::Bytes(4 << 14));
    group.bench_function("avx ifft_vecwise_loop 2^14", |b| {
        b.iter(|| unsafe {
            ifft::ifft_vecwise_loop(
                transmute(values.data.as_mut_ptr()),
                &twiddle_dbls
                    .iter()
                    .map(|x| x.as_slice())
                    .collect::<Vec<_>>(),
                9,
                0,
            );
        });
    });

    group.bench_function("avx ifft3_loop 2^14", |b| {
        b.iter(|| unsafe {
            ifft::ifft3_loop(
                transmute(values.data.as_mut_ptr()),
                &twiddle_dbls
                    .iter()
                    .skip(3)
                    .map(|x| x.as_slice())
                    .collect::<Vec<_>>(),
                7,
                4,
                0,
            );
        });
    });

    let (mut values, _twiddle_dbls) = prepare_values(20);
    group.throughput(Throughput::Bytes(4 << 20));
    group.bench_function("avx transpose_vecs 2^20", |b| {
        b.iter(|| unsafe {
            transpose_vecs(transmute(values.data.as_mut_ptr()), (20 - 4) as usize);
        });
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_rfft(c: &mut Criterion) {
    use stwo::core::backend::avx512::fft::rfft;
    use stwo::platform;
    if !platform::avx512_detected() {
        return;
    }

    const LOG_SIZE: u32 = 20;
    let (values, twiddle_dbls) = prepare_values(LOG_SIZE);

    c.bench_function("avx rfft 20bit", |b| {
        b.iter(|| unsafe {
            let mut target = Vec::<PackedBaseField>::with_capacity(values.data.len());
            #[allow(clippy::uninit_vec)]
            target.set_len(values.data.len());

            rfft::fft(
                transmute(values.data.as_ptr()),
                transmute(target.as_mut_ptr()),
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
fn prepare_values(log_size: u32) -> (BaseFieldVec, Vec<Vec<i32>>) {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let values = (0..domain.size())
        .map(|i| BaseField::from_u32_unchecked(i as u32))
        .collect::<Vec<_>>();
    let values = BaseFieldVec::from_iter(values);
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
    (values, twiddle_dbls)
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    name=ifft;
    config = Criterion::default().sample_size(10);
    targets=simd_ifft, simd_ifft_parts, simd_rfft, avx512_ifft, avx512_ifft_parts, avx512_rfft);
#[cfg(not(target_arch = "x86_64"))]
criterion_group!(
    name=ifft;
    config = Criterion::default().sample_size(10);
    targets=simd_ifft, simd_ifft_parts, simd_rfft, cpu_rfft);
criterion_main!(ifft);
