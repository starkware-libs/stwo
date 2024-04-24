#![feature(iter_array_chunks)]

use criterion::{BenchmarkId, Criterion, Throughput};
use stwo_prover::core::backend::avx512::fft::ifft::get_itwiddle_dbls;
use stwo_prover::core::backend::avx512::fft::transpose_vecs;
use stwo_prover::core::backend::avx512::{BaseFieldVec, PackedBaseField};
use stwo_prover::core::poly::circle::CanonicCoset;
use stwo_verifier::core::fields::m31::BaseField;

#[cfg(target_arch = "x86_64")]
pub fn avx512_ifft(c: &mut criterion::Criterion) {
    use stwo_prover::core::backend::avx512::fft::ifft;
    use stwo_prover::platform;
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
                    std::mem::transmute(values.data.as_mut_ptr()),
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
pub fn avx512_ifft_parts(c: &mut criterion::Criterion) {
    use stwo_prover::core::backend::avx512::fft::ifft;
    use stwo_prover::platform;
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
                std::mem::transmute(values.data.as_mut_ptr()),
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
                std::mem::transmute(values.data.as_mut_ptr()),
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
            transpose_vecs(
                std::mem::transmute(values.data.as_mut_ptr()),
                (20 - 4) as usize,
            );
        });
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_rfft(c: &mut criterion::Criterion) {
    use stwo_prover::core::backend::avx512::fft::rfft;
    use stwo_prover::platform;
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
                std::mem::transmute(values.data.as_ptr()),
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
criterion::criterion_group!(
    name=avx_ifft;
    config = Criterion::default().sample_size(10);
    targets=avx512_ifft, avx512_ifft_parts, avx512_rfft);
criterion::criterion_main!(avx_ifft);
