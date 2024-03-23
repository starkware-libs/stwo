#![feature(iter_array_chunks)]

use criterion::Criterion;
use stwo::core::backend::avx512::fft::ifft::get_itwiddle_dbls;
use stwo::core::backend::avx512::fft::transpose_vecs;

pub fn avx512_ifft(c: &mut criterion::Criterion) {
    use stwo::core::backend::avx512::fft::ifft;
    use stwo::core::backend::avx512::BaseFieldVec;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::platform;
    if !platform::avx512_detected() {
        return;
    }

    let mut group = c.benchmark_group("iffts");
    for log_size in 16..=28 {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = (0..domain.size())
            .map(|i| BaseField::from_u32_unchecked(i as u32))
            .collect::<Vec<_>>();
        let mut values = BaseFieldVec::from_iter(values);
        let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);

        group.throughput(criterion::Throughput::Bytes(4 << log_size));
        group.bench_function(criterion::BenchmarkId::new("avx ifft", log_size), |b| {
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
pub fn avx512_ifft_parts(c: &mut criterion::Criterion) {
    use stwo::core::backend::avx512::fft::ifft;
    use stwo::core::backend::avx512::BaseFieldVec;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::platform;
    if !platform::avx512_detected() {
        return;
    }

    let log_size = 20;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let values = (0..domain.size())
        .map(|i| BaseField::from_u32_unchecked(i as u32))
        .collect::<Vec<_>>();
    let mut values = BaseFieldVec::from_iter(values);
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
    let mut group = c.benchmark_group("ifft parts");

    group.throughput(criterion::Throughput::Bytes(4 << 14));
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

    group.throughput(criterion::Throughput::Bytes(4 << 20));
    group.bench_function("avx transpose_vecs 2^20", |b| {
        b.iter(|| unsafe {
            transpose_vecs(
                std::mem::transmute(values.data.as_mut_ptr()),
                (log_size - 4) as usize,
            );
        });
    });
}

criterion::criterion_group!(
    name=avx_ifft;
    config = Criterion::default().sample_size(10);
    targets=avx512_ifft,avx512_ifft_parts);
criterion::criterion_main!(avx_ifft);
