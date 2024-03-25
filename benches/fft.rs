#![feature(iter_array_chunks)]

use criterion::Criterion;
use stwo::core::backend::avx512::fft::ifft::get_itwiddle_dbls;
use stwo::core::backend::avx512::fft::rfft::get_twiddle_dbls;
use stwo::core::backend::avx512::PackedBaseField;

pub fn avx512_ifft(c: &mut criterion::Criterion) {
    use stwo::core::backend::avx512::fft::ifft;
    use stwo::core::backend::avx512::BaseFieldVec;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::platform;
    if !platform::avx512_detected() {
        return;
    }

    const LOG_SIZE: u32 = 26;
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let values = (0..domain.size())
        .map(|i| BaseField::from_u32_unchecked(i as u32))
        .collect::<Vec<_>>();

    // Compute.
    let mut values = BaseFieldVec::from_iter(values);
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);

    c.bench_function("avx ifft 26bit", |b| {
        b.iter(|| unsafe {
            ifft::ifft(
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddle_dbls
                    .iter()
                    .map(|x| x.as_slice())
                    .collect::<Vec<_>>(),
                LOG_SIZE as usize,
            );
        })
    });
}

pub fn avx512_rfft(c: &mut criterion::Criterion) {
    use stwo::core::backend::avx512::fft::rfft;
    use stwo::core::backend::avx512::BaseFieldVec;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::platform;
    if !platform::avx512_detected() {
        return;
    }

    const LOG_SIZE: u32 = 20;
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let values = (0..domain.size())
        .map(|i| BaseField::from_u32_unchecked(i as u32))
        .collect::<Vec<_>>();

    // Compute.
    let values = BaseFieldVec::from_iter(values);
    let twiddle_dbls = get_twiddle_dbls(domain.half_coset);

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

criterion::criterion_group!(
    name=avx_ifft;
    config = Criterion::default().sample_size(10);
    targets=avx512_ifft, avx512_rfft);
criterion::criterion_main!(avx_ifft);
