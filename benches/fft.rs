#![feature(iter_array_chunks)]

use criterion::Criterion;
use stwo::core::backend::avx512::fft::ifft::get_itwiddle_dbls;

pub fn avx512_ifft(c: &mut criterion::Criterion) {
    use stwo::core::backend::avx512::fft::ifft;
    use stwo::core::backend::avx512::BaseFieldVec;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::platform;
    if !platform::avx512_detected() {
        return;
    }

    const LOG_SIZE: u32 = 28;
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let values = (0..domain.size())
        .map(|i| BaseField::from_u32_unchecked(i as u32))
        .collect::<Vec<_>>();

    // Compute.
    let mut values = BaseFieldVec::from_iter(values);
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);

    c.bench_function("avx ifft", |b| {
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

criterion::criterion_group!(
    name=avx_ifft;
    config = Criterion::default().sample_size(10);
    targets=avx512_ifft);
criterion::criterion_main!(avx_ifft);
