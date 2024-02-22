#![feature(iter_array_chunks)]

use criterion::Criterion;

#[cfg(target_arch = "x86_64")]
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
    let twiddle_dbls = (0..(LOG_SIZE as i32 - 1))
        .map(|log_n| (0..(1 << log_n)).collect::<Vec<_>>())
        .rev()
        .collect::<Vec<_>>();
    // TODO(spapini): When batch inverse is implemented, replace with real twiddles.
    // let twiddle_dbls = get_itwiddle_dbls(domain);

    c.bench_function("avx ifft", |b| {
        b.iter(|| unsafe {
            ifft::ifft(
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddle_dbls[..],
                LOG_SIZE as usize,
            );
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=avx_ifft;
    config = Criterion::default().sample_size(10);
    targets=avx512_ifft);
criterion::criterion_main!(avx_ifft);
