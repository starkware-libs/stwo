#![feature(iter_array_chunks)]

#[cfg(target_arch = "x86_64")]
pub fn avx512_ifft(c: &mut criterion::Criterion) {
    use bytemuck::Zeroable;
    use criterion::black_box;
    use stwo::commitment_scheme::blake2_hash::Blake2sHasher;
    use stwo::commitment_scheme::hasher::Hasher;
    use stwo::core::air::evaluation::SecureColumn;
    use stwo::core::backend::avx512::cm31::PackedCM31;
    use stwo::core::backend::avx512::fft::ifft::get_itwiddle_dbls;
    use stwo::core::backend::avx512::fft::{ifft, rfft};
    use stwo::core::backend::avx512::qm31::PackedQM31;
    use stwo::core::backend::avx512::{AVX512Backend, BaseFieldVec};
    use stwo::core::channel::{Blake2sChannel, Channel};
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

    let domain = CanonicCoset::new(LOG_SIZE + 2).circle_domain();
    let values = (0..domain.size()).map(|i| BaseField::from_u32_unchecked(i as u32));
    let mut values = BaseFieldVec::from_iter(values);
    let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);

    c.bench_function("avx fft 28bit", |b| {
        b.iter(|| unsafe {
            rfft::fft(
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddle_dbls
                    .iter()
                    .map(|x| x.as_slice())
                    .collect::<Vec<_>>(),
                LOG_SIZE as usize + 2,
            );
        })
    });

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }

    let values: SecureColumn<AVX512Backend> = test_channel()
        .draw_felts(domain.size())
        .into_iter()
        .collect();
    let packed_midpoint = values.cols[0].data.len() / 2;

    c.bench_function("avx product 26bit", |b| {
        b.iter(|| {
            let mut acc = PackedQM31::zeroed();

            for i in 0..packed_midpoint {
                let lhs = PackedQM31([
                    PackedCM31([values.cols[0].data[i], values.cols[1].data[i]]),
                    PackedCM31([values.cols[2].data[i], values.cols[3].data[i]]),
                ]);

                let rhs = PackedQM31([
                    PackedCM31([
                        values.cols[0].data[i + packed_midpoint],
                        values.cols[1].data[i + packed_midpoint],
                    ]),
                    PackedCM31([
                        values.cols[2].data[i + packed_midpoint],
                        values.cols[3].data[i + packed_midpoint],
                    ]),
                ]);

                let a = lhs * rhs;
                acc += a * a * rhs;
            }

            black_box(acc)
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=avx_ifft;
    config = criterion::Criterion::default().sample_size(10);
    targets=avx512_ifft);
#[cfg(target_arch = "x86_64")]
criterion::criterion_main!(avx_ifft);

#[cfg(not(target_arch = "x86_64"))]
fn main() {}
