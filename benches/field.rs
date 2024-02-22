use rand::rngs::ThreadRng;
use rand::Rng;
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::SecureField;
pub const N_ELEMENTS: usize = 1 << 16;
pub const N_STATE_ELEMENTS: usize = 8;

#[cfg(target_arch = "x86_64")]
use stwo::core::fields::avx512_m31::{K_BLOCK_SIZE, M31AVX512, M512ONE};
use stwo::platform;

pub fn get_random_m31_element(rng: &mut ThreadRng) -> M31 {
    M31::from_u32_unchecked(rng.gen::<u32>() % P)
}

pub fn get_random_cm31_element(rng: &mut ThreadRng) -> CM31 {
    CM31::from_m31(get_random_m31_element(rng), get_random_m31_element(rng))
}

pub fn get_random_qm31_element(rng: &mut ThreadRng) -> SecureField {
    SecureField::from_m31(
        get_random_m31_element(rng),
        get_random_m31_element(rng),
        get_random_m31_element(rng),
        get_random_m31_element(rng),
    )
}

pub fn m31_operations_bench(c: &mut criterion::Criterion) {
    let mut rng = rand::thread_rng();
    let mut elements: Vec<M31> = Vec::new();
    let mut state: [M31; N_STATE_ELEMENTS] =
        [(); N_STATE_ELEMENTS].map(|_| get_random_m31_element(&mut rng));

    for _ in 0..(N_ELEMENTS) {
        elements.push(get_random_m31_element(&mut rng));
    }

    c.bench_function("M31 mul", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem *= *elem;
                    }
                }
            }
        })
    });

    c.bench_function("M31 add", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem += *elem;
                    }
                }
            }
        })
    });
}

pub fn cm31_operations_bench(c: &mut criterion::Criterion) {
    let mut rng = rand::thread_rng();
    let mut elements: Vec<CM31> = Vec::new();
    let mut state: [CM31; N_STATE_ELEMENTS] =
        [(); N_STATE_ELEMENTS].map(|_| get_random_cm31_element(&mut rng));

    for _ in 0..(N_ELEMENTS) {
        elements.push(get_random_cm31_element(&mut rng));
    }

    c.bench_function("CM31 mul", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem *= *elem;
                    }
                }
            }
        })
    });

    c.bench_function("CM31 add", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem += *elem;
                    }
                }
            }
        })
    });
}

pub fn qm31_operations_bench(c: &mut criterion::Criterion) {
    let mut rng = rand::thread_rng();
    let mut elements: Vec<SecureField> = Vec::new();
    let mut state: [SecureField; N_STATE_ELEMENTS] =
        [(); N_STATE_ELEMENTS].map(|_| get_random_qm31_element(&mut rng));

    for _ in 0..(N_ELEMENTS) {
        elements.push(get_random_qm31_element(&mut rng));
    }

    c.bench_function("SecureField mul", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem *= *elem;
                    }
                }
            }
        })
    });

    c.bench_function("SecureField add", |b| {
        b.iter(|| {
            for elem in &elements {
                for _ in 0..128 {
                    for state_elem in &mut state {
                        *state_elem += *elem;
                    }
                }
            }
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_m31_operations_bench(c: &mut criterion::Criterion) {
    if !platform::avx512_detected() {
        return;
    }

    let mut rng = rand::thread_rng();
    let mut elements: Vec<M31AVX512> = Vec::new();
    let mut states: Vec<M31AVX512> =
        vec![M31AVX512::from_m512_unchecked(M512ONE); N_STATE_ELEMENTS];

    for _ in 0..(N_ELEMENTS / K_BLOCK_SIZE) {
        elements.push(M31AVX512::from_slice(
            &[get_random_m31_element(&mut rng); K_BLOCK_SIZE],
        ));
    }

    c.bench_function("mul_avx512", |b| {
        b.iter(|| {
            for elem in elements.iter() {
                for _ in 0..128 {
                    for state in states.iter_mut() {
                        *state *= *elem;
                    }
                }
            }
        })
    });

    c.bench_function("add_avx512", |b| {
        b.iter(|| {
            for elem in elements.iter() {
                for _ in 0..128 {
                    for state in states.iter_mut() {
                        *state += *elem;
                    }
                }
            }
        })
    });

    c.bench_function("sub_avx512", |b| {
        b.iter(|| {
            for elem in elements.iter() {
                for _ in 0..128 {
                    for state in states.iter_mut() {
                        *state -= *elem;
                    }
                }
            }
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    m31_benches,
    m31_operations_bench,
    avx512_m31_operations_bench
);
#[cfg(not(target_arch = "x86_64"))]
criterion::criterion_group!(m31_benches, m31_operations_bench);

criterion::criterion_group!(
    field_comparison,
    m31_operations_bench,
    cm31_operations_bench,
    qm31_operations_bench
);
criterion::criterion_main!(field_comparison);
