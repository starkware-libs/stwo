use prover_research::core::fields::m31::{M31, P};
use rand::{rngs::ThreadRng, Rng};
pub const N_ELEMENTS: usize = 1 << 16;
pub const N_STATE_ELEMENTS: usize = 8;

pub fn get_random_element(rng: &mut ThreadRng) -> M31 {
    M31::from_u32_unchecked(rng.gen::<u32>() % P)
}

pub fn field_operations_bench(c: &mut criterion::Criterion) {
    let mut rng = rand::thread_rng();
    let mut elements: Vec<M31> = Vec::new();
    let mut state: [M31; N_STATE_ELEMENTS] =
        [(); N_STATE_ELEMENTS].map(|_| get_random_element(&mut rng));

    for _ in 0..(N_ELEMENTS) {
        elements.push(get_random_element(&mut rng));
    }

    c.bench_function("mul", |b| {
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

    c.bench_function("add", |b| {
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

pub fn avx512_field_operations_bench(c: &mut criterion::Criterion) {
    use prover_research::platform;
    if !platform::avx512_detected() {
        return;
    }

    // AVX512 is supported by the platform.
    use prover_research::core::fields::avx512_m31::{K_BLOCK_SIZE, M31AVX512, M512ONE};

    let mut rng = rand::thread_rng();
    let mut elements: Vec<M31AVX512> = Vec::new();
    let mut states: Vec<M31AVX512> =
        vec![M31AVX512::from_m512_unchecked(M512ONE); N_STATE_ELEMENTS];

    for _ in 0..(N_ELEMENTS / K_BLOCK_SIZE) {
        elements.push(M31AVX512::from_vec(&vec![
            get_random_element(&mut rng);
            K_BLOCK_SIZE
        ]));
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
}

criterion::criterion_group!(
    benches,
    field_operations_bench,
    avx512_field_operations_bench
);
criterion::criterion_main!(benches);
