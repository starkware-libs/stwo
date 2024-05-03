use criterion::Criterion;
use num_traits::One;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_prover::core::backend::simd::m31::N_LANES;
use stwo_prover::core::fields::cm31::CM31;
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::qm31::SecureField;

pub const N_ELEMENTS: usize = 1 << 16;
pub const N_STATE_ELEMENTS: usize = 8;

pub fn m31_operations_bench(c: &mut criterion::Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<M31> = (0..N_ELEMENTS).map(|_| rng.gen()).collect();
    let mut state: [M31; N_STATE_ELEMENTS] = rng.gen();

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
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<CM31> = (0..N_ELEMENTS).map(|_| rng.gen()).collect();
    let mut state: [CM31; N_STATE_ELEMENTS] = rng.gen();

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
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<SecureField> = (0..N_ELEMENTS).map(|_| rng.gen()).collect();
    let mut state: [SecureField; N_STATE_ELEMENTS] = rng.gen();

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
    use stwo_prover::core::backend::avx512::m31::{PackedBaseField, K_BLOCK_SIZE};
    use stwo_prover::platform;

    if !platform::avx512_detected() {
        return;
    }

    let mut rng = SmallRng::seed_from_u64(0);
    let mut elements: Vec<PackedBaseField> = Vec::new();
    let mut states: Vec<PackedBaseField> =
        vec![PackedBaseField::from_array([1.into(); K_BLOCK_SIZE]); N_STATE_ELEMENTS];

    for _ in 0..(N_ELEMENTS / K_BLOCK_SIZE) {
        elements.push(PackedBaseField::from_array(rng.gen()));
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

pub fn simd_m31_operations_bench(c: &mut criterion::Criterion) {
    use stwo_prover::core::backend::simd::m31::PackedBaseField;

    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<PackedBaseField> = (0..N_ELEMENTS / N_LANES).map(|_| rng.gen()).collect();
    let mut states = vec![PackedBaseField::broadcast(BaseField::one()); N_STATE_ELEMENTS];

    c.bench_function("mul_simd", |b| {
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

    c.bench_function("add_simd", |b| {
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

    c.bench_function("sub_simd", |b| {
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
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = m31_operations_bench, cm31_operations_bench, qm31_operations_bench, 
        avx512_m31_operations_bench, simd_m31_operations_bench);
#[cfg(not(target_arch = "x86_64"))]
criterion::criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = m31_operations_bench, cm31_operations_bench, qm31_operations_bench, 
        simd_m31_operations_bench);
criterion::criterion_main!(benches);
