use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use num_traits::One;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_prover::core::backend::gpu::m31::{PackedBaseField as GpuPackedBaseField, TestBaseField};
use stwo_prover::core::backend::simd::m31::{PackedBaseField, N_LANES};
use stwo_prover::core::fields::cm31::CM31;
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::qm31::SecureField;

pub const N_ELEMENTS: usize = 1 << 16;
pub const N_STATE_ELEMENTS: usize = 8;

pub fn m31_operations_bench(c: &mut Criterion) {
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

pub fn cm31_operations_bench(c: &mut Criterion) {
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

pub fn qm31_operations_bench(c: &mut Criterion) {
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

pub fn simd_m31_operations_bench(c: &mut Criterion) {
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

pub fn gpu_m31_operations_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut elements: Vec<TestBaseField> =
        (0..N_ELEMENTS).map(|_| TestBaseField(rng.gen())).collect();
    let states: [M31; N_STATE_ELEMENTS] = rng.gen();
    let mut states: [TestBaseField; N_STATE_ELEMENTS] = states.map(TestBaseField);

    // let mut elements: Vec<GpuBaseField> = (0..N_ELEMENTS)
    //     .map(|_| GpuBaseField::from_host(rng.gen()))
    //     .collect();
    // let mut states: Vec<GpuBaseField> = vec![GpuBaseField::one(); N_STATE_ELEMENTS];

    // c.bench_function("mul_gpu", |b| {
    //     b.iter(|| {
    //         for elem in elements.iter() {
    //             for _ in 0..128 {
    //                 for state in states.iter_mut() {
    //                     *state *= elem.clone();
    //                 }
    //             }
    //         }
    //     })
    // });

    c.bench_function("add_gpu", |b| {
        b.iter(|| {
            for elem in elements.iter_mut() {
                for _ in 0..128 {
                    for state in states.iter_mut() {
                        state.add_assign_ref(elem);
                    }
                }
            }
        })
    });

    // c.bench_function("sub_gpu", |b| {
    //     b.iter(|| {
    //         for elem in elements.iter() {
    //             for _ in 0..128 {
    //                 for state in states.iter_mut() {
    //                     *state -= *(elem.clone());
    //                 }
    //             }
    //         }
    //     })
    // });
}

// todo:: cuda function to

pub fn gpu_packed_operations_bench(c: &mut Criterion) {
    fn setup() -> (GpuPackedBaseField, GpuPackedBaseField) {
        let mut rng = SmallRng::seed_from_u64(0);
        let values: [M31; 524288] = (0..524288)
            .map(|_| rng.gen())
            .collect_vec()
            .try_into()
            .unwrap();
        let elements: GpuPackedBaseField = GpuPackedBaseField::from_array(values);
        let states = GpuPackedBaseField::from_array(values); // GpuPackedBaseField::one();
        (elements, states)
    }

    // TODO:: Convert to CUDA function with respective thread blocks for 2d array flattened
    // // The Vec State is flattened into CudaSlice and Vec Element is adjusted respectively
    fn setup_cuda_parallel() -> (Vec<GpuPackedBaseField>, GpuPackedBaseField) {
        let mut rng = SmallRng::seed_from_u64(0);
        const SIZE: usize = N_STATE_ELEMENTS * N_LANES;
        const ELEMENTS_SIZE: usize = N_ELEMENTS / SIZE;

        let element_values: Vec<GpuPackedBaseField> = (0..ELEMENTS_SIZE)
            .map(|_| {
                GpuPackedBaseField::from_array::<SIZE>(
                    (0..SIZE)
                        .map(|_| rng.gen())
                        .collect_vec()
                        .try_into()
                        .unwrap(),
                )
            })
            .collect();
        let state_values: GpuPackedBaseField = GpuPackedBaseField::one(); //
        GpuPackedBaseField::broadcast(M31(1), Some(SIZE));
        (element_values, state_values)
    }

    c.bench_function("mul_gpu_amortized_fixed", |b| {
        // let (elements, states) = ;
        b.iter_batched(
            || setup_cuda_parallel(),
            |(elements, states)| {
                for element in elements.iter() {
                    for _ in 0..128 {
                        states.mul_assign_ref(&element);
                    }
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    c.bench_function("mul_gpu_non_amortized", |b| {
        let (elements, states) = setup();
        b.iter(|| {
            for _ in 0..128 {
                states.mul_assign_ref(&elements);
            }
        })
    });

    c.bench_function("mul_gpu_amortized", |b| {
        b.iter_batched(
            || setup(),
            |(elements, states)| {
                for _ in 0..128 {
                    states.mul_assign_ref(&elements);
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    c.bench_function("add_gpu_non_amortized", |b| {
        let (elements, states) = setup();
        b.iter(|| {
            for _ in 0..128 {
                states.add_assign_ref(&elements);
            }
        })
    });

    c.bench_function("add_gpu_amortized", |b| {
        b.iter_batched(
            || setup(),
            |(elements, states)| {
                for _ in 0..128 {
                    states.add_assign_ref(&elements);
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    c.bench_function("sub_gpu_non_amortized", |b| {
        let (elements, states) = setup();
        b.iter(|| {
            for _ in 0..128 {
                states.sub_assign_ref(&elements);
            }
        })
    });

    c.bench_function("sub_gpu_amortized", |b| {
        b.iter_batched(
            || setup(),
            |(elements, states)| {
                for _ in 0..128 {
                    states.sub_assign_ref(&elements);
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets =  /*m31_operations_bench, cm31_operations_bench, qm31_operations_bench,
        simd_m31_operations_bench,*/  gpu_packed_operations_bench);
criterion_main!(benches);
