extern crate criterion;

use criterion::Criterion;
use prover_research::core::field::field::M31;
use prover_research::core::field::field::P;
use prover_research::core::field::field_avx512::{Consts, kBlockSize, M31Avx512, Operations};
use rand::Rng;

const N_THR_BITS: usize = 3;
const N_THR: usize = 1 << N_THR_BITS;


pub fn mul_bench(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut values: Vec<M31> = Vec::new();
    for _ in 0..(1 << 20) {
        values.push(M31::from_u32_unchecked(rng.gen::<u32>() % P));
    }
    let mut x0 = M31::from_u32_unchecked(rng.gen::<u32>() % P);
    let mut x1 = M31::from_u32_unchecked(rng.gen::<u32>() % P);
    let mut x2 = M31::from_u32_unchecked(rng.gen::<u32>() % P);
    let mut x3 = M31::from_u32_unchecked(rng.gen::<u32>() % P);
    let mut x4 = M31::from_u32_unchecked(rng.gen::<u32>() % P);
    let mut x5 = M31::from_u32_unchecked(rng.gen::<u32>() % P);
    let mut x6 = M31::from_u32_unchecked(rng.gen::<u32>() % P);
    let mut x7 = M31::from_u32_unchecked(rng.gen::<u32>() % P);

    c.bench_function("mul", |b| {
        b.iter(|| {
            for i in (0..values.len()).step_by(kBlockSize) {
                for _ in 0..50 {
                    x0 *= values[i + 0];
                    x1 *= values[i + 1];
                    x2 *= values[i + 2];
                    x3 *= values[i + 3];
                    x4 *= values[i + 4];
                    x5 *= values[i + 5];
                    x6 *= values[i + 6];
                    x7 *= values[i + 7];
                }
            }
        })
    });
}


pub fn mul_avx512_bench(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let cn = Consts::new();
    let mut values: Vec<M31> = Vec::new();
    let mut state: Vec<M31> = Vec::new();
    for _ in 0..(1 << 20) {
        values.push(M31::from_u32_unchecked(rng.gen::<u32>() % P));
    }
    for _ in 0..kBlockSize {
        state.push(M31::from_u32_unchecked(rng.gen::<u32>() % P));
    }

    let mut avx_values: Vec<M31Avx512> = Vec::new();
    let mut avx_state = M31Avx512::load_avx512(&state);
    for i in (0..values.len()).step_by(kBlockSize) {
        avx_values.push(M31Avx512::load_avx512(&values[i..i + kBlockSize].to_vec()));
    }

    c.bench_function("mul_avx512", |b| {
        b.iter(|| {
            for avx_value in avx_values.iter() {
                for _ in 0..50 {
                    avx_state = avx_value.mul(&cn, avx_state);
                }
            }
        })
    });
    // println!("avx_state = {:?}", avx_state.unload_avx512());
}

fn main() {
    let mut criterion = Criterion::default();
    mul_bench(&mut criterion);
    mul_avx512_bench(&mut criterion);
    criterion.final_summary();
}
