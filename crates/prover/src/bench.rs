use std::time::{Duration, Instant};

use num_traits::WrappingAdd;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::parallel_iter;

/// Benchmarks memory access by performing random reads on a vector of specified size
/// Returns the total time taken and number of accesses performed
pub fn benchmark_memory_access(size_mb: usize, num_accesses: usize) -> (Duration, usize) {
    // Calculate vector size (1MB = 1024 * 1024 bytes)
    // Using u64 elements (8 bytes each)
    let elements = size_mb * 1024 * 1024 / std::mem::size_of::<u64>();

    // Allocate vector
    let vec: Vec<u64> = (0..elements as u64).collect();

    // Perform random accesses and accumulate to prevent optimization
    let start = Instant::now();
    let iter = parallel_iter!(0..num_accesses);
    let sum: u64 = iter
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(i as u64);
            let idx = rng.gen_range(0..elements);
            let second_idx = rng.gen_range(0..elements);
            let mut sum = 0;
            sum += sum.wrapping_add(&vec[idx]);
            sum += sum.wrapping_add(vec[second_idx]);
            sum
        })
        .sum();

    let duration = start.elapsed();

    // Prevent the compiler from optimizing away the accesses
    if sum == 123456789 {
        println!("Extremely unlikely value encountered");
    }

    (duration, num_accesses)
}

/// Runs a memory benchmark and prints results
pub fn run_memory_benchmark(size_mb: usize, num_accesses: usize) {
    println!(
        "Running memory benchmark with {} MB vector and {} accesses",
        size_mb, num_accesses
    );

    let (duration, accesses) = benchmark_memory_access(size_mb, num_accesses);

    println!("Completed {} random accesses in {:?}", accesses, duration);
    println!("Average time per access: {:?}", duration / accesses as u32);

    let throughput_mb_per_sec = size_mb as f64 / duration.as_secs_f64();
    println!("Memory throughput: {:.2} MB/s", throughput_mb_per_sec);
}

#[test]
fn test_test() {
    run_memory_benchmark(2024, 500000000);
}

#[test]
fn test_test2() {
    use std::mem::transmute;

    use itertools::Itertools;

    use crate::core::backend::simd::column::BaseColumn;
    use crate::core::backend::simd::fft::rfft::get_twiddle_dbls;
    use crate::core::backend::simd::m31::PackedBaseField;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CanonicCoset;
    const LOG_SIZE: u32 = 23;

    let start = Instant::now();
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddle_dbls = get_twiddle_dbls(domain.half_coset);
    let twiddle_dbls_refs = twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec();
    let mut values: BaseColumn = (0..domain.size()).map(BaseField::from).collect();

    #[allow(clippy::uninit_vec)]
    let mut target = Vec::<PackedBaseField>::with_capacity(values.data.len());
    #[allow(clippy::uninit_vec)]
    unsafe {
        target.set_len(values.data.len());
    }
    println!("{}", start.elapsed().as_secs_f64());
    for _i in 0..5000 {
        unsafe {
            crate::core::backend::simd::fft::ifft::ifft(
                transmute::<*mut PackedBaseField, *mut u32>(values.data.as_mut_ptr()),
                std::hint::black_box(&twiddle_dbls_refs),
                std::hint::black_box(LOG_SIZE as usize),
            );
        }
    }

    println!("{:?}", values.data[0]);
}
