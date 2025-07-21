use std::sync::OnceLock;

use core_affinity::CoreId;

use crate::prover::backend::simd::fft::rfft::{fft1_loop, fft2_loop, fft3_loop, fft_vecwise_loop};
use crate::prover::backend::simd::fft::{transpose_vecs, CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
use crate::prover::backend::simd::m31::LOG_N_LANES;
use crate::prover::backend::simd::utils::{UnsafeConst, UnsafeMut};

pub struct CpuTopology {
    pub available_cores: Vec<CoreId>,
}
impl CpuTopology {
    pub fn detect() -> Self {
        Self {
            available_cores: core_affinity::get_core_ids().unwrap_or_default(),
        }
    }
}

static CPU_TOPOLOGY: OnceLock<CpuTopology> = OnceLock::new();

/// Performs a Circle Fast Fourier Transform (CFFT) on the given values.
///
/// # Arguments
///
/// * `src`: A pointer to the values to transform.
/// * `dst`: A pointer to the destination array.
/// * `twiddle_dbl`: A reference to the doubles of the twiddle factors.
/// * `log_n_elements`: The log of the number of elements in the `values` array.
///
/// # Panics
///
/// This function will panic if `log_n_elements` is less than `MIN_FFT_LOG_SIZE`.
///
/// # Safety
///
/// Behavior is undefined if `src` and `dst` do not have the same alignment as [`PackedBaseField`].
pub unsafe fn fft(src: *const u32, dst: *mut u32, twiddle_dbl: &[&[u32]], log_n_elements: usize) {
    assert!(log_n_elements >= MIN_FFT_LOG_SIZE as usize);
    let log_n_vecs = log_n_elements - LOG_N_LANES as usize;
    if log_n_elements <= CACHED_FFT_LOG_SIZE as usize {
        fft_lower_with_vecwise(src, dst, twiddle_dbl, log_n_elements, log_n_elements);
        return;
    }

    let fft_layers_pre_transpose = log_n_vecs.div_ceil(2);
    let fft_layers_post_transpose = log_n_vecs / 2;
    fft_lower_without_vecwise(
        src,
        dst,
        &twiddle_dbl[(3 + fft_layers_pre_transpose)..],
        log_n_elements,
        fft_layers_post_transpose,
    );
    // transpose_vecs(dst, log_n_vecs);
    fft_lower_with_vecwise(
        dst,
        dst,
        &twiddle_dbl[..3 + fft_layers_pre_transpose],
        log_n_elements,
        fft_layers_pre_transpose + LOG_N_LANES as usize,
    );
}

/// # Safety
///
/// Behavior is undefined if `src` and `dst` do not have the same alignment as [`PackedBaseField`].
pub unsafe fn fft_lower_without_vecwise(
    src: *const u32,
    dst: *mut u32,
    twiddle_dbl: &[&[u32]],
    log_size: usize,
    fft_layers: usize,
) {
    assert!(log_size >= LOG_N_LANES as usize);

    let src = UnsafeConst(src);
    let dst = UnsafeMut(dst);

    let range_size: usize = 1 << (log_size - fft_layers - LOG_N_LANES as usize);
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let chunk_size = range_size.div_ceil(num_threads);

    std::thread::scope(|scope| {
        for thread_id in 0..num_threads {
            let start = thread_id * chunk_size;
            let end = (start + chunk_size).min(range_size);

            let src = UnsafeConst(src.get());
            let dst = UnsafeMut(dst.get());

            if start >= range_size {
                break;
            }

            let core_id = CPU_TOPOLOGY
                .get_or_init(CpuTopology::detect)
                .available_cores[thread_id % CPU_TOPOLOGY.get().unwrap().available_cores.len()];

            scope.spawn(move || {
                core_affinity::set_for_current(core_id);
                for index_h in start..end {
                    let mut src = src.get();
                    let dst = dst.get();
                    for layer in (0..fft_layers).step_by(3).rev() {
                        let fixed_layer = layer + LOG_N_LANES as usize;
                        match fft_layers - layer {
                            1 => {
                                fft1_loop(src, dst, &twiddle_dbl[layer..], fixed_layer, index_h);
                            }
                            2 => {
                                fft2_loop(src, dst, &twiddle_dbl[layer..], fixed_layer, index_h);
                            }
                            _ => {
                                fft3_loop(
                                    src,
                                    dst,
                                    &twiddle_dbl[layer..],
                                    fft_layers - layer - 3,
                                    fixed_layer,
                                    index_h,
                                );
                            }
                        }
                        src = dst;
                    }
                }
            });
        }
    });
}

/// Computes partial fft on `2^log_size` M31 elements.
///
/// # Arguments
///
/// - `src`: A pointer to the values to transform, aligned to 64 bytes.
/// - `dst`: A pointer to the destination array, aligned to 64 bytes.
/// - `twiddle_dbl`: The doubles of the twiddle factors for each layer of the the fft. Layer `i`
///   holds `2^(log_size - 1 - i)` twiddles.
/// - `log_size`: The log of the number of number of M31 elements in the array.
/// - `fft_layers`: The number of fft layers to apply, out of log_size.
///
/// # Panics
///
/// Panics if `log_size` is not at least 5.
///
/// # Safety
///
/// `src` and `dst` must have same alignment as [`PackedBaseField`].
/// `fft_layers` must be at least 5.
pub unsafe fn fft_lower_with_vecwise(
    src: *const u32,
    dst: *mut u32,
    twiddle_dbl: &[&[u32]],
    log_size: usize,
    fft_layers: usize,
) {
    const VECWISE_FFT_BITS: usize = LOG_N_LANES as usize + 1;
    assert!(log_size >= VECWISE_FFT_BITS);

    assert_eq!(twiddle_dbl[0].len(), 1 << (log_size - 2));

    let src = UnsafeConst(src);
    let dst = UnsafeMut(dst);

    let range_size: usize = 1 << (log_size - fft_layers);
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let chunk_size = range_size.div_ceil(num_threads);

    std::thread::scope(|scope| {
        for thread_id in 0..num_threads {
            let start = thread_id * chunk_size;
            let end = (start + chunk_size).min(range_size);

            let src = UnsafeConst(src.get());
            let dst = UnsafeMut(dst.get());

            if start >= range_size {
                break;
            }

            let core_id = CPU_TOPOLOGY
                .get_or_init(CpuTopology::detect)
                .available_cores[thread_id % CPU_TOPOLOGY.get().unwrap().available_cores.len()];
            scope.spawn(move || {
                core_affinity::set_for_current(core_id);
                for index_h in start..end {
                    let mut src = src.get();
                    let dst = dst.get();
                    for layer in (VECWISE_FFT_BITS..fft_layers).step_by(3).rev() {
                        match fft_layers - layer {
                            1 => {
                                fft1_loop(src, dst, &twiddle_dbl[(layer - 1)..], layer, index_h);
                            }
                            2 => {
                                fft2_loop(src, dst, &twiddle_dbl[(layer - 1)..], layer, index_h);
                            }
                            _ => {
                                fft3_loop(
                                    src,
                                    dst,
                                    &twiddle_dbl[(layer - 1)..],
                                    fft_layers - layer - 3,
                                    layer,
                                    index_h,
                                );
                            }
                        }
                        src = dst;
                    }
                    fft_vecwise_loop(
                        src,
                        dst,
                        twiddle_dbl,
                        fft_layers - VECWISE_FFT_BITS,
                        index_h,
                    );
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use core::mem::transmute;

    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain};
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::fft::rfft::get_twiddle_dbls;
    use crate::prover::backend::simd::m31::PackedBaseField;
    use crate::prover::backend::Column;

    #[test]
    fn test_fft_full() {
        for log_size in CACHED_FFT_LOG_SIZE + 1..CACHED_FFT_LOG_SIZE + 7 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let mut rng = SmallRng::seed_from_u64(0);
            let values = (0..domain.size()).map(|_| rng.gen()).collect_vec();
            let twiddle_dbls = get_twiddle_dbls(domain.half_coset);

            let mut res = values.iter().copied().collect::<BaseColumn>();
            unsafe {
                transpose_vecs(
                    transmute::<*mut PackedBaseField, *mut u32>(res.data.as_mut_ptr()),
                    log_size as usize - 4,
                );
                fft(
                    transmute::<*const PackedBaseField, *const u32>(res.data.as_ptr()),
                    transmute::<*mut PackedBaseField, *mut u32>(res.data.as_mut_ptr()),
                    &twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec(),
                    log_size as usize,
                );
            }

            assert_eq!(res.to_cpu(), ground_truth_fft(domain, &values));
        }
    }

    fn ground_truth_fft(domain: CircleDomain, values: &[BaseField]) -> Vec<BaseField> {
        let poly = CpuCirclePoly::new(values.to_vec());
        poly.evaluate(domain).values
    }
}
