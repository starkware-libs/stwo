#![allow(unused_variables)]
use std::simd::{simd_swizzle, u32x16, u32x8};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::m31::PackedBaseField;
use super::utils::UnsafeMut;
use crate::core::fields::m31::P;
use crate::parallel_iter;

pub mod ifft;
pub mod rfft;

pub const CACHED_FFT_LOG_SIZE: u32 = 16;

pub const MIN_FFT_LOG_SIZE: u32 = 5;

// TODO(andrew): FFTs return a redundant representation, that can get the value P. need to deal with
// it. Either: reduce before commitment or regenerate proof with new seed if redundant value
// decommitted.

/// Transposes the SIMD vectors in the given array.
///
/// Swaps the bit index abc <-> cba, where |a|=|c| and |b| = 0 or 1, according to the parity of
/// `log_n_vecs`.
/// When log_n_vecs is odd, transforms the index abc <-> cba, w
///
/// # Arguments
///
/// - `values`: A mutable pointer to the values that are to be transposed.
/// - `log_n_vecs`: The log of the number of SIMD vectors in the `values` array.
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`u32x16`].
pub unsafe fn transpose_vecs(values: *mut u32, log_n_vecs: usize) {
    let half = log_n_vecs / 2;

    let values = UnsafeMut(values);
    parallel_iter!(0..1 << half).for_each(|a| {
        let values = values.get();
        for b in 0..1 << (log_n_vecs & 1) {
            for c in 0..1 << half {
                let i = (a << (log_n_vecs - half)) | (b << half) | c;
                let j = (c << (log_n_vecs - half)) | (b << half) | a;
                if i >= j {
                    continue;
                }
                let val0 = load(values.add(i << 4).cast_const());
                let val1 = load(values.add(j << 4).cast_const());
                store(values.add(i << 4), val1);
                store(values.add(j << 4), val0);
            }
        }
    });
}

/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`u32x16`].
#[cfg(feature = "parallel")]
pub unsafe fn transpose_vecs2(
    values: *mut u32,
    log_n_vecs: usize,
    log_tile_edge: usize,
    buffer0: *mut u32,
    buffer1: *mut u32,
) {
    let n_vecs = 1 << log_n_vecs;
    let half = log_n_vecs / 2;
    let log_tile_edge = std::cmp::min(half, log_tile_edge);
    let tile_edge = 1 << log_tile_edge;
    let tile_size = tile_edge * tile_edge;
    let log_edge = half - log_tile_edge;
    let log_row_length = log_n_vecs.div_ceil(2);

    let buffer0 = UnsafeMut(buffer0);
    let buffer1 = UnsafeMut(buffer1);

    // Precompute all tile pairs (r,c) with r <= c
    let mut tile_pairs = vec![];
    for r in 0..1 << log_edge {
        for c in r..1 << log_edge {
            tile_pairs.push((r, c));
        }
    }

    for b in 0..=(log_n_vecs & 1) {
        // Parallel over tile-pairs
        let base = UnsafeMut(values);
        tile_pairs.iter().for_each(|(r, c)| {
            let vals = base.get();
            let row_off = r * tile_edge;
            let col_off = c * tile_edge;

            if r == c {
                // In-place within diagonal tile T_{r,r}
                for i in 0..tile_edge {
                    for j in (i + 1)..tile_edge {
                        let idx_i = ((row_off + i) << log_row_length) + (b << half) + j + col_off;
                        let idx_j = perm_index(idx_i, log_n_vecs, half);

                        // Bounds checking
                        debug_assert!(idx_i < n_vecs, "idx_i {} >= n_vecs {}", idx_i, n_vecs);
                        debug_assert!(idx_j < n_vecs, "idx_j {} >= n_vecs {}", idx_j, n_vecs);

                        let ptr_i = vals.add(idx_i << 4);
                        let ptr_j = vals.add(idx_j << 4);
                        let v0 = load(ptr_i.cast_const());
                        let v1 = load(ptr_j.cast_const());
                        store(ptr_i, v1);
                        store(ptr_j, v0);
                    }
                }
            } else {
                // // Swap off-diagonal tile T_{r,c} with transpose of T_{c,r}
                // for i in 0..tile_edge {
                //     for j in 0..tile_edge {
                //         let idx_i = ((row_off + i) << log_row_length) + j + col_off;
                //         let idx_j = perm_index(idx_i, log_n_vecs, half);

                //         let ptr_i = vals.add(idx_i << 4);
                //         let ptr_j = vals.add(idx_j << 4);
                //         let v0 = load(ptr_i.cast_const());
                //         let v1 = load(ptr_j.cast_const());
                //         store(ptr_i, v1);
                //         store(ptr_j, v0);
                //     }
                // }

                // Copy T_{r,c} and T_{c,r} to a buffer.
                let buffer0 = buffer0.get();
                let buffer1 = buffer1.get();
                for i in 0..tile_edge {
                    for j in 0..tile_edge {
                        let idx = ((row_off + i) << log_row_length) + (b << half) + j + col_off;
                        debug_assert!(idx < n_vecs, "Copy idx {} >= n_vecs {}", idx, n_vecs);

                        let offset_in_buffer = i * tile_edge * 16 + j * 16;
                        debug_assert!(
                            offset_in_buffer + 16 <= tile_size * 16,
                            "Buffer overflow: {} + 16 > {}",
                            offset_in_buffer,
                            tile_size * 16
                        );

                        let ptr = buffer0.add(offset_in_buffer);
                        store(ptr, load(vals.add(idx << 4).cast_const()));

                        let idx = perm_index(idx, log_n_vecs, half);
                        debug_assert!(idx < n_vecs, "Perm idx {} >= n_vecs {}", idx, n_vecs);
                        let ptr = buffer1.add(offset_in_buffer);
                        store(ptr, load(vals.add(idx << 4).cast_const()));
                    }
                }

                // Copy the buffers to T_{c,r} and T_{r,c}
                for i in 0..tile_edge {
                    for j in 0..tile_edge {
                        let offset_in_buffer = i * tile_edge * 16 + j * 16;
                        debug_assert!(
                            offset_in_buffer + 16 <= tile_size * 16,
                            "Final buffer overflow: {} + 16 > {}",
                            offset_in_buffer,
                            tile_size * 16
                        );

                        let ptr = buffer1.add(offset_in_buffer);
                        let idx = ((row_off + i) << log_row_length) + (b << half) + j + col_off;
                        debug_assert!(idx < n_vecs, "Final idx {} >= n_vecs {}", idx, n_vecs);
                        store(vals.add(idx << 4), load(ptr.cast_const()));

                        let idx = perm_index(idx, log_n_vecs, half);
                        debug_assert!(idx < n_vecs, "Final perm idx {} >= n_vecs {}", idx, n_vecs);
                        let ptr = buffer0.add(offset_in_buffer);
                        store(vals.add(idx << 4), load(ptr.cast_const()));
                    }
                }
            }
        });
    }
}

#[allow(unused)]
/// Compute the permuted index swapping bits abc <-> cba.
const fn perm_index(x: usize, log_n: usize, half: usize) -> usize {
    let a = x >> (log_n - half);
    let b = (x >> half) & (log_n & 1);
    let c = x & ((1 << half) - 1);
    (c << (log_n - half)) | (b << half) | a
}

/// Computes the twiddles for the first fft layer from the second, and loads both to SIMD registers.
///
/// Returns the twiddles for the first layer and the twiddles for the second layer.
pub fn compute_first_twiddles(twiddle1_dbl: u32x8) -> (u32x16, u32x16) {
    // Start by loading the twiddles for the second layer (layer 1):
    let t1 = simd_swizzle!(
        twiddle1_dbl,
        twiddle1_dbl,
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
    );

    // The twiddles for layer 0 can be computed from the twiddles for layer 1.
    // Since the twiddles are bit reversed, we consider the circle domain in bit reversed order.
    // Each consecutive 4 points in the bit reversed order of a coset form a circle coset of size 4.
    // A circle coset of size 4 in bit reversed order looks like this:
    //   [(x, y), (-x, -y), (y, -x), (-y, x)]
    // Note: This is related to the choice of M31_CIRCLE_GEN, and the fact the a quarter rotation
    //   is (0,-1) and not (0,1). (0,1) would yield another relation.
    // The twiddles for layer 0 are the y coordinates:
    //   [y, -y, -x, x]
    // The twiddles for layer 1 in bit reversed order are the x coordinates:
    //   [x, y]
    // Works also for inverse of the twiddles.

    // The twiddles for layer 0 are computed like this:
    //   t0[4i:4i+3] = [t1[2i+1], -t1[2i+1], -t1[2i], t1[2i]]
    // Xoring a double twiddle with P*2 transforms it to the double of it negation.
    // Note that this keeps the values as a double of a value in the range [0, P].
    const P2: u32 = P * 2;
    const NEGATION_MASK: u32x16 =
        u32x16::from_array([0, P2, P2, 0, 0, P2, P2, 0, 0, P2, P2, 0, 0, P2, P2, 0]);
    let t0 = simd_swizzle!(
        t1,
        [
            0b0001, 0b0001, 0b0000, 0b0000, 0b0011, 0b0011, 0b0010, 0b0010, 0b0101, 0b0101, 0b0100,
            0b0100, 0b0111, 0b0111, 0b0110, 0b0110,
        ]
    ) ^ NEGATION_MASK;
    (t0, t1)
}

#[inline]
const unsafe fn load(mem_addr: *const u32) -> u32x16 {
    std::ptr::read(mem_addr as *const u32x16)
}

#[inline]
const unsafe fn store(mem_addr: *mut u32, a: u32x16) {
    std::ptr::write(mem_addr as *mut u32x16, a);
}

/// Computes `v * twiddle`
fn mul_twiddle(v: PackedBaseField, twiddle_dbl: u32x16) -> PackedBaseField {
    // TODO: Come up with a better approach than `cfg`ing on target_feature.
    // TODO: Ensure all these branches get tested in the CI.
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
            // TODO: For architectures that when multiplying require doubling then the twiddles
            // should be precomputed as double. For other architectures, the twiddle should be
            // precomputed without doubling.
            crate::prover::backend::simd::m31::mul_doubled_neon(v, twiddle_dbl)
        } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
            crate::prover::backend::simd::m31::mul_doubled_wasm(v, twiddle_dbl)
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
            crate::prover::backend::simd::m31::mul_doubled_avx512(v, twiddle_dbl)
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            crate::prover::backend::simd::m31::mul_doubled_avx2(v, twiddle_dbl)
        } else {
            crate::prover::backend::simd::m31::mul_doubled_simd(v, twiddle_dbl)
        }
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_transpose_vecs_sequential_vs_parallel() {
        let mut rng = SmallRng::seed_from_u64(0);

        // Test various sizes
        let log_n_vecs = 7;
        let n_vecs = 1 << log_n_vecs;
        let n_u32s = n_vecs * 16; // Each SIMD vector contains 16 u32s

        // Generate random data
        let data_original: Vec<u32> = (0..n_u32s).map(|_| rng.gen()).collect();
        let data_parallel = data_original.clone();
        let data_sequential = data_original.clone();

        // Ensure proper alignment by using aligned allocation
        let mut aligned_parallel = vec![0u32; n_u32s];
        let mut aligned_sequential = vec![0u32; n_u32s];

        let mut buffer0 = vec![0u32; n_u32s];
        let mut buffer1 = vec![0u32; n_u32s];

        aligned_parallel.copy_from_slice(&data_parallel);
        aligned_sequential.copy_from_slice(&data_sequential);

        // Apply both transpose functions
        unsafe {
            transpose_vecs(aligned_parallel.as_mut_ptr(), log_n_vecs);
            transpose_vecs2(
                aligned_sequential.as_mut_ptr(),
                log_n_vecs,
                5,
                buffer0.as_mut_ptr(),
                buffer1.as_mut_ptr(),
            );
        }

        // Compare results
        assert_eq!(
            aligned_parallel, aligned_sequential,
            "Mismatch for log_n_vecs={}\n \
                orignal vec: {:?}",
            log_n_vecs, data_parallel
        );
    }

    #[test]
    fn test_transpose_vecs_identity() {
        let mut rng = SmallRng::seed_from_u64(42);

        // Test that applying transpose twice gives back the original
        for log_n_vecs in 3..=6 {
            let n_vecs = 1 << log_n_vecs;
            let n_u32s = n_vecs * 16;

            let original_data: Vec<u32> = (0..n_u32s).map(|_| rng.gen()).collect();
            let mut data = original_data.clone();

            let mut buffer0 = vec![0u32; n_u32s];
            let mut buffer1 = vec![0u32; n_u32s];

            // Apply transpose twice
            unsafe {
                transpose_vecs2(
                    data.as_mut_ptr(),
                    log_n_vecs,
                    1,
                    buffer0.as_mut_ptr(),
                    buffer1.as_mut_ptr(),
                );
                transpose_vecs2(
                    data.as_mut_ptr(),
                    log_n_vecs,
                    1,
                    buffer0.as_mut_ptr(),
                    buffer1.as_mut_ptr(),
                );
            }

            // Should be back to original
            assert_eq!(
                data, original_data,
                "Double transpose didn't restore original for log_n_vecs={}",
                log_n_vecs
            );
        }
    }

    // #[test]
    // fn test_transpose_vecs_small_case() {
    //     // Test a small known case to verify the bit swapping logic
    //     let log_n_vecs = 2; // 4 vectors
    //     let n_u32s = 4 * 16; // 64 u32s

    //     // Create test data where each SIMD vector has a recognizable pattern
    //     let mut data = vec![0u32; n_u32s];
    //     for i in 0..4 {
    //         for j in 0..16 {
    //             data[i * 16 + j] = (i as u32) << 16 | (j as u32);
    //         }
    //     }

    //     let original_data = data.clone();

    //     // Apply transpose
    //     unsafe {
    //         transpose_vecs2(
    //             data.as_mut_ptr(),
    //             log_n_vecs,
    //             1,
    //             buffer0.as_mut_ptr(),
    //             buffer1.as_mut_ptr(),
    //         );
    //         transpose_vecs2(
    //             data.as_mut_ptr(),
    //             log_n_vecs,
    //             1,
    //             buffer0.as_mut_ptr(),
    //             buffer1.as_mut_ptr(),
    //         );
    //     }

    //     // Verify the data changed (it should transpose)
    //     assert_ne!(data, original_data, "Transpose should change the data");

    //     // Apply transpose again to get back to original
    //     unsafe {
    //         transpose_vecs2(
    //             data.as_mut_ptr(),
    //             log_n_vecs,
    //             1,
    //             buffer0.as_mut_ptr(),
    //             buffer1.as_mut_ptr(),
    //         );
    //     }

    //     // Should be back to original
    //     assert_eq!(
    //         data, original_data,
    //         "Double transpose should restore original"
    //     );
    // }
}
