// #![allow(unused_variables)]
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
pub unsafe fn full_copy_block_transpose(
    values: &mut [u32x16],
    log_tile_edge: u32, // In vecs.
    buffer0: &mut [u32x16],
    buffer1: &mut [u32x16],
) {
    use std::ptr::{read, write};

    let n_vecs = values.len();
    assert!(n_vecs.is_power_of_two());
    let log_n_vecs = n_vecs.ilog2();
    let half = log_n_vecs / 2;
    let log_tile_edge = std::cmp::min(half, log_tile_edge);
    assert!(buffer0.len() >= 1 << (log_tile_edge * 2));
    assert!(buffer1.len() >= 1 << (log_tile_edge * 2));
    assert!(buffer0.len() == buffer1.len());
    let tile_edge = (1 << log_tile_edge) as u32;
    let n_workers = buffer0.len() / (tile_edge * tile_edge) as usize;
    let log_row_length = log_n_vecs.div_ceil(2);
    let log_height = half - log_tile_edge;

    // Precompute all tile pairs (r,c) with r <= c
    let mut tile_pairs = vec![];
    for r in 0..1 << log_height {
        for c in r..1 << log_height {
            tile_pairs.push((r, c));
        }
    }
    // assert!(tile_pairs.len() % n_workers == 0);
    let n_blocks_per_worker = tile_pairs.len() / n_workers;

    let values = UnsafeMut(values.as_mut_ptr());

    for b in 0..=(log_n_vecs & 1) {
        // Parallel over tile-pairs
        tile_pairs
            .par_chunks(n_blocks_per_worker)
            .zip(buffer0.par_chunks_mut((tile_edge * tile_edge) as usize))
            .zip(buffer1.par_chunks_mut((tile_edge * tile_edge) as usize))
            .for_each(|((chunk, buffer0), buffer1)| {
                let values = values.get();
                chunk.iter().for_each(|(r, c)| {
                    let row_off = r * tile_edge;
                    let col_off = c * tile_edge;

                    if r == c {
                        // In-place within diagonal tile T_{r,r}
                        for i in 0..tile_edge {
                            for j in (i + 1)..tile_edge {
                                let idx_i =
                                    ((row_off + i) << log_row_length) + j + col_off + (b << half);
                                let idx_j = perm_index(idx_i, log_n_vecs, half);

                                let v0 = read(values.add(idx_i as usize));
                                let v1 = read(values.add(idx_j as usize));
                                write(values.add(idx_i as usize), v1);
                                write(values.add(idx_j as usize), v0);
                            }
                        }
                    } else {
                        // 1. Copy T_{r,c} to buffer0
                        for i in 0..tile_edge {
                            for j in 0..tile_edge {
                                let idx_i =
                                    ((row_off + i) << log_row_length) + j + col_off + (b << half);
                                buffer0[(j * tile_edge + i) as usize] =
                                    read(values.add(idx_i as usize));
                            }
                        }

                        // 2. Copy T_{c,r} to buffer1
                        for i in 0..tile_edge {
                            for j in 0..tile_edge {
                                let idx_i =
                                    ((col_off + i) << log_row_length) + j + row_off + (b << half);
                                buffer1[(j * tile_edge + i) as usize] =
                                    read(values.add(idx_i as usize));
                            }
                        }

                        // 3. Copy buffer0 to T_{c,r}
                        for i in 0..tile_edge {
                            for j in 0..tile_edge {
                                let idx_i =
                                    ((col_off + i) << log_row_length) + j + row_off + (b << half);
                                write(
                                    values.add(idx_i as usize),
                                    buffer0[(i * tile_edge + j) as usize],
                                );
                            }
                        }

                        // 4. Copy buffer1 to T_{r,c}
                        for i in 0..tile_edge {
                            for j in 0..tile_edge {
                                let idx_i =
                                    ((row_off + i) << log_row_length) + j + col_off + (b << half);
                                write(
                                    values.add(idx_i as usize),
                                    buffer1[(i * tile_edge + j) as usize],
                                );
                            }
                        }
                    }
                });
            });
    }
}

/// Compute the permuted index swapping bits abc <-> cba.
const fn perm_index(x: u32, log_n: u32, half: u32) -> u32 {
    let a = x >> (log_n - half);
    let b = (x >> half) & (log_n & 1);
    let c = x & ((1 << half) - 1);
    (c << (log_n - half)) | (b << half) | a
}

/// Transpose `src` (rows×cols) into `dst` (cols×rows), both in row-major order.
///
/// # Panics
/// - if `src.len() != rows*cols`
/// - if `dst.len() != rows*cols`
pub fn cache_oblivious_transpose<T: Copy>(src: &[T], dst: &mut [T], rows: usize, cols: usize) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dst.len(), rows * cols);
    transpose_rec(src, dst, 0, 0, rows, cols, cols, rows);
}

/// Recursive helper splitting the larger of height/width.
/// - `r0`,`c0`: top-left corner in src
/// - `h`,`w`: tile height/width
/// - `src_stride`: number of elements per src row
/// - `dst_stride`: number of elements per dst row
#[allow(clippy::too_many_arguments)]
fn transpose_rec<T: Copy>(
    src: &[T],
    dst: &mut [T],
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    if h == 1 && w == 1 {
        // copy single element
        dst[c0 * dst_stride + r0] = src[r0 * src_stride + c0];
    } else if h >= w {
        // split height in half
        let h2 = h / 2;
        transpose_rec(src, dst, r0, c0, h2, w, src_stride, dst_stride);
        transpose_rec(src, dst, r0 + h2, c0, h - h2, w, src_stride, dst_stride);
    } else {
        // split width in half
        let w2 = w / 2;
        transpose_rec(src, dst, r0, c0, h, w2, src_stride, dst_stride);
        transpose_rec(src, dst, r0, c0 + w2, h, w - w2, src_stride, dst_stride);
    }
}

/// Transpose `src` (rows×cols) into `dst` (cols×rows), both in row-major order.
/// Parallel, cache-oblivious, out-of-place version.
///
/// # Panics
/// - if `src.len() != rows*cols`
/// - if `dst.len() != rows*cols`
#[cfg(feature = "parallel")]
pub fn cache_oblivious_transpose_par<T: Copy + Sync + Send>(
    src: &[T],
    dst: &mut [T],
    rows: usize,
    cols: usize,
) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dst.len(), rows * cols);
    let dst = UnsafeMut(dst.as_mut_ptr());
    transpose_rec_par(src, dst, 0, 0, rows, cols, cols, rows);
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn transpose_rec_par<T: Copy + Sync + Send>(
    src: &[T],
    dst: UnsafeMut<T>, // unsafe pointer to dst
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    if h == 1 && w == 1 {
        // base case: copy single element
        unsafe {
            std::ptr::write(dst.get().add(c0 * dst_stride + r0), src[r0 * src_stride + c0]);
        }
    } else if h >= w {
        // split height in half, recurse in parallel
        let h2 = h / 2;
        rayon::join(
            || transpose_rec_par(src, dst, r0, c0, h2, w, src_stride, dst_stride),
            || transpose_rec_par(src, dst, r0 + h2, c0, h - h2, w, src_stride, dst_stride),
        );
    } else {
        // split width in half, recurse in parallel
        let w2 = w / 2;
        rayon::join(
            || transpose_rec_par(src, dst, r0, c0, h, w2, src_stride, dst_stride),
            || transpose_rec_par(src, dst, r0, c0 + w2, h, w - w2, src_stride, dst_stride),
        );
    }
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
        let log_n_vecs = 20;
        let n_vecs = 1 << log_n_vecs;
        let n_u32s = n_vecs * 16; // Each SIMD vector contains 16 u32s
        let log_tile_edge = 5;

        // Generate random data
        let data_original: Vec<u32> = (0..n_u32s).map(|_| rng.gen()).collect();
        let data_parallel = data_original.clone();
        let data_sequential = data_original.clone();

        // Ensure proper alignment by using aligned allocation
        let mut aligned_parallel = vec![0u32; n_u32s];
        let mut aligned_sequential = vec![u32x16::splat(0); n_u32s / 16];

        let mut buffer0 = vec![u32x16::splat(0); 6 << (log_tile_edge * 2)];
        let mut buffer1 = vec![u32x16::splat(0); 6 << (log_tile_edge * 2)];

        aligned_parallel.copy_from_slice(&data_parallel);
        aligned_sequential.copy_from_slice(
            &data_sequential
                .chunks(16)
                .map(|chunk| u32x16::from_slice(chunk))
                .collect::<Vec<_>>(),
        );

        // Apply both transpose functions
        unsafe {
            transpose_vecs(aligned_parallel.as_mut_ptr(), log_n_vecs);
            full_copy_block_transpose(
                &mut aligned_sequential,
                log_tile_edge,
                &mut buffer0,
                &mut buffer1,
            );
        }

        let aligned_parallel = aligned_parallel
            .chunks(16)
            .map(|chunk| u32x16::from_slice(chunk))
            .collect::<Vec<_>>();
        // Compare results
        assert_eq!(
            aligned_parallel, aligned_sequential,
            "Mismatch for log_n_vecs={}\n \
                orignal vec: {:?}",
            log_n_vecs, data_parallel
        );
    }

    #[test]
    fn test_cache_oblivious_transpose() {
        // example 5×3 matrix
        let rows = 1 << 12;
        let cols = 1 << 12;
        let src: Vec<u32x16> = (0..(rows * cols) as u32).map(u32x16::splat).collect();
        let mut expected = src.clone();
        unsafe {
            transpose_vecs(expected.as_mut_ptr() as *mut u32, 24);
        }

        let mut dst = vec![u32x16::splat(0); rows * cols];
        cache_oblivious_transpose(&src, &mut dst, rows, cols);

        assert_eq!(dst, expected);
    }


    #[test]
    fn test_cache_oblivious_transpose_par() {
        // example 5×3 matrix
        let rows = 1 << 12;
        let cols = 1 << 12;
        let src: Vec<u32x16> = (0..(rows * cols) as u32).map(u32x16::splat).collect();
        let mut expected = src.clone();
        unsafe {
            transpose_vecs(expected.as_mut_ptr() as *mut u32, 24);
        }

        let mut dst = vec![u32x16::splat(0); rows * cols];
        cache_oblivious_transpose_par(&src, &mut dst, rows, cols);

        assert_eq!(dst, expected);
    }
}
