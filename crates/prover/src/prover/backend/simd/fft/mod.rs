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
            crate::core::backend::simd::m31::mul_doubled_neon(v, twiddle_dbl)
        } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
            crate::core::backend::simd::m31::mul_doubled_wasm(v, twiddle_dbl)
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
            crate::core::backend::simd::m31::mul_doubled_avx512(v, twiddle_dbl)
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            crate::core::backend::simd::m31::mul_doubled_avx2(v, twiddle_dbl)
        } else {
            crate::core::backend::simd::m31::mul_doubled_simd(v, twiddle_dbl)
        }
    }
}
