//! Inverse fft.

use std::simd::{simd_swizzle, u32x16, u32x2, u32x4};

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    compute_first_twiddles, mul_twiddle, transpose_vecs, CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE,
};
use crate::core::backend::cpu::bit_reverse;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::utils::UnsafeMut;
use crate::core::circle::Coset;
use crate::parallel_iter;

/// Performs an Inverse Circle Fast Fourier Transform (ICFFT) on the given values.
///
/// # Arguments
///
/// - `values`: A mutable pointer to the values on which the ICFFT is to be performed.
/// - `twiddle_dbl`: A reference to the doubles of the twiddle factors.
/// - `log_n_elements`: The log of the number of elements in the `values` array.
///
/// # Panics
///
/// Panic if `log_n_elements` is less than [`MIN_FFT_LOG_SIZE`].
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`PackedBaseField`].
pub unsafe fn ifft(values: *mut u32, twiddle_dbl: &[&[u32]], log_n_elements: usize) {
    assert!(log_n_elements >= MIN_FFT_LOG_SIZE as usize);

    let log_n_vecs = log_n_elements - LOG_N_LANES as usize;
    if log_n_elements <= CACHED_FFT_LOG_SIZE as usize {
        ifft_lower_with_vecwise(values, twiddle_dbl, log_n_elements, log_n_elements);
        return;
    }

    let fft_layers_pre_transpose = log_n_vecs.div_ceil(2);
    let fft_layers_post_transpose = log_n_vecs / 2;
    ifft_lower_with_vecwise(
        values,
        &twiddle_dbl[..3 + fft_layers_pre_transpose],
        log_n_elements,
        fft_layers_pre_transpose + LOG_N_LANES as usize,
    );
    transpose_vecs(values, log_n_vecs);
    ifft_lower_without_vecwise(
        values,
        &twiddle_dbl[3 + fft_layers_pre_transpose..],
        log_n_elements,
        fft_layers_post_transpose,
    );
}

/// Computes partial ifft on `2^log_size` M31 elements.
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array, aligned to 64 bytes.
/// - `twiddle_dbl`: The doubles of the twiddle factors for each layer of the the ifft. Layer i
///   holds `2^(log_size - 1 - i)` twiddles.
/// - `log_size`: The log of the number of number of M31 elements in the array.
/// - `fft_layers`: The number of ifft layers to apply, out of log_size.
///
/// # Panics
///
/// Panics if `log_size` is not at least 5.
///
/// # Safety
///
/// `values` must have the same alignment as [`PackedBaseField`].
/// `fft_layers` must be at least 5.
pub unsafe fn ifft_lower_with_vecwise(
    values: *mut u32,
    twiddle_dbl: &[&[u32]],
    log_size: usize,
    fft_layers: usize,
) {
    const VECWISE_FFT_BITS: usize = LOG_N_LANES as usize + 1;
    assert!(log_size >= VECWISE_FFT_BITS);

    assert_eq!(twiddle_dbl[0].len(), 1 << (log_size - 2));

    let values = UnsafeMut(values);
    parallel_iter!(0..1 << (log_size - fft_layers)).for_each(|index_h| {
        let values = values.get();
        ifft_vecwise_loop(values, twiddle_dbl, fft_layers - VECWISE_FFT_BITS, index_h);
        for layer in (VECWISE_FFT_BITS..fft_layers).step_by(3) {
            match fft_layers - layer {
                1 => {
                    ifft1_loop(values, &twiddle_dbl[(layer - 1)..], layer, index_h);
                }
                2 => {
                    ifft2_loop(values, &twiddle_dbl[(layer - 1)..], layer, index_h);
                }
                _ => {
                    ifft3_loop(
                        values,
                        &twiddle_dbl[(layer - 1)..],
                        fft_layers - layer - 3,
                        layer,
                        index_h,
                    );
                }
            }
        }
    });
}

/// Computes partial ifft on `2^log_size` M31 elements, skipping the vecwise layers (lower 4 bits of
/// the index).
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array, aligned to 64 bytes.
/// - `twiddle_dbl`: The doubles of the twiddle factors for each layer of the the ifft.
/// - `log_size`: The log of the number of number of M31 elements in the array.
/// - `fft_layers`: The number of ifft layers to apply, out of `log_size - LOG_N_LANES`.
///
/// # Panics
///
/// Panics if `log_size` is not at least 4.
///
/// # Safety
///
/// `values` must have the same alignment as [`PackedBaseField`].
/// `fft_layers` must be at least 4.
pub unsafe fn ifft_lower_without_vecwise(
    values: *mut u32,
    twiddle_dbl: &[&[u32]],
    log_size: usize,
    fft_layers: usize,
) {
    assert!(log_size >= LOG_N_LANES as usize);

    let values = UnsafeMut(values);
    parallel_iter!(0..1 << (log_size - fft_layers - LOG_N_LANES as usize)).for_each(|index_h| {
        let values = values.get();
        for layer in (0..fft_layers).step_by(3) {
            let fixed_layer = layer + LOG_N_LANES as usize;
            match fft_layers - layer {
                1 => {
                    ifft1_loop(values, &twiddle_dbl[layer..], fixed_layer, index_h);
                }
                2 => {
                    ifft2_loop(values, &twiddle_dbl[layer..], fixed_layer, index_h);
                }
                _ => {
                    ifft3_loop(
                        values,
                        &twiddle_dbl[layer..],
                        fft_layers - layer - 3,
                        fixed_layer,
                        index_h,
                    );
                }
            }
        }
    });
}

/// Runs the first 5 ifft layers across the entire array.
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array, aligned to 64 bytes.
/// - `twiddle_dbl`: The doubles of the twiddle factors for each of the 5 ifft layers.
/// - `high_bits`: The number of bits this loops needs to run on.
/// - `index_h`: The higher part of the index, iterated by the caller.
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`PackedBaseField`].
pub unsafe fn ifft_vecwise_loop(
    values: *mut u32,
    twiddle_dbl: &[&[u32]],
    loop_bits: usize,
    index_h: usize,
) {
    for index_l in 0..1 << loop_bits {
        let index = (index_h << loop_bits) + index_l;
        let mut val0 = PackedBaseField::load(values.add(index * 32).cast_const());
        let mut val1 = PackedBaseField::load(values.add(index * 32 + 16).cast_const());
        (val0, val1) = vecwise_ibutterflies(
            val0,
            val1,
            std::array::from_fn(|i| *twiddle_dbl[0].get_unchecked(index * 8 + i)),
            std::array::from_fn(|i| *twiddle_dbl[1].get_unchecked(index * 4 + i)),
            std::array::from_fn(|i| *twiddle_dbl[2].get_unchecked(index * 2 + i)),
        );
        (val0, val1) = simd_ibutterfly(
            val0,
            val1,
            u32x16::splat(*twiddle_dbl[3].get_unchecked(index)),
        );
        val0.store(values.add(index * 32));
        val1.store(values.add(index * 32 + 16));
    }
}

/// Runs 3 ifft layers across the entire array.
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array, aligned to 64 bytes.
/// - `twiddle_dbl`: The doubles of the twiddle factors for each of the 3 ifft layers.
/// - `loop_bits`: The number of bits this loops needs to run on.
/// - `layer`: The layer number of the first ifft layer to apply. The layers `layer`, `layer + 1`,
///   `layer + 2` are applied.
/// - `index_h`: The higher part of the index, iterated by the caller.
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`PackedBaseField`].
pub unsafe fn ifft3_loop(
    values: *mut u32,
    twiddle_dbl: &[&[u32]],
    loop_bits: usize,
    layer: usize,
    index_h: usize,
) {
    for index_l in 0..1 << loop_bits {
        let index = (index_h << loop_bits) + index_l;
        let offset = index << (layer + 3);
        for l in (0..1 << layer).step_by(1 << LOG_N_LANES as usize) {
            ifft3(
                values,
                offset + l,
                layer,
                std::array::from_fn(|i| {
                    *twiddle_dbl[0].get_unchecked((index * 4 + i) & (twiddle_dbl[0].len() - 1))
                }),
                std::array::from_fn(|i| {
                    *twiddle_dbl[1].get_unchecked((index * 2 + i) & (twiddle_dbl[1].len() - 1))
                }),
                std::array::from_fn(|i| {
                    *twiddle_dbl[2].get_unchecked((index + i) & (twiddle_dbl[2].len() - 1))
                }),
            );
        }
    }
}

/// Runs 2 ifft layers across the entire array.
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array, aligned to 64 bytes.
/// - `twiddle_dbl`: The doubles of the twiddle factors for each of the 2 ifft layers.
/// - `loop_bits`: The number of bits this loops needs to run on.
/// - `layer`: The layer number of the first ifft layer to apply. The layers `layer`, `layer + 1`
///   are applied.
/// - `index`: The index, iterated by the caller.
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`PackedBaseField`].
unsafe fn ifft2_loop(values: *mut u32, twiddle_dbl: &[&[u32]], layer: usize, index: usize) {
    let offset = index << (layer + 2);
    for l in (0..1 << layer).step_by(1 << LOG_N_LANES as usize) {
        ifft2(
            values,
            offset + l,
            layer,
            std::array::from_fn(|i| {
                *twiddle_dbl[0].get_unchecked((index * 2 + i) & (twiddle_dbl[0].len() - 1))
            }),
            std::array::from_fn(|i| {
                *twiddle_dbl[1].get_unchecked((index + i) & (twiddle_dbl[1].len() - 1))
            }),
        );
    }
}

/// Runs 1 ifft layer across the entire array.
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array, aligned to 64 bytes.
/// - `twiddle_dbl`: The doubles of the twiddle factors for the ifft layer.
/// - `layer`: The layer number of the ifft layer to apply.
/// - `index_h`: The higher part of the index, iterated by the caller.
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`PackedBaseField`].
unsafe fn ifft1_loop(values: *mut u32, twiddle_dbl: &[&[u32]], layer: usize, index: usize) {
    let offset = index << (layer + 1);
    for l in (0..1 << layer).step_by(1 << LOG_N_LANES as usize) {
        ifft1(
            values,
            offset + l,
            layer,
            std::array::from_fn(|i| {
                *twiddle_dbl[0].get_unchecked((index + i) & (twiddle_dbl[0].len() - 1))
            }),
        );
    }
}

/// Computes the ibutterfly operation for packed M31 elements.
///
/// Returns `val0 + val1, t (val0 - val1)`. `val0, val1` are packed M31 elements. 16 M31 words at
/// each. Each value is assumed to be in unreduced form, [0, P] including P. `twiddle_dbl` holds 16
/// values, each is a *double* of a twiddle factor, in unreduced form.
pub fn simd_ibutterfly(
    val0: PackedBaseField,
    val1: PackedBaseField,
    twiddle_dbl: u32x16,
) -> (PackedBaseField, PackedBaseField) {
    let r0 = val0 + val1;
    let r1 = val0 - val1;
    let prod = mul_twiddle(r1, twiddle_dbl);
    (r0, prod)
}

/// Runs ifft on 2 vectors of 16 M31 elements.
///
/// This amounts to 4 butterfly layers, each with 16 butterflies.
/// Each of the vectors represents a bit reversed evaluation.
/// Each value in a vectors is in unreduced form: [0, P] including P.
/// Takes 3 twiddle arrays, one for each layer after the first, holding the double of the
/// corresponding twiddle.
/// The first layer's twiddles (lower bit of the index) are computed from the second layer's
/// twiddles. The second layer takes 8 twiddles.
/// The third layer takes 4 twiddles.
/// The fourth layer takes 2 twiddles.
pub fn vecwise_ibutterflies(
    mut val0: PackedBaseField,
    mut val1: PackedBaseField,
    twiddle1_dbl: [u32; 8],
    twiddle2_dbl: [u32; 4],
    twiddle3_dbl: [u32; 2],
) -> (PackedBaseField, PackedBaseField) {
    // TODO(andrew): Can the permute be fused with the _mm512_srli_epi64 inside the butterfly?

    // Each `ibutterfly` take 2 512-bit registers, and does 16 butterflies element by element.
    // We need to permute the 512-bit registers to get the right order for the butterflies.
    // Denote the index of the 16 M31 elements in register i as i:abcd.
    // At each layer we apply the following permutation to the index:
    //   i:abcd => d:iabc
    // This is how it looks like at each iteration.
    //   i:abcd
    //   d:iabc
    //    ifft on d
    //   c:diab
    //    ifft on c
    //   b:cdia
    //    ifft on b
    //   a:bcid
    //    ifft on a
    //   i:abcd

    let (t0, t1) = compute_first_twiddles(twiddle1_dbl.into());

    // Apply the permutation, resulting in indexing d:iabc.
    (val0, val1) = val0.deinterleave(val1);
    (val0, val1) = simd_ibutterfly(val0, val1, t0);

    // Apply the permutation, resulting in indexing c:diab.
    (val0, val1) = val0.deinterleave(val1);
    (val0, val1) = simd_ibutterfly(val0, val1, t1);

    let t = simd_swizzle!(
        u32x4::from(twiddle2_dbl),
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    );
    // Apply the permutation, resulting in indexing b:cdia.
    (val0, val1) = val0.deinterleave(val1);
    (val0, val1) = simd_ibutterfly(val0, val1, t);

    let t = simd_swizzle!(
        u32x2::from(twiddle3_dbl),
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    );
    // Apply the permutation, resulting in indexing a:bcid.
    (val0, val1) = val0.deinterleave(val1);
    (val0, val1) = simd_ibutterfly(val0, val1, t);

    // Apply the permutation, resulting in indexing i:abcd.
    val0.deinterleave(val1)
}

/// Returns the line twiddles (x points) for an ifft on a coset.
pub fn get_itwiddle_dbls(mut coset: Coset) -> Vec<Vec<u32>> {
    let mut res = vec![];
    for _ in 0..coset.log_size() {
        res.push(
            coset
                .iter()
                .take(coset.size() / 2)
                .map(|p| p.x.inverse().0 * 2)
                .collect_vec(),
        );
        bit_reverse(res.last_mut().unwrap());
        coset = coset.double();
    }

    res
}

/// Applies 3 ibutterfly layers on 8 vectors of 16 M31 elements.
///
/// Vectorized over the 16 elements of the vectors.
/// Used for radix-8 ifft.
/// Each butterfly layer, has 3 SIMD butterflies.
/// Total of 12 SIMD butterflies.
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array.
/// - `offset`: The offset of the first value in the array.
/// - `log_step`: The log of the distance in the array, in M31 elements, between each pair of values
///   that need to be transformed. For layer i this is i - 4.
/// - `twiddles_dbl0/1/2`: The double of the twiddles for the 3 layers of ibutterflies. Each layer
///   has 4/2/1 twiddles.
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`PackedBaseField`].
pub unsafe fn ifft3(
    values: *mut u32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [u32; 4],
    twiddles_dbl1: [u32; 2],
    twiddles_dbl2: [u32; 1],
) {
    // Load the 8 SIMD vectors from the array.
    let mut val0 = PackedBaseField::load(values.add(offset + (0 << log_step)).cast_const());
    let mut val1 = PackedBaseField::load(values.add(offset + (1 << log_step)).cast_const());
    let mut val2 = PackedBaseField::load(values.add(offset + (2 << log_step)).cast_const());
    let mut val3 = PackedBaseField::load(values.add(offset + (3 << log_step)).cast_const());
    let mut val4 = PackedBaseField::load(values.add(offset + (4 << log_step)).cast_const());
    let mut val5 = PackedBaseField::load(values.add(offset + (5 << log_step)).cast_const());
    let mut val6 = PackedBaseField::load(values.add(offset + (6 << log_step)).cast_const());
    let mut val7 = PackedBaseField::load(values.add(offset + (7 << log_step)).cast_const());

    // Apply the first layer of ibutterflies.
    (val0, val1) = simd_ibutterfly(val0, val1, u32x16::splat(twiddles_dbl0[0]));
    (val2, val3) = simd_ibutterfly(val2, val3, u32x16::splat(twiddles_dbl0[1]));
    (val4, val5) = simd_ibutterfly(val4, val5, u32x16::splat(twiddles_dbl0[2]));
    (val6, val7) = simd_ibutterfly(val6, val7, u32x16::splat(twiddles_dbl0[3]));

    // Apply the second layer of ibutterflies.
    (val0, val2) = simd_ibutterfly(val0, val2, u32x16::splat(twiddles_dbl1[0]));
    (val1, val3) = simd_ibutterfly(val1, val3, u32x16::splat(twiddles_dbl1[0]));
    (val4, val6) = simd_ibutterfly(val4, val6, u32x16::splat(twiddles_dbl1[1]));
    (val5, val7) = simd_ibutterfly(val5, val7, u32x16::splat(twiddles_dbl1[1]));

    // Apply the third layer of ibutterflies.
    (val0, val4) = simd_ibutterfly(val0, val4, u32x16::splat(twiddles_dbl2[0]));
    (val1, val5) = simd_ibutterfly(val1, val5, u32x16::splat(twiddles_dbl2[0]));
    (val2, val6) = simd_ibutterfly(val2, val6, u32x16::splat(twiddles_dbl2[0]));
    (val3, val7) = simd_ibutterfly(val3, val7, u32x16::splat(twiddles_dbl2[0]));

    // Store the 8 SIMD vectors back to the array.
    val0.store(values.add(offset + (0 << log_step)));
    val1.store(values.add(offset + (1 << log_step)));
    val2.store(values.add(offset + (2 << log_step)));
    val3.store(values.add(offset + (3 << log_step)));
    val4.store(values.add(offset + (4 << log_step)));
    val5.store(values.add(offset + (5 << log_step)));
    val6.store(values.add(offset + (6 << log_step)));
    val7.store(values.add(offset + (7 << log_step)));
}

/// Applies 2 ibutterfly layers on 4 vectors of 16 M31 elements.
///
/// Vectorized over the 16 elements of the vectors.
/// Used for radix-4 ifft.
/// Each ibutterfly layer, has 2 SIMD butterflies.
/// Total of 4 SIMD butterflies.
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array.
/// - `offset`: The offset of the first value in the array.
/// - `log_step`: The log of the distance in the array, in M31 elements, between each pair of values
///   that need to be transformed. For layer `i` this is `i - 4`.
/// - `twiddles_dbl0/1`: The double of the twiddles for the 2 layers of ibutterflies. Each layer has
///   2/1 twiddles.
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`PackedBaseField`].
pub unsafe fn ifft2(
    values: *mut u32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [u32; 2],
    twiddles_dbl1: [u32; 1],
) {
    // Load the 4 SIMD vectors from the array.
    let mut val0 = PackedBaseField::load(values.add(offset + (0 << log_step)).cast_const());
    let mut val1 = PackedBaseField::load(values.add(offset + (1 << log_step)).cast_const());
    let mut val2 = PackedBaseField::load(values.add(offset + (2 << log_step)).cast_const());
    let mut val3 = PackedBaseField::load(values.add(offset + (3 << log_step)).cast_const());

    // Apply the first layer of butterflies.
    (val0, val1) = simd_ibutterfly(val0, val1, u32x16::splat(twiddles_dbl0[0]));
    (val2, val3) = simd_ibutterfly(val2, val3, u32x16::splat(twiddles_dbl0[1]));

    // Apply the second layer of butterflies.
    (val0, val2) = simd_ibutterfly(val0, val2, u32x16::splat(twiddles_dbl1[0]));
    (val1, val3) = simd_ibutterfly(val1, val3, u32x16::splat(twiddles_dbl1[0]));

    // Store the 4 SIMD vectors back to the array.
    val0.store(values.add(offset + (0 << log_step)));
    val1.store(values.add(offset + (1 << log_step)));
    val2.store(values.add(offset + (2 << log_step)));
    val3.store(values.add(offset + (3 << log_step)));
}

/// Applies 1 ibutterfly layers on 2 vectors of 16 M31 elements.
///
/// Vectorized over the 16 elements of the vectors.
///
/// # Arguments
///
/// - `values`: Pointer to the entire value array.
/// - `offset`: The offset of the first value in the array.
/// - `log_step`: The log of the distance in the array, in M31 elements, between each pair of values
///   that need to be transformed. For layer `i` this is `i - 4`.
/// - `twiddles_dbl0`: The double of the twiddles for the ibutterfly layer.
///
/// # Safety
///
/// Behavior is undefined if `values` does not have the same alignment as [`PackedBaseField`].
pub unsafe fn ifft1(values: *mut u32, offset: usize, log_step: usize, twiddles_dbl0: [u32; 1]) {
    // Load the 2 SIMD vectors from the array.
    let mut val0 = PackedBaseField::load(values.add(offset + (0 << log_step)).cast_const());
    let mut val1 = PackedBaseField::load(values.add(offset + (1 << log_step)).cast_const());

    (val0, val1) = simd_ibutterfly(val0, val1, u32x16::splat(twiddles_dbl0[0]));

    // Store the 2 SIMD vectors back to the array.
    val0.store(values.add(offset + (0 << log_step)));
    val1.store(values.add(offset + (1 << log_step)));
}

#[cfg(test)]
mod tests {
    use std::mem::transmute;
    use std::simd::u32x16;

    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{
        get_itwiddle_dbls, ifft, ifft3, ifft_lower_with_vecwise, simd_ibutterfly,
        vecwise_ibutterflies,
    };
    use crate::core::backend::cpu::CpuCircleEvaluation;
    use crate::core::backend::simd::column::BaseColumn;
    use crate::core::backend::simd::fft::{transpose_vecs, CACHED_FFT_LOG_SIZE};
    use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
    use crate::core::backend::Column;
    use crate::core::fft::ibutterfly as ground_truth_ibutterfly;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain};

    #[test]
    fn test_ibutterfly() {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut v0: [BaseField; N_LANES] = rng.gen();
        let mut v1: [BaseField; N_LANES] = rng.gen();
        let twiddle: [BaseField; N_LANES] = rng.gen();
        let twiddle_dbl = twiddle.map(|v| v.0 * 2);

        let (r0, r1) = simd_ibutterfly(v0.into(), v1.into(), twiddle_dbl.into());

        let r0 = r0.to_array();
        let r1 = r1.to_array();
        for i in 0..N_LANES {
            ground_truth_ibutterfly(&mut v0[i], &mut v1[i], twiddle[i]);
            assert_eq!((v0[i], v1[i]), (r0[i], r1[i]), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_ifft3() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values = rng.gen::<[BaseField; 8]>().map(PackedBaseField::broadcast);
        let twiddles0: [BaseField; 4] = rng.gen();
        let twiddles1: [BaseField; 2] = rng.gen();
        let twiddles2: [BaseField; 1] = rng.gen();
        let twiddles0_dbl = twiddles0.map(|v| v.0 * 2);
        let twiddles1_dbl = twiddles1.map(|v| v.0 * 2);
        let twiddles2_dbl = twiddles2.map(|v| v.0 * 2);

        let mut res = values;
        unsafe {
            ifft3(
                transmute::<*mut PackedBaseField, *mut u32>(res.as_mut_ptr()),
                0,
                LOG_N_LANES as usize,
                twiddles0_dbl,
                twiddles1_dbl,
                twiddles2_dbl,
            )
        };

        let mut expected = values.map(|v| v.to_array()[0]);
        for i in 0..8 {
            let j = i ^ 1;
            if i > j {
                continue;
            }
            let (mut v0, mut v1) = (expected[i], expected[j]);
            ground_truth_ibutterfly(&mut v0, &mut v1, twiddles0[i / 2]);
            (expected[i], expected[j]) = (v0, v1);
        }
        for i in 0..8 {
            let j = i ^ 2;
            if i > j {
                continue;
            }
            let (mut v0, mut v1) = (expected[i], expected[j]);
            ground_truth_ibutterfly(&mut v0, &mut v1, twiddles1[i / 4]);
            (expected[i], expected[j]) = (v0, v1);
        }
        for i in 0..8 {
            let j = i ^ 4;
            if i > j {
                continue;
            }
            let (mut v0, mut v1) = (expected[i], expected[j]);
            ground_truth_ibutterfly(&mut v0, &mut v1, twiddles2[0]);
            (expected[i], expected[j]) = (v0, v1);
        }
        for i in 0..8 {
            assert_eq!(
                res[i].to_array(),
                [expected[i]; N_LANES],
                "mismatch at i={i}"
            );
        }
    }

    #[test]
    fn test_vecwise_ibutterflies() {
        let domain = CanonicCoset::new(5).circle_domain();
        let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);
        assert_eq!(twiddle_dbls.len(), 4);
        let mut rng = SmallRng::seed_from_u64(0);
        let values: [[BaseField; 16]; 2] = rng.gen();

        let res = {
            let (val0, val1) = vecwise_ibutterflies(
                values[0].into(),
                values[1].into(),
                twiddle_dbls[0].clone().try_into().unwrap(),
                twiddle_dbls[1].clone().try_into().unwrap(),
                twiddle_dbls[2].clone().try_into().unwrap(),
            );
            let (val0, val1) = simd_ibutterfly(val0, val1, u32x16::splat(twiddle_dbls[3][0]));
            [val0.to_array(), val1.to_array()].concat()
        };

        assert_eq!(res, ground_truth_ifft(domain, values.as_flattened()));
    }

    #[test]
    fn test_ifft_lower_with_vecwise() {
        for log_size in 5..12 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let mut rng = SmallRng::seed_from_u64(0);
            let values = (0..domain.size()).map(|_| rng.gen()).collect_vec();
            let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);

            let mut res = values.iter().copied().collect::<BaseColumn>();
            unsafe {
                ifft_lower_with_vecwise(
                    transmute::<*mut PackedBaseField, *mut u32>(res.data.as_mut_ptr()),
                    &twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec(),
                    log_size as usize,
                    log_size as usize,
                );
            }

            assert_eq!(res.to_cpu(), ground_truth_ifft(domain, &values));
        }
    }

    #[test]
    fn test_ifft_full() {
        for log_size in CACHED_FFT_LOG_SIZE + 1..CACHED_FFT_LOG_SIZE + 3 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let mut rng = SmallRng::seed_from_u64(0);
            let values = (0..domain.size()).map(|_| rng.gen()).collect_vec();
            let twiddle_dbls = get_itwiddle_dbls(domain.half_coset);

            let mut res = values.iter().copied().collect::<BaseColumn>();
            unsafe {
                ifft(
                    transmute::<*mut PackedBaseField, *mut u32>(res.data.as_mut_ptr()),
                    &twiddle_dbls.iter().map(|x| x.as_slice()).collect_vec(),
                    log_size as usize,
                );
                transpose_vecs(
                    transmute::<*mut PackedBaseField, *mut u32>(res.data.as_mut_ptr()),
                    log_size as usize - 4,
                );
            }

            assert_eq!(res.to_cpu(), ground_truth_ifft(domain, &values));
        }
    }

    fn ground_truth_ifft(domain: CircleDomain, values: &[BaseField]) -> Vec<BaseField> {
        let eval = CpuCircleEvaluation::new(domain, values.to_vec());
        let mut res = eval.interpolate().coeffs;
        let denorm = BaseField::from(domain.size());
        res.iter_mut().for_each(|v| *v *= denorm);
        res
    }
}
