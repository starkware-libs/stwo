//! Regular (forward) fft.

use std::arch::x86_64::{
    __m512i, _mm512_broadcast_i32x4, _mm512_mul_epu32, _mm512_permutex2var_epi32,
    _mm512_set1_epi32, _mm512_set1_epi64, _mm512_srli_epi64,
};

use super::{compute_first_twiddles, EVENS_INTERLEAVE_EVENS, ODDS_INTERLEAVE_ODDS};
use crate::core::backend::avx512::fft::{transpose_vecs, CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
use crate::core::backend::avx512::{PackedBaseField, VECS_LOG_SIZE};
use crate::core::circle::Coset;
use crate::core::utils::bit_reverse;

/// Performs a Circle Fast Fourier Transform (ICFFT) on the given values.
///
/// # Safety
/// This function is unsafe because it takes a raw pointer to i32 values.
/// `values` must be aligned to 64 bytes.
///
/// # Arguments
/// * `src`: A pointer to the values to transform.
/// * `dst`: A pointer to the destination array.
/// * `twiddle_dbl`: A reference to the doubles of the twiddle factors.
/// * `log_n_elements`: The log of the number of elements in the `values` array.
///
/// # Panics
/// This function will panic if `log_n_elements` is less than `MIN_FFT_LOG_SIZE`.
pub unsafe fn fft(src: *const i32, dst: *mut i32, twiddle_dbl: &[&[i32]], log_n_elements: usize) {
    assert!(log_n_elements >= MIN_FFT_LOG_SIZE);
    let log_n_vecs = log_n_elements - VECS_LOG_SIZE;
    if log_n_elements <= CACHED_FFT_LOG_SIZE {
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
    transpose_vecs(dst, log_n_vecs);
    fft_lower_with_vecwise(
        dst,
        dst,
        &twiddle_dbl[..(3 + fft_layers_pre_transpose)],
        log_n_elements,
        fft_layers_pre_transpose + VECS_LOG_SIZE,
    );
}

/// Computes partial fft on `2^log_size` M31 elements.
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for each layer of the the fft.
///     layer i holds 2^(log_size - 1 - i) twiddles.
///   log_size - The log of the number of number of M31 elements in the array.
///   fft_layers - The number of fft layers to apply, out of log_size.
/// # Safety
/// `values` must be aligned to 64 bytes.
/// `log_size` must be at least 5.
/// `fft_layers` must be at least 5.
pub unsafe fn fft_lower_with_vecwise(
    src: *const i32,
    dst: *mut i32,
    twiddle_dbl: &[&[i32]],
    log_size: usize,
    fft_layers: usize,
) {
    const VECWISE_FFT_BITS: usize = VECS_LOG_SIZE + 1;
    assert!(log_size >= VECWISE_FFT_BITS);

    assert_eq!(twiddle_dbl[0].len(), 1 << (log_size - 2));

    for index_h in 0..(1 << (log_size - fft_layers)) {
        let mut src = src;
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
}

/// Computes partial fft on `2^log_size` M31 elements, skipping the vecwise layers (lower 4 bits
///   of the index).
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for each layer of the the fft.
///   log_size - The log of the number of number of M31 elements in the array.
///   fft_layers - The number of fft layers to apply, out of log_size - VEC_LOG_SIZE.
///
/// # Safety
/// `values` must be aligned to 64 bytes.
/// `log_size` must be at least 4.
/// `fft_layers` must be at least 4.
pub unsafe fn fft_lower_without_vecwise(
    src: *const i32,
    dst: *mut i32,
    twiddle_dbl: &[&[i32]],
    log_size: usize,
    fft_layers: usize,
) {
    assert!(log_size >= VECS_LOG_SIZE);

    for index_h in 0..(1 << (log_size - fft_layers - VECS_LOG_SIZE)) {
        let mut src = src;
        for layer in (0..fft_layers).step_by(3).rev() {
            let fixed_layer = layer + VECS_LOG_SIZE;
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
}

/// Runs the last 5 fft layers across the entire array.
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for each of the 5 fft layers.
///   high_bits - The number of bits this loops needs to run on.
///   index_h - The higher part of the index, iterated by the caller.
/// # Safety
unsafe fn fft_vecwise_loop(
    src: *const i32,
    dst: *mut i32,
    twiddle_dbl: &[&[i32]],
    loop_bits: usize,
    index_h: usize,
) {
    for index_l in 0..(1 << loop_bits) {
        let index = (index_h << loop_bits) + index_l;
        let mut val0 = PackedBaseField::load(src.add(index * 32));
        let mut val1 = PackedBaseField::load(src.add(index * 32 + 16));
        (val0, val1) = avx_butterfly(
            val0,
            val1,
            _mm512_set1_epi32(*twiddle_dbl[3].get_unchecked(index)),
        );
        (val0, val1) = vecwise_butterflies(
            val0,
            val1,
            std::array::from_fn(|i| *twiddle_dbl[0].get_unchecked(index * 8 + i)),
            std::array::from_fn(|i| *twiddle_dbl[1].get_unchecked(index * 4 + i)),
            std::array::from_fn(|i| *twiddle_dbl[2].get_unchecked(index * 2 + i)),
        );
        val0.store(dst.add(index * 32));
        val1.store(dst.add(index * 32 + 16));
    }
}

/// Runs 3 fft layers across the entire array.
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for each of the 3 fft layers.
///   loop_bits - The number of bits this loops needs to run on.
///   layer - The layer number of the first fft layer to apply.
///     The layers `layer`, `layer + 1`, `layer + 2` are applied.
///   index_h - The higher part of the index, iterated by the caller.
/// # Safety
unsafe fn fft3_loop(
    src: *const i32,
    dst: *mut i32,
    twiddle_dbl: &[&[i32]],
    loop_bits: usize,
    layer: usize,
    index_h: usize,
) {
    for index_l in 0..(1 << loop_bits) {
        let index = (index_h << loop_bits) + index_l;
        let offset = index << (layer + 3);
        for l in (0..(1 << layer)).step_by(1 << VECS_LOG_SIZE) {
            fft3(
                src,
                dst,
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

/// Runs 2 fft layers across the entire array.
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for each of the 2 fft layers.
///   loop_bits - The number of bits this loops needs to run on.
///   layer - The layer number of the first fft layer to apply.
///     The layers `layer`, `layer + 1` are applied.
///   index - The index, iterated by the caller.
/// # Safety
unsafe fn fft2_loop(
    src: *const i32,
    dst: *mut i32,
    twiddle_dbl: &[&[i32]],
    layer: usize,
    index: usize,
) {
    let offset = index << (layer + 2);
    for l in (0..(1 << layer)).step_by(1 << VECS_LOG_SIZE) {
        fft2(
            src,
            dst,
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

/// Runs 1 fft layer across the entire array.
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for the fft layer.
///   layer - The layer number of the fft layer to apply.
///   index_h - The higher part of the index, iterated by the caller.
/// # Safety
unsafe fn fft1_loop(
    src: *const i32,
    dst: *mut i32,
    twiddle_dbl: &[&[i32]],
    layer: usize,
    index: usize,
) {
    let offset = index << (layer + 1);
    for l in (0..(1 << layer)).step_by(1 << VECS_LOG_SIZE) {
        fft1(
            src,
            dst,
            offset + l,
            layer,
            std::array::from_fn(|i| {
                *twiddle_dbl[0].get_unchecked((index + i) & (twiddle_dbl[0].len() - 1))
            }),
        );
    }
}

/// Computes the butterfly operation for packed M31 elements.
///   val0 + t val1, val0 - t val1.
/// val0, val1 are packed M31 elements. 16 M31 words at each.
/// Each value is assumed to be in unreduced form, [0, P] including P.
/// Returned values are in unreduced form, [0, P] including P.
/// twiddle_dbl holds 16 values, each is a *double* of a twiddle factor, in unreduced form.
/// # Safety
/// This function is safe.
pub unsafe fn avx_butterfly(
    val0: PackedBaseField,
    val1: PackedBaseField,
    twiddle_dbl: __m512i,
) -> (PackedBaseField, PackedBaseField) {
    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of val0.
    let val1_e = val1.0;
    // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of val0.
    let val1_o = _mm512_srli_epi64(val1.0, 32);
    let twiddle_dbl_e = twiddle_dbl;
    let twiddle_dbl_o = _mm512_srli_epi64(twiddle_dbl, 32);

    // To compute prod = val1 * twiddle start by multiplying
    // val1_e/o by twiddle_dbl_e/o.
    let prod_e_dbl = _mm512_mul_epu32(val1_e, twiddle_dbl_e);
    let prod_o_dbl = _mm512_mul_epu32(val1_o, twiddle_dbl_o);

    // The result of a multiplication holds val1*twiddle_dbl in as 64-bits.
    // Each 64b-bit word looks like this:
    //               1    31       31    1
    // prod_e_dbl - |0|prod_e_h|prod_e_l|0|
    // prod_o_dbl - |0|prod_o_h|prod_o_l|0|

    // Interleave the even words of prod_e_dbl with the even words of prod_o_dbl:
    let prod_ls = _mm512_permutex2var_epi32(prod_e_dbl, EVENS_INTERLEAVE_EVENS, prod_o_dbl);
    // prod_ls -    |prod_o_l|0|prod_e_l|0|

    // Divide by 2:
    let prod_ls = _mm512_srli_epi64(prod_ls, 1);
    // prod_ls -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
    let prod_hs = _mm512_permutex2var_epi32(prod_e_dbl, ODDS_INTERLEAVE_ODDS, prod_o_dbl);
    // prod_hs -    |0|prod_o_h|0|prod_e_h|

    let prod = PackedBaseField(prod_ls) + PackedBaseField(prod_hs);

    let r0 = val0 + prod;
    let r1 = val0 - prod;

    (r0, r1)
}

/// Runs fft on 2 vectors of 16 M31 elements.
/// This amounts to 4 butterfly layers, each with 16 butterflies.
/// Each of the vectors represents natural ordered polynomial coefficeint.
/// Each value in a vectors is in unreduced form: [0, P] including P.
/// Takes 4 twiddle arrays, one for each layer, holding the double of the corresponding twiddle.
/// The first layer (higher bit of the index) takes 2 twiddles.
/// The second layer takes 4 twiddles.
/// etc.
/// # Safety
pub unsafe fn vecwise_butterflies(
    mut val0: PackedBaseField,
    mut val1: PackedBaseField,
    twiddle1_dbl: [i32; 8],
    twiddle2_dbl: [i32; 4],
    twiddle3_dbl: [i32; 2],
) -> (PackedBaseField, PackedBaseField) {
    // TODO(spapini): Compute twiddle0 from twiddle1.
    // TODO(spapini): The permute can be fused with the _mm512_srli_epi64 inside the butterfly.
    // The implementation is the exact reverse of vecwise_ibutterflies().
    // See the comments in its body for more info.
    let t = _mm512_set1_epi64(std::mem::transmute(twiddle3_dbl));
    (val0, val1) = val0.interleave_with(val1);
    (val0, val1) = avx_butterfly(val0, val1, t);

    let t = _mm512_broadcast_i32x4(std::mem::transmute(twiddle2_dbl));
    (val0, val1) = val0.interleave_with(val1);
    (val0, val1) = avx_butterfly(val0, val1, t);

    let (t0, t1) = compute_first_twiddles(twiddle1_dbl);
    (val0, val1) = val0.interleave_with(val1);
    (val0, val1) = avx_butterfly(val0, val1, t1);

    (val0, val1) = val0.interleave_with(val1);
    (val0, val1) = avx_butterfly(val0, val1, t0);

    val0.interleave_with(val1)
}

/// Returns the line twiddles (x points) for an fft on a coset.
pub fn get_twiddle_dbls(mut coset: Coset) -> Vec<Vec<i32>> {
    let mut res = vec![];
    for _ in 0..coset.log_size() {
        res.push(
            coset
                .iter()
                .take(coset.size() / 2)
                .map(|p| (p.x.0 * 2) as i32)
                .collect::<Vec<_>>(),
        );
        bit_reverse(res.last_mut().unwrap());
        coset = coset.double();
    }

    res
}

/// Applies 3 butterfly layers on 8 vectors of 16 M31 elements.
/// Vectorized over the 16 elements of the vectors.
/// Used for radix-8 ifft.
/// Each butterfly layer, has 3 AVX butterflies.
/// Total of 12 AVX butterflies.
/// Parameters:
///   values - Pointer to the entire value array.
///   offset - The offset of the first value in the array.
///   log_step - The log of the distance in the array, in M31 elements, between each pair of
///     values that need to be transformed. For layer i this is i - 4.
///   twiddles_dbl0/1/2 - The double of the twiddles for the 3 layers of butterflies.
///   Each layer has 4/2/1 twiddles.
/// # Safety
pub unsafe fn fft3(
    src: *const i32,
    dst: *mut i32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [i32; 4],
    twiddles_dbl1: [i32; 2],
    twiddles_dbl2: [i32; 1],
) {
    // Load the 8 AVX vectors from the array.
    let mut val0 = PackedBaseField::load(src.add(offset + (0 << log_step)));
    let mut val1 = PackedBaseField::load(src.add(offset + (1 << log_step)));
    let mut val2 = PackedBaseField::load(src.add(offset + (2 << log_step)));
    let mut val3 = PackedBaseField::load(src.add(offset + (3 << log_step)));
    let mut val4 = PackedBaseField::load(src.add(offset + (4 << log_step)));
    let mut val5 = PackedBaseField::load(src.add(offset + (5 << log_step)));
    let mut val6 = PackedBaseField::load(src.add(offset + (6 << log_step)));
    let mut val7 = PackedBaseField::load(src.add(offset + (7 << log_step)));

    // Apply the third layer of butterflies.
    (val0, val4) = avx_butterfly(val0, val4, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val1, val5) = avx_butterfly(val1, val5, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val2, val6) = avx_butterfly(val2, val6, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val3, val7) = avx_butterfly(val3, val7, _mm512_set1_epi32(twiddles_dbl2[0]));

    // Apply the second layer of butterflies.
    (val0, val2) = avx_butterfly(val0, val2, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val1, val3) = avx_butterfly(val1, val3, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val4, val6) = avx_butterfly(val4, val6, _mm512_set1_epi32(twiddles_dbl1[1]));
    (val5, val7) = avx_butterfly(val5, val7, _mm512_set1_epi32(twiddles_dbl1[1]));

    // Apply the first layer of butterflies.
    (val0, val1) = avx_butterfly(val0, val1, _mm512_set1_epi32(twiddles_dbl0[0]));
    (val2, val3) = avx_butterfly(val2, val3, _mm512_set1_epi32(twiddles_dbl0[1]));
    (val4, val5) = avx_butterfly(val4, val5, _mm512_set1_epi32(twiddles_dbl0[2]));
    (val6, val7) = avx_butterfly(val6, val7, _mm512_set1_epi32(twiddles_dbl0[3]));

    // Store the 8 AVX vectors back to the array.
    val0.store(dst.add(offset + (0 << log_step)));
    val1.store(dst.add(offset + (1 << log_step)));
    val2.store(dst.add(offset + (2 << log_step)));
    val3.store(dst.add(offset + (3 << log_step)));
    val4.store(dst.add(offset + (4 << log_step)));
    val5.store(dst.add(offset + (5 << log_step)));
    val6.store(dst.add(offset + (6 << log_step)));
    val7.store(dst.add(offset + (7 << log_step)));
}

/// Applies 2 butterfly layers on 4 vectors of 16 M31 elements.
/// Vectorized over the 16 elements of the vectors.
/// Used for radix-4 fft.
/// Each butterfly layer, has 2 AVX butterflies.
/// Total of 4 AVX butterflies.
/// Parameters:
///   values - Pointer to the entire value array.
///   offset - The offset of the first value in the array.
///   log_step - The log of the distance in the array, in M31 elements, between each pair of
///     values that need to be transformed. For layer i this is i - 4.
///   twiddles_dbl0/1 - The double of the twiddles for the 2 layers of butterflies.
///   Each layer has 2/1 twiddles.
/// # Safety
pub unsafe fn fft2(
    src: *const i32,
    dst: *mut i32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [i32; 2],
    twiddles_dbl1: [i32; 1],
) {
    // Load the 4 AVX vectors from the array.
    let mut val0 = PackedBaseField::load(src.add(offset + (0 << log_step)));
    let mut val1 = PackedBaseField::load(src.add(offset + (1 << log_step)));
    let mut val2 = PackedBaseField::load(src.add(offset + (2 << log_step)));
    let mut val3 = PackedBaseField::load(src.add(offset + (3 << log_step)));

    // Apply the second layer of butterflies.
    (val0, val2) = avx_butterfly(val0, val2, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val1, val3) = avx_butterfly(val1, val3, _mm512_set1_epi32(twiddles_dbl1[0]));

    // Apply the first layer of butterflies.
    (val0, val1) = avx_butterfly(val0, val1, _mm512_set1_epi32(twiddles_dbl0[0]));
    (val2, val3) = avx_butterfly(val2, val3, _mm512_set1_epi32(twiddles_dbl0[1]));

    // Store the 4 AVX vectors back to the array.
    val0.store(dst.add(offset + (0 << log_step)));
    val1.store(dst.add(offset + (1 << log_step)));
    val2.store(dst.add(offset + (2 << log_step)));
    val3.store(dst.add(offset + (3 << log_step)));
}

/// Applies 1 butterfly layers on 2 vectors of 16 M31 elements.
/// Vectorized over the 16 elements of the vectors.
/// Parameters:
///   values - Pointer to the entire value array.
///   offset - The offset of the first value in the array.
///   log_step - The log of the distance in the array, in M31 elements, between each pair of
///     values that need to be transformed. For layer i this is i - 4.
///   twiddles_dbl0 - The double of the twiddles for the butterfly layer.
/// # Safety
pub unsafe fn fft1(
    src: *const i32,
    dst: *mut i32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [i32; 1],
) {
    // Load the 2 AVX vectors from the array.
    let mut val0 = PackedBaseField::load(src.add(offset + (0 << log_step)));
    let mut val1 = PackedBaseField::load(src.add(offset + (1 << log_step)));

    (val0, val1) = avx_butterfly(val0, val1, _mm512_set1_epi32(twiddles_dbl0[0]));

    // Store the 2 AVX vectors back to the array.
    val0.store(dst.add(offset + (0 << log_step)));
    val1.store(dst.add(offset + (1 << log_step)));
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use std::arch::x86_64::{_mm512_add_epi32, _mm512_set1_epi32, _mm512_setr_epi32};

    use super::*;
    use crate::core::backend::avx512::{BaseFieldVec, PackedBaseField};
    use crate::core::backend::cpu::CPUCirclePoly;
    use crate::core::backend::Column;
    use crate::core::fft::butterfly;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain};

    #[test]
    fn test_butterfly() {
        unsafe {
            let val0 = PackedBaseField(_mm512_setr_epi32(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            ));
            let val1 = PackedBaseField(_mm512_setr_epi32(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ));
            let twiddle = _mm512_setr_epi32(
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            );
            let twiddle_dbl = _mm512_add_epi32(twiddle, twiddle);
            let (r0, r1) = avx_butterfly(val0, val1, twiddle_dbl);

            let val0: [BaseField; 16] = std::mem::transmute(val0);
            let val1: [BaseField; 16] = std::mem::transmute(val1);
            let twiddle: [BaseField; 16] = std::mem::transmute(twiddle);
            let r0: [BaseField; 16] = std::mem::transmute(r0);
            let r1: [BaseField; 16] = std::mem::transmute(r1);

            for i in 0..16 {
                let mut x = val0[i];
                let mut y = val1[i];
                let twiddle = twiddle[i];
                butterfly(&mut x, &mut y, twiddle);
                assert_eq!(x, r0[i]);
                assert_eq!(y, r1[i]);
            }
        }
    }

    #[test]
    fn test_fft3() {
        unsafe {
            let mut values: Vec<PackedBaseField> = (0..8)
                .map(|i| {
                    PackedBaseField::from_array(std::array::from_fn(|_| {
                        BaseField::from_u32_unchecked(i)
                    }))
                })
                .collect();
            let twiddles0 = [32, 33, 34, 35];
            let twiddles1 = [36, 37];
            let twiddles2 = [38];
            let twiddles0_dbl = std::array::from_fn(|i| twiddles0[i] * 2);
            let twiddles1_dbl = std::array::from_fn(|i| twiddles1[i] * 2);
            let twiddles2_dbl = std::array::from_fn(|i| twiddles2[i] * 2);
            fft3(
                std::mem::transmute(values.as_ptr()),
                std::mem::transmute(values.as_mut_ptr()),
                0,
                VECS_LOG_SIZE,
                twiddles0_dbl,
                twiddles1_dbl,
                twiddles2_dbl,
            );

            let expected: [u32; 8] = std::array::from_fn(|i| i as u32);
            let mut expected: [BaseField; 8] = std::mem::transmute(expected);
            let twiddles0: [BaseField; 4] = std::mem::transmute(twiddles0);
            let twiddles1: [BaseField; 2] = std::mem::transmute(twiddles1);
            let twiddles2: [BaseField; 1] = std::mem::transmute(twiddles2);
            for i in 0..8 {
                let j = i ^ 4;
                if i > j {
                    continue;
                }
                let (mut v0, mut v1) = (expected[i], expected[j]);
                butterfly(&mut v0, &mut v1, twiddles2[0]);
                (expected[i], expected[j]) = (v0, v1);
            }
            for i in 0..8 {
                let j = i ^ 2;
                if i > j {
                    continue;
                }
                let (mut v0, mut v1) = (expected[i], expected[j]);
                butterfly(&mut v0, &mut v1, twiddles1[i / 4]);
                (expected[i], expected[j]) = (v0, v1);
            }
            for i in 0..8 {
                let j = i ^ 1;
                if i > j {
                    continue;
                }
                let (mut v0, mut v1) = (expected[i], expected[j]);
                butterfly(&mut v0, &mut v1, twiddles0[i / 2]);
                (expected[i], expected[j]) = (v0, v1);
            }
            for i in 0..8 {
                assert_eq!(values[i].to_array()[0], expected[i]);
            }
        }
    }

    fn ref_fft(domain: CircleDomain, values: Vec<BaseField>) -> Vec<BaseField> {
        let poly = CPUCirclePoly::new(values);
        poly.evaluate(domain).values
    }

    #[test]
    fn test_vecwise_butterflies() {
        let domain = CanonicCoset::new(5).circle_domain();
        let twiddle_dbls = get_twiddle_dbls(domain.half_coset);
        assert_eq!(twiddle_dbls.len(), 4);
        let values0: [i32; 16] = std::array::from_fn(|i| i as i32);
        let values1: [i32; 16] = std::array::from_fn(|i| (i + 16) as i32);
        let result: [BaseField; 32] = unsafe {
            let (val0, val1) = avx_butterfly(
                std::mem::transmute(values0),
                std::mem::transmute(values1),
                _mm512_set1_epi32(twiddle_dbls[3][0]),
            );
            let (val0, val1) = vecwise_butterflies(
                val0,
                val1,
                twiddle_dbls[0].clone().try_into().unwrap(),
                twiddle_dbls[1].clone().try_into().unwrap(),
                twiddle_dbls[2].clone().try_into().unwrap(),
            );
            std::mem::transmute([val0, val1])
        };

        // ref.
        let mut values = values0.to_vec();
        values.extend_from_slice(&values1);
        let expected = ref_fft(domain, values.into_iter().map(BaseField::from).collect());

        // Compare.
        for i in 0..32 {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_fft_lower() {
        for log_size in 5..12 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let values = (0..domain.size())
                .map(|i| BaseField::from_u32_unchecked(i as u32))
                .collect::<Vec<_>>();
            let expected_coeffs = ref_fft(domain, values.clone());

            // Compute.
            let mut values = BaseFieldVec::from_iter(values);
            let twiddle_dbls = get_twiddle_dbls(domain.half_coset);

            unsafe {
                fft_lower_with_vecwise(
                    std::mem::transmute(values.data.as_ptr()),
                    std::mem::transmute(values.data.as_mut_ptr()),
                    &twiddle_dbls
                        .iter()
                        .map(|x| x.as_slice())
                        .collect::<Vec<_>>(),
                    log_size as usize,
                    log_size as usize,
                );

                // Compare.
                assert_eq!(values.to_vec(), expected_coeffs);
            }
        }
    }

    fn run_fft_full_test(log_size: u32) {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = (0..domain.size())
            .map(|i| BaseField::from_u32_unchecked(i as u32))
            .collect::<Vec<_>>();
        let expected_coeffs = ref_fft(domain, values.clone());

        // Compute.
        let mut values = BaseFieldVec::from_iter(values);
        let twiddle_dbls = get_twiddle_dbls(domain.half_coset);

        unsafe {
            transpose_vecs(
                std::mem::transmute(values.data.as_mut_ptr()),
                (log_size - 4) as usize,
            );
            fft(
                std::mem::transmute(values.data.as_ptr()),
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddle_dbls
                    .iter()
                    .map(|x| x.as_slice())
                    .collect::<Vec<_>>(),
                log_size as usize,
            );

            // Compare.
            assert_eq!(values.to_vec(), expected_coeffs);
        }
    }

    #[test]
    fn test_fft_full() {
        for i in (CACHED_FFT_LOG_SIZE + 1)..(CACHED_FFT_LOG_SIZE + 3) {
            run_fft_full_test(i as u32);
        }
    }
}
