use std::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_broadcast_i32x4, _mm512_broadcast_i64x4, _mm512_load_epi32,
    _mm512_min_epu32, _mm512_mul_epu32, _mm512_permutex2var_epi32, _mm512_set1_epi32,
    _mm512_set1_epi64, _mm512_srli_epi64, _mm512_store_epi32, _mm512_sub_epi32,
};

use crate::core::backend::avx512::VECS_LOG_SIZE;

/// An input to _mm512_permutex2var_epi32, and is used to interleave the even words of a
/// with the even words of b.
const EVENS_INTERLEAVE_EVENS: __m512i = unsafe {
    core::mem::transmute([
        0b00000, 0b10000, 0b00010, 0b10010, 0b00100, 0b10100, 0b00110, 0b10110, 0b01000, 0b11000,
        0b01010, 0b11010, 0b01100, 0b11100, 0b01110, 0b11110,
    ])
};
/// An input to _mm512_permutex2var_epi32, and is used to interleave the odd words of a
/// with the odd words of b.
const ODDS_INTERLEAVE_ODDS: __m512i = unsafe {
    core::mem::transmute([
        0b00001, 0b10001, 0b00011, 0b10011, 0b00101, 0b10101, 0b00111, 0b10111, 0b01001, 0b11001,
        0b01011, 0b11011, 0b01101, 0b11101, 0b01111, 0b11111,
    ])
};

/// An input to _mm512_permutex2var_epi32, and is used to concat the even words of a
/// with the even words of b.
const EVENS_CONCAT_EVENS: __m512i = unsafe {
    core::mem::transmute([
        0b00000, 0b00010, 0b00100, 0b00110, 0b01000, 0b01010, 0b01100, 0b01110, 0b10000, 0b10010,
        0b10100, 0b10110, 0b11000, 0b11010, 0b11100, 0b11110,
    ])
};
/// An input to _mm512_permutex2var_epi32, and is used to concat the odd words of a
/// with the odd words of b.
const ODDS_CONCAT_ODDS: __m512i = unsafe {
    core::mem::transmute([
        0b00001, 0b00011, 0b00101, 0b00111, 0b01001, 0b01011, 0b01101, 0b01111, 0b10001, 0b10011,
        0b10101, 0b10111, 0b11001, 0b11011, 0b11101, 0b11111,
    ])
};
/// An input to _mm512_permutex2var_epi32, and is used to interleave the low half of a
/// with the low half of b.
const LHALF_INTERLEAVE_LHALF: __m512i = unsafe {
    core::mem::transmute([
        0b00000, 0b10000, 0b00001, 0b10001, 0b00010, 0b10010, 0b00011, 0b10011, 0b00100, 0b10100,
        0b00101, 0b10101, 0b00110, 0b10110, 0b00111, 0b10111,
    ])
};
/// An input to _mm512_permutex2var_epi32, and is used to interleave the high half of a
/// with the high half of b.
const HHALF_INTERLEAVE_HHALF: __m512i = unsafe {
    core::mem::transmute([
        0b01000, 0b11000, 0b01001, 0b11001, 0b01010, 0b11010, 0b01011, 0b11011, 0b01100, 0b11100,
        0b01101, 0b11101, 0b01110, 0b11110, 0b01111, 0b11111,
    ])
};
const P: __m512i = unsafe { core::mem::transmute([(1u32 << 31) - 1; 16]) };

// TODO(spapini): FFTs return a redundant representation, that can get the value P. need to reduce
// it somewhere.

/// Computes partial ifft on `2^log_size` M31 elements.
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for each layer of the the ifft.
///     layer i holds 2^(log_size - 1 - i) twiddles.
///   log_size - The log of the number of number of M31 elements in the array.
///   fft_layers - The number of ifft layers to apply, out of log_size.
/// # Safety
/// `values` must be aligned to 64 bytes.
/// `log_size` must be at least 5.
/// `fft_layers` must be at least 5.
pub unsafe fn ifft_lower_with_vecwise(
    values: *mut i32,
    twiddle_dbl: &[Vec<i32>],
    log_size: usize,
    fft_layers: usize,
) {
    const VECWISE_FFT_BITS: usize = VECS_LOG_SIZE + 1;
    assert!(log_size >= VECWISE_FFT_BITS);

    assert_eq!(twiddle_dbl[0].len(), 1 << (log_size - 1));

    for index_h in 0..(1 << (log_size - fft_layers)) {
        ifft_vecwise_loop(values, twiddle_dbl, fft_layers - VECWISE_FFT_BITS, index_h);
        let mut layer = VECWISE_FFT_BITS;
        while fft_layers - layer >= 3 {
            ifft3_loop(
                values,
                &twiddle_dbl[layer..],
                fft_layers - layer - 3,
                layer,
                index_h,
            );
            layer += 3;
        }
        if fft_layers - layer != 0 {
            todo!()
        }
    }
}

/// Runs the first 5 ifft layers across the entire array.
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for each of the 5 ifft layers.
///   high_bits - The number of bits this loops needs to run on.
///   index_h - The higher part of the index, iterated by the caller.
/// # Safety
unsafe fn ifft_vecwise_loop(
    values: *mut i32,
    twiddle_dbl: &[Vec<i32>],
    loop_bits: usize,
    index_h: usize,
) {
    for index_l in 0..(1 << loop_bits) {
        let index = (index_h << loop_bits) + index_l;
        let mut val0 = _mm512_load_epi32(values.add(index * 32).cast_const());
        let mut val1 = _mm512_load_epi32(values.add(index * 32 + 16).cast_const());
        (val0, val1) = vecwise_ibutterflies(
            val0,
            val1,
            std::array::from_fn(|i| *twiddle_dbl[0].get_unchecked(index * 16 + i)),
            std::array::from_fn(|i| *twiddle_dbl[1].get_unchecked(index * 8 + i)),
            std::array::from_fn(|i| *twiddle_dbl[2].get_unchecked(index * 4 + i)),
            std::array::from_fn(|i| *twiddle_dbl[3].get_unchecked(index * 2 + i)),
        );
        (val0, val1) = avx_ibutterfly(
            val0,
            val1,
            _mm512_set1_epi32(*twiddle_dbl[4].get_unchecked(index)),
        );
        _mm512_store_epi32(values.add(index * 32), val0);
        _mm512_store_epi32(values.add(index * 32 + 16), val1);
    }
}

/// Runs 3 ifft layers across the entire array.
/// Parameters:
///   values - Pointer to the entire value array, aligned to 64 bytes.
///   twiddle_dbl - The doubles of the twiddle factors for each of the 3 ifft layers.
///   loop_bits - The number of bits this loops needs to run on.
///   layer - The layer number of the first ifft layer to apply.
///     The layers `layer`, `layer + 1`, `layer + 2` are applied.
///   index_h - The higher part of the index, iterated by the caller.
/// # Safety
unsafe fn ifft3_loop(
    values: *mut i32,
    twiddle_dbl: &[Vec<i32>],
    loop_bits: usize,
    layer: usize,
    index_h: usize,
) {
    for index_l in 0..(1 << loop_bits) {
        let index = (index_h << loop_bits) + index_l;
        let offset = index << (layer + 3);
        for l in (0..(1 << layer)).step_by(1 << VECS_LOG_SIZE) {
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

/// Computes the butterfly operation for packed M31 elements.
///   val0 + t val1, val0 - t val1.
/// val0, val1 are packed M31 elements. 16 M31 words at each.
/// Each value is assumed to be in unreduced form, [0, P] including P.
/// Returned values are in unreduced form, [0, P] including P.
/// twiddle_dbl holds 16 values, each is a *double* of a twiddle factor, in reduced form.
/// # Safety
/// This function is safe.
pub unsafe fn avx_butterfly(
    val0: __m512i,
    val1: __m512i,
    twiddle_dbl: __m512i,
) -> (__m512i, __m512i) {
    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of val0.
    let val1_e = val1;
    // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of val0.
    let val1_o = _mm512_srli_epi64(val1, 32);
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

    let prod = add_mod_p(prod_ls, prod_hs);

    let r0 = add_mod_p(val0, prod);
    let r1 = sub_mod_p(val0, prod);

    (r0, r1)
}

/// Computes the ibutterfly operation for packed M31 elements.
///   val0 + val1, t (val0 - val1).
/// val0, val1 are packed M31 elements. 16 M31 words at each.
/// Each value is assumed to be in unreduced form, [0, P] including P.
/// twiddle_dbl holds 16 values, each is a *double* of a twiddle factor, in reduced form.
/// # Safety
/// This function is safe.
pub unsafe fn avx_ibutterfly(
    val0: __m512i,
    val1: __m512i,
    twiddle_dbl: __m512i,
) -> (__m512i, __m512i) {
    let r0 = add_mod_p(val0, val1);
    let r1 = sub_mod_p(val0, val1);

    // Extract the even and odd parts of r1 and twiddle_dbl, and spread as 8 64bit values.
    let r1_e = r1;
    let r1_o = _mm512_srli_epi64(r1, 32);
    let twiddle_dbl_e = twiddle_dbl;
    let twiddle_dbl_o = _mm512_srli_epi64(twiddle_dbl, 32);

    // To compute prod = r1 * twiddle start by multiplying
    // r1_e/o by twiddle_dbl_e/o.
    let prod_e_dbl = _mm512_mul_epu32(r1_e, twiddle_dbl_e);
    let prod_o_dbl = _mm512_mul_epu32(r1_o, twiddle_dbl_o);

    // The result of a multiplication holds r1*twiddle_dbl in as 64-bits.
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

    let prod = add_mod_p(prod_ls, prod_hs);

    (r0, prod)
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
    mut val0: __m512i,
    mut val1: __m512i,
    twiddle0_dbl: [i32; 16],
    twiddle1_dbl: [i32; 8],
    twiddle2_dbl: [i32; 4],
    twiddle3_dbl: [i32; 2],
) -> (__m512i, __m512i) {
    // TODO(spapini): Compute twiddle0 from twiddle1.
    // TODO(spapini): The permute can be fused with the _mm512_srli_epi64 inside the butterfly.
    // The implementation is the exact reverse of vecwise_ibutterflies().
    // See the comments in its body for more info.
    let t = _mm512_set1_epi64(std::mem::transmute(twiddle3_dbl));
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, LHALF_INTERLEAVE_LHALF, val1),
        _mm512_permutex2var_epi32(val0, HHALF_INTERLEAVE_HHALF, val1),
    );
    (val0, val1) = avx_butterfly(val0, val1, t);

    let t = _mm512_broadcast_i32x4(std::mem::transmute(twiddle2_dbl));
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, LHALF_INTERLEAVE_LHALF, val1),
        _mm512_permutex2var_epi32(val0, HHALF_INTERLEAVE_HHALF, val1),
    );
    (val0, val1) = avx_butterfly(val0, val1, t);

    let t = _mm512_broadcast_i64x4(std::mem::transmute(twiddle1_dbl));
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, LHALF_INTERLEAVE_LHALF, val1),
        _mm512_permutex2var_epi32(val0, HHALF_INTERLEAVE_HHALF, val1),
    );
    (val0, val1) = avx_butterfly(val0, val1, t);

    let t: __m512i = std::mem::transmute(twiddle0_dbl);
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, LHALF_INTERLEAVE_LHALF, val1),
        _mm512_permutex2var_epi32(val0, HHALF_INTERLEAVE_HHALF, val1),
    );
    (val0, val1) = avx_butterfly(val0, val1, t);

    (
        _mm512_permutex2var_epi32(val0, LHALF_INTERLEAVE_LHALF, val1),
        _mm512_permutex2var_epi32(val0, HHALF_INTERLEAVE_HHALF, val1),
    )
}

/// Runs ifft on 2 vectors of 16 M31 elements.
/// This amounts to 4 butterfly layers, each with 16 butterflies.
/// Each of the vectors represents a bit reversed evaluation.
/// Each value in a vectors is in unreduced form: [0, P] including P.
/// Takes 4 twiddle arrays, one for each layer, holding the double of the corresponding twiddle.
/// The first layer (lower bit of the index) takes 16 twiddles.
/// The second layer takes 8 twiddles.
/// etc.
/// # Safety
pub unsafe fn vecwise_ibutterflies(
    mut val0: __m512i,
    mut val1: __m512i,
    twiddle0_dbl: [i32; 16],
    twiddle1_dbl: [i32; 8],
    twiddle2_dbl: [i32; 4],
    twiddle3_dbl: [i32; 2],
) -> (__m512i, __m512i) {
    // TODO(spapini): Compute twiddle0 from twiddle1.
    // TODO(spapini): The permute can be fused with the _mm512_srli_epi64 inside the butterfly.

    // Each avx_ibutterfly take 2 512-bit registers, and does 16 butterflies element by element.
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

    // The twiddles for layer 0 are unique and arranged as follows:
    //   0 1 2 3 4 5 6 7 8 9 a b c d e f
    let t: __m512i = std::mem::transmute(twiddle0_dbl);
    // Apply the permutation, resulting in indexing d:iabc.
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, EVENS_CONCAT_EVENS, val1),
        _mm512_permutex2var_epi32(val0, ODDS_CONCAT_ODDS, val1),
    );
    (val0, val1) = avx_ibutterfly(val0, val1, t);

    // The twiddles for layer 1 are replicated in the following pattern:
    //   0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
    let t = _mm512_broadcast_i64x4(std::mem::transmute(twiddle1_dbl));
    // Apply the permutation, resulting in indexing c:diab.
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, EVENS_CONCAT_EVENS, val1),
        _mm512_permutex2var_epi32(val0, ODDS_CONCAT_ODDS, val1),
    );
    (val0, val1) = avx_ibutterfly(val0, val1, t);

    // The twiddles for layer 2 are replicated in the following pattern:
    //   0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
    let t = _mm512_broadcast_i32x4(std::mem::transmute(twiddle2_dbl));
    // Apply the permutation, resulting in indexing b:cdia.
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, EVENS_CONCAT_EVENS, val1),
        _mm512_permutex2var_epi32(val0, ODDS_CONCAT_ODDS, val1),
    );
    (val0, val1) = avx_ibutterfly(val0, val1, t);

    // The twiddles for layer 3 are replicated in the following pattern:
    //  0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
    let t = _mm512_set1_epi64(std::mem::transmute(twiddle3_dbl));
    // Apply the permutation, resulting in indexing a:bcid.
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, EVENS_CONCAT_EVENS, val1),
        _mm512_permutex2var_epi32(val0, ODDS_CONCAT_ODDS, val1),
    );
    (val0, val1) = avx_ibutterfly(val0, val1, t);

    // Apply the permutation, resulting in indexing i:abcd.
    (
        _mm512_permutex2var_epi32(val0, EVENS_CONCAT_EVENS, val1),
        _mm512_permutex2var_epi32(val0, ODDS_CONCAT_ODDS, val1),
    )
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
pub unsafe fn ifft3(
    values: *mut i32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [i32; 4],
    twiddles_dbl1: [i32; 2],
    twiddles_dbl2: [i32; 1],
) {
    // Load the 8 AVX vectors from the array.
    let mut val0 = _mm512_load_epi32(values.add(offset + (0 << log_step)).cast_const());
    let mut val1 = _mm512_load_epi32(values.add(offset + (1 << log_step)).cast_const());
    let mut val2 = _mm512_load_epi32(values.add(offset + (2 << log_step)).cast_const());
    let mut val3 = _mm512_load_epi32(values.add(offset + (3 << log_step)).cast_const());
    let mut val4 = _mm512_load_epi32(values.add(offset + (4 << log_step)).cast_const());
    let mut val5 = _mm512_load_epi32(values.add(offset + (5 << log_step)).cast_const());
    let mut val6 = _mm512_load_epi32(values.add(offset + (6 << log_step)).cast_const());
    let mut val7 = _mm512_load_epi32(values.add(offset + (7 << log_step)).cast_const());

    // Apply the first layer of butterflies.
    (val0, val1) = avx_ibutterfly(val0, val1, _mm512_set1_epi32(twiddles_dbl0[0]));
    (val2, val3) = avx_ibutterfly(val2, val3, _mm512_set1_epi32(twiddles_dbl0[1]));
    (val4, val5) = avx_ibutterfly(val4, val5, _mm512_set1_epi32(twiddles_dbl0[2]));
    (val6, val7) = avx_ibutterfly(val6, val7, _mm512_set1_epi32(twiddles_dbl0[3]));

    // Apply the second layer of butterflies.
    (val0, val2) = avx_ibutterfly(val0, val2, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val1, val3) = avx_ibutterfly(val1, val3, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val4, val6) = avx_ibutterfly(val4, val6, _mm512_set1_epi32(twiddles_dbl1[1]));
    (val5, val7) = avx_ibutterfly(val5, val7, _mm512_set1_epi32(twiddles_dbl1[1]));

    // Apply the third layer of butterflies.
    (val0, val4) = avx_ibutterfly(val0, val4, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val1, val5) = avx_ibutterfly(val1, val5, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val2, val6) = avx_ibutterfly(val2, val6, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val3, val7) = avx_ibutterfly(val3, val7, _mm512_set1_epi32(twiddles_dbl2[0]));

    // Store the 8 AVX vectors back to the array.
    _mm512_store_epi32(values.add(offset + (0 << log_step)), val0);
    _mm512_store_epi32(values.add(offset + (1 << log_step)), val1);
    _mm512_store_epi32(values.add(offset + (2 << log_step)), val2);
    _mm512_store_epi32(values.add(offset + (3 << log_step)), val3);
    _mm512_store_epi32(values.add(offset + (4 << log_step)), val4);
    _mm512_store_epi32(values.add(offset + (5 << log_step)), val5);
    _mm512_store_epi32(values.add(offset + (6 << log_step)), val6);
    _mm512_store_epi32(values.add(offset + (7 << log_step)), val7);
}

// TODO(spapini): Move these to M31 AVX.

/// Adds two packed M31 elements, and reduces the result to the range [0,P].
/// Each value is assumed to be in unreduced form, [0, P] including P.
/// # Safety
/// This function is safe.
pub unsafe fn add_mod_p(a: __m512i, b: __m512i) -> __m512i {
    // Add word by word. Each word is in the range [0, 2P].
    let c = _mm512_add_epi32(a, b);
    // Apply min(c, c-P) to each word.
    // When c in [P,2P], then c-P in [0,P] which is always less than [P,2P].
    // When c in [0,P-1], then c-P in [2^32-P,2^32-1] which is always greater than [0,P-1].
    _mm512_min_epu32(c, _mm512_sub_epi32(c, P))
}

/// Subtracts two packed M31 elements, and reduces the result to the range [0,P].
/// Each value is assumed to be in unreduced form, [0, P] including P.
/// # Safety
/// This function is safe.
pub unsafe fn sub_mod_p(a: __m512i, b: __m512i) -> __m512i {
    // Subtract word by word. Each word is in the range [-P, P].
    let c = _mm512_sub_epi32(a, b);
    // Apply min(c, c+P) to each word.
    // When c in [0,P], then c+P in [P,2P] which is always greater than [0,P].
    // When c in [2^32-P,2^32-1], then c+P in [0,P-1] which is always less than [2^32-P,2^32-1].
    _mm512_min_epu32(_mm512_add_epi32(c, P), c)
}

#[cfg(test)]
mod tests {
    use std::arch::x86_64::_mm512_setr_epi32;

    use super::*;
    use crate::core::backend::avx512::m31::PackedBaseField;
    use crate::core::backend::avx512::BaseFieldVec;
    use crate::core::backend::CPUBackend;
    use crate::core::fft::{butterfly, ibutterfly};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::{Column, Field};
    use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation};
    use crate::core::utils::bit_reverse;

    #[test]
    fn test_butterfly() {
        unsafe {
            let val0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            let val1 = _mm512_setr_epi32(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            );
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
    fn test_ibutterfly() {
        unsafe {
            let val0 = _mm512_setr_epi32(2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            let val1 = _mm512_setr_epi32(
                3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            );
            let twiddle = _mm512_setr_epi32(
                1177558791, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            );
            let twiddle_dbl = _mm512_add_epi32(twiddle, twiddle);
            let (r0, r1) = avx_ibutterfly(val0, val1, twiddle_dbl);

            let val0: [BaseField; 16] = std::mem::transmute(val0);
            let val1: [BaseField; 16] = std::mem::transmute(val1);
            let twiddle: [BaseField; 16] = std::mem::transmute(twiddle);
            let r0: [BaseField; 16] = std::mem::transmute(r0);
            let r1: [BaseField; 16] = std::mem::transmute(r1);

            for i in 0..16 {
                let mut x = val0[i];
                let mut y = val1[i];
                let twiddle = twiddle[i];
                ibutterfly(&mut x, &mut y, twiddle);
                assert_eq!(x, r0[i]);
                assert_eq!(y, r1[i]);
            }
        }
    }

    #[test]
    fn test_vecwise_butterflies() {
        unsafe {
            let val0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            let val1 = _mm512_setr_epi32(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            );
            let twiddles0 = [
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            ];
            let twiddles1 = [48, 49, 50, 51, 52, 53, 54, 55];
            let twiddles2 = [56, 57, 58, 59];
            let twiddles3 = [60, 61];
            let twiddle0_dbl = std::array::from_fn(|i| twiddles0[i] * 2);
            let twiddle1_dbl = std::array::from_fn(|i| twiddles1[i] * 2);
            let twiddle2_dbl = std::array::from_fn(|i| twiddles2[i] * 2);
            let twiddle3_dbl = std::array::from_fn(|i| twiddles3[i] * 2);

            let (r0, r1) = vecwise_butterflies(
                val0,
                val1,
                twiddle0_dbl,
                twiddle1_dbl,
                twiddle2_dbl,
                twiddle3_dbl,
            );

            let mut val0: [BaseField; 16] = std::mem::transmute(val0);
            let mut val1: [BaseField; 16] = std::mem::transmute(val1);
            let r0: [BaseField; 16] = std::mem::transmute(r0);
            let r1: [BaseField; 16] = std::mem::transmute(r1);
            let twiddles0: [BaseField; 16] = std::mem::transmute(twiddles0);
            let twiddles1: [BaseField; 8] = std::mem::transmute(twiddles1);
            let twiddles2: [BaseField; 4] = std::mem::transmute(twiddles2);
            let twiddles3: [BaseField; 2] = std::mem::transmute(twiddles3);

            for i in 0..16 {
                let j = i ^ 8;
                if i > j {
                    continue;
                }
                let (mut v00, mut v01, mut v10, mut v11) = (val0[i], val0[j], val1[i], val1[j]);
                butterfly(&mut v00, &mut v01, twiddles3[0]);
                butterfly(&mut v10, &mut v11, twiddles3[1]);
                (val0[i], val0[j], val1[i], val1[j]) = (v00, v01, v10, v11);
            }
            for i in 0..16 {
                let j = i ^ 4;
                if i > j {
                    continue;
                }
                let (mut v00, mut v01, mut v10, mut v11) = (val0[i], val0[j], val1[i], val1[j]);
                butterfly(&mut v00, &mut v01, twiddles2[i / 8]);
                butterfly(&mut v10, &mut v11, twiddles2[2 + i / 8]);
                (val0[i], val0[j], val1[i], val1[j]) = (v00, v01, v10, v11);
            }
            for i in 0..16 {
                let j = i ^ 2;
                if i > j {
                    continue;
                }
                let (mut v00, mut v01, mut v10, mut v11) = (val0[i], val0[j], val1[i], val1[j]);
                butterfly(&mut v00, &mut v01, twiddles1[i / 4]);
                butterfly(&mut v10, &mut v11, twiddles1[4 + i / 4]);
                (val0[i], val0[j], val1[i], val1[j]) = (v00, v01, v10, v11);
            }
            for i in 0..16 {
                let j = i ^ 1;
                if i > j {
                    continue;
                }
                let (mut v00, mut v01, mut v10, mut v11) = (val0[i], val0[j], val1[i], val1[j]);
                butterfly(&mut v00, &mut v01, twiddles0[i / 2]);
                butterfly(&mut v10, &mut v11, twiddles0[8 + i / 2]);
                (val0[i], val0[j], val1[i], val1[j]) = (v00, v01, v10, v11);
            }

            // Compare
            for i in 0..16 {
                assert_eq!(val0[i], r0[i]);
                assert_eq!(val1[i], r1[i]);
            }
        }
    }

    #[test]
    fn test_vecwise_ibutterflies() {
        unsafe {
            let val0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            let val1 = _mm512_setr_epi32(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            );
            let twiddles0 = [
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            ];
            let twiddles1 = [48, 49, 50, 51, 52, 53, 54, 55];
            let twiddles2 = [56, 57, 58, 59];
            let twiddles3 = [60, 61];
            let twiddle0_dbl = std::array::from_fn(|i| twiddles0[i] * 2);
            let twiddle1_dbl = std::array::from_fn(|i| twiddles1[i] * 2);
            let twiddle2_dbl = std::array::from_fn(|i| twiddles2[i] * 2);
            let twiddle3_dbl = std::array::from_fn(|i| twiddles3[i] * 2);

            let (r0, r1) = vecwise_ibutterflies(
                val0,
                val1,
                twiddle0_dbl,
                twiddle1_dbl,
                twiddle2_dbl,
                twiddle3_dbl,
            );

            let mut val0: [BaseField; 16] = std::mem::transmute(val0);
            let mut val1: [BaseField; 16] = std::mem::transmute(val1);
            let r0: [BaseField; 16] = std::mem::transmute(r0);
            let r1: [BaseField; 16] = std::mem::transmute(r1);
            let twiddles0: [BaseField; 16] = std::mem::transmute(twiddles0);
            let twiddles1: [BaseField; 8] = std::mem::transmute(twiddles1);
            let twiddles2: [BaseField; 4] = std::mem::transmute(twiddles2);
            let twiddles3: [BaseField; 2] = std::mem::transmute(twiddles3);

            for i in 0..16 {
                let j = i ^ 1;
                if i > j {
                    continue;
                }
                let (mut v00, mut v01, mut v10, mut v11) = (val0[i], val0[j], val1[i], val1[j]);
                ibutterfly(&mut v00, &mut v01, twiddles0[i / 2]);
                ibutterfly(&mut v10, &mut v11, twiddles0[8 + i / 2]);
                (val0[i], val0[j], val1[i], val1[j]) = (v00, v01, v10, v11);
            }
            for i in 0..16 {
                let j = i ^ 2;
                if i > j {
                    continue;
                }
                let (mut v00, mut v01, mut v10, mut v11) = (val0[i], val0[j], val1[i], val1[j]);
                ibutterfly(&mut v00, &mut v01, twiddles1[i / 4]);
                ibutterfly(&mut v10, &mut v11, twiddles1[4 + i / 4]);
                (val0[i], val0[j], val1[i], val1[j]) = (v00, v01, v10, v11);
            }
            for i in 0..16 {
                let j = i ^ 4;
                if i > j {
                    continue;
                }
                let (mut v00, mut v01, mut v10, mut v11) = (val0[i], val0[j], val1[i], val1[j]);
                ibutterfly(&mut v00, &mut v01, twiddles2[i / 8]);
                ibutterfly(&mut v10, &mut v11, twiddles2[2 + i / 8]);
                (val0[i], val0[j], val1[i], val1[j]) = (v00, v01, v10, v11);
            }
            for i in 0..16 {
                let j = i ^ 8;
                if i > j {
                    continue;
                }
                let (mut v00, mut v01, mut v10, mut v11) = (val0[i], val0[j], val1[i], val1[j]);
                ibutterfly(&mut v00, &mut v01, twiddles3[0]);
                ibutterfly(&mut v10, &mut v11, twiddles3[1]);
                (val0[i], val0[j], val1[i], val1[j]) = (v00, v01, v10, v11);
            }
            // Compare
            for i in 0..16 {
                assert_eq!(val0[i], r0[i]);
                assert_eq!(val1[i], r1[i]);
            }
        }
    }

    #[test]
    fn test_ifft3() {
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
            ifft3(
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
                let j = i ^ 1;
                if i > j {
                    continue;
                }
                let (mut v0, mut v1) = (expected[i], expected[j]);
                ibutterfly(&mut v0, &mut v1, twiddles0[i / 2]);
                (expected[i], expected[j]) = (v0, v1);
            }
            for i in 0..8 {
                let j = i ^ 2;
                if i > j {
                    continue;
                }
                let (mut v0, mut v1) = (expected[i], expected[j]);
                ibutterfly(&mut v0, &mut v1, twiddles1[i / 4]);
                (expected[i], expected[j]) = (v0, v1);
            }
            for i in 0..8 {
                let j = i ^ 4;
                if i > j {
                    continue;
                }
                let (mut v0, mut v1) = (expected[i], expected[j]);
                ibutterfly(&mut v0, &mut v1, twiddles2[0]);
                (expected[i], expected[j]) = (v0, v1);
            }
            for i in 0..8 {
                assert_eq!(values[i].to_array()[0], expected[i]);
            }
        }
    }

    fn get_itwiddle_dbls(domain: CircleDomain) -> Vec<Vec<i32>> {
        let mut coset = domain.half_coset;

        let mut res = vec![];
        res.push(
            coset
                .iter()
                .map(|p| (p.y.inverse().0 * 2) as i32)
                .collect::<Vec<_>>(),
        );
        bit_reverse(res.last_mut().unwrap());
        for _ in 0..coset.log_size() {
            res.push(
                coset
                    .iter()
                    .take(coset.size() / 2)
                    .map(|p| (p.x.inverse().0 * 2) as i32)
                    .collect::<Vec<_>>(),
            );
            bit_reverse(res.last_mut().unwrap());
            coset = coset.double();
        }

        res
    }

    fn ref_ifft(domain: CircleDomain, mut values: Vec<BaseField>) -> Vec<BaseField> {
        bit_reverse(&mut values);
        let eval = CircleEvaluation::<CPUBackend, _>::new(domain, values);
        let mut expected_coeffs = eval.interpolate().coeffs;
        for x in expected_coeffs.iter_mut() {
            *x *= BaseField::from_u32_unchecked(domain.size() as u32);
        }
        bit_reverse(&mut expected_coeffs);
        expected_coeffs
    }

    #[test]
    fn test_vecwise_ibutterflies_real() {
        let domain = CanonicCoset::new(5).circle_domain();
        let twiddle_dbls = get_itwiddle_dbls(domain);
        assert_eq!(twiddle_dbls.len(), 5);
        let values0: [i32; 16] = std::array::from_fn(|i| i as i32);
        let values1: [i32; 16] = std::array::from_fn(|i| (i + 16) as i32);
        let result: [BaseField; 32] = unsafe {
            let (val0, val1) = vecwise_ibutterflies(
                std::mem::transmute(values0),
                std::mem::transmute(values1),
                twiddle_dbls[0].clone().try_into().unwrap(),
                twiddle_dbls[1].clone().try_into().unwrap(),
                twiddle_dbls[2].clone().try_into().unwrap(),
                twiddle_dbls[3].clone().try_into().unwrap(),
            );
            let (val0, val1) = avx_ibutterfly(val0, val1, _mm512_set1_epi32(twiddle_dbls[4][0]));
            std::mem::transmute([val0, val1])
        };

        // ref.
        let mut values = values0.to_vec();
        values.extend_from_slice(&values1);
        let expected = ref_ifft(domain, values.into_iter().map(BaseField::from).collect());

        // Compare.
        for i in 0..32 {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_ifft_lower_with_vecwise() {
        let log_size = 5 + 3 + 3;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = (0..domain.size())
            .map(|i| BaseField::from_u32_unchecked(i as u32))
            .collect::<Vec<_>>();
        let expected_coeffs = ref_ifft(domain, values.clone());

        // Compute.
        let mut values = BaseFieldVec::from_iter(values);
        let twiddle_dbls = get_itwiddle_dbls(domain);

        unsafe {
            ifft_lower_with_vecwise(
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddle_dbls,
                log_size as usize,
                log_size as usize,
            );

            // Compare.
            assert_eq!(values.to_vec(), expected_coeffs);
        }
    }
}
