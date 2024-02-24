use std::arch::x86_64::{
    __m512i, _mm512_broadcast_i32x4, _mm512_broadcast_i64x4, _mm512_load_epi32, _mm512_mul_epu32,
    _mm512_permutex2var_epi32, _mm512_permutexvar_epi32, _mm512_set1_epi32, _mm512_set1_epi64,
    _mm512_srli_epi64, _mm512_store_epi32, _mm512_xor_epi32,
};

use super::{
    add_mod_p, sub_mod_p, transpose_vecs, CACHED_FFT_LOG_SIZE, EVENS_INTERLEAVE_EVENS,
    HHALF_INTERLEAVE_HHALF, LHALF_INTERLEAVE_LHALF, MIN_FFT_LOG_SIZE, ODDS_INTERLEAVE_ODDS,
};
use crate::core::backend::avx512::VECS_LOG_SIZE;
use crate::core::poly::circle::CircleDomain;
use crate::core::utils::bit_reverse;

/// # Safety
pub unsafe fn fft(values: *mut i32, twiddle_dbl: &[&[i32]], log_n_elements: usize) {
    assert!(log_n_elements >= MIN_FFT_LOG_SIZE);
    let log_n_vecs = log_n_elements - VECS_LOG_SIZE;
    if log_n_elements <= CACHED_FFT_LOG_SIZE {
        // 16 {
        fft_lower_with_vecwise(values, twiddle_dbl, log_n_vecs, log_n_vecs);
        return;
    }
    let log_n_fft_vecs0 = log_n_vecs / 2;
    let log_n_fft_vecs1 = (log_n_vecs + 1) / 2;
    fft_lower_without_vecwise(
        values,
        &twiddle_dbl[(3 + log_n_fft_vecs1)..],
        log_n_vecs,
        log_n_fft_vecs0,
    );
    transpose_vecs(values, log_n_vecs);
    fft_lower_with_vecwise(
        values,
        &twiddle_dbl[..(3 + log_n_fft_vecs1)],
        log_n_vecs,
        log_n_fft_vecs1,
    );
}

/// # Safety
pub unsafe fn fft_lower_with_vecwise(
    values: *mut i32,
    twiddle_dbl: &[&[i32]],
    log_n_vecs: usize,
    fft_bits: usize,
) {
    assert!(fft_bits >= 1);
    assert_eq!(twiddle_dbl[0].len(), 1 << (log_n_vecs + 2));
    for h in (0..(1 << (log_n_vecs - fft_bits))).rev() {
        for bit_i in (1..fft_bits).step_by(3).rev() {
            match fft_bits - bit_i {
                1 => {
                    fft1_loop(values, &twiddle_dbl[3..], fft_bits, bit_i, h);
                }
                2 => {
                    fft2_loop(values, &twiddle_dbl[3..], fft_bits, bit_i, h);
                }
                _ => {
                    fft3_loop(values, &twiddle_dbl[3..], fft_bits, bit_i, h);
                }
            }
        }
        fft_vecwise_loop(values, twiddle_dbl, fft_bits, h);
    }
}

/// # Safety
pub unsafe fn fft_lower_without_vecwise(
    values: *mut i32,
    twiddle_dbl: &[&[i32]],
    log_n_vecs: usize,
    fft_bits: usize,
) {
    assert!(fft_bits >= 1);
    for h in 0..(1 << (log_n_vecs - fft_bits)) {
        for bit_i in (0..fft_bits).step_by(3).rev() {
            match fft_bits - bit_i {
                1 => {
                    fft1_loop(values, twiddle_dbl, fft_bits, bit_i, h);
                }
                2 => {
                    fft2_loop(values, twiddle_dbl, fft_bits, bit_i, h);
                }
                _ => {
                    fft3_loop(values, twiddle_dbl, fft_bits, bit_i, h);
                }
            }
        }
    }
}

/// # Safety
unsafe fn fft_vecwise_loop(values: *mut i32, twiddle_dbl: &[&[i32]], fft_bits: usize, h: usize) {
    for l in 0..(1 << (fft_bits - 1)) {
        let index = (h << (fft_bits - 1)) + l;
        let mut val0 = _mm512_load_epi32(values.add(index * 32).cast_const());
        let mut val1 = _mm512_load_epi32(values.add(index * 32 + 16).cast_const());
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
        _mm512_store_epi32(values.add(index * 32), val0);
        _mm512_store_epi32(values.add(index * 32 + 16), val1);
    }
}

/// # Safety
unsafe fn fft3_loop(
    values: *mut i32,
    twiddle_dbl: &[&[i32]],
    fft_bits: usize,
    bit_i: usize,
    index: usize,
) {
    for m in 0..(1 << (fft_bits - 3 - bit_i)) {
        let index = (index << (fft_bits - bit_i - 3)) + m;
        let offset = index << (bit_i + 3);
        for l in 0..(1 << bit_i) {
            fft3(
                values,
                offset + l,
                bit_i,
                std::array::from_fn(|i| {
                    *twiddle_dbl[bit_i]
                        .get_unchecked((index * 4 + i) & (twiddle_dbl[bit_i].len() - 1))
                }),
                std::array::from_fn(|i| {
                    *twiddle_dbl[bit_i + 1]
                        .get_unchecked((index * 2 + i) & (twiddle_dbl[bit_i + 1].len() - 1))
                }),
                std::array::from_fn(|i| {
                    *twiddle_dbl[bit_i + 2]
                        .get_unchecked((index + i) & (twiddle_dbl[bit_i + 2].len() - 1))
                }),
            );
        }
    }
}

/// # Safety
unsafe fn fft2_loop(
    values: *mut i32,
    twiddle_dbl: &[&[i32]],
    fft_bits: usize,
    bit_i: usize,
    index: usize,
) {
    for m in 0..(1 << (fft_bits - 2 - bit_i)) {
        let index = (index << (fft_bits - bit_i - 2)) + m;
        let offset = index << (bit_i + 2);
        for l in 0..(1 << bit_i) {
            fft2(
                values,
                offset + l,
                bit_i,
                std::array::from_fn(|i| {
                    *twiddle_dbl[bit_i]
                        .get_unchecked((index * 2 + i) & (twiddle_dbl[bit_i].len() - 1))
                }),
                std::array::from_fn(|i| {
                    *twiddle_dbl[bit_i + 1]
                        .get_unchecked((index + i) & (twiddle_dbl[bit_i + 1].len() - 1))
                }),
            );
        }
    }
}

/// # Safety
unsafe fn fft1_loop(
    values: *mut i32,
    twiddle_dbl: &[&[i32]],
    fft_bits: usize,
    bit_i: usize,
    index: usize,
) {
    for m in 0..(1 << (fft_bits - 1 - bit_i)) {
        let index = (index << (fft_bits - bit_i - 1)) + m;
        let offset = index << (bit_i + 1);
        for l in 0..(1 << bit_i) {
            fft1(
                values,
                offset + l,
                bit_i,
                std::array::from_fn(|i| {
                    *twiddle_dbl[bit_i].get_unchecked((index + i) & (twiddle_dbl[bit_i].len() - 1))
                }),
            );
        }
    }
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

    const INDICES_FROM_T1: __m512i = unsafe {
        core::mem::transmute([
            0b0001, 0b0001, 0b0000, 0b0000, 0b0011, 0b0011, 0b0010, 0b0010, 0b0101, 0b0101, 0b0100,
            0b0100, 0b0111, 0b0111, 0b0110, 0b0110,
        ])
    };
    const NEGATION_MASK: __m512i = unsafe {
        core::mem::transmute([0i32, -2, -2, 0, 0, -2, -2, 0, 0, -2, -2, 0, 0, -2, -2, 0])
    };
    let t = _mm512_permutexvar_epi32(INDICES_FROM_T1, t);
    let t = _mm512_xor_epi32(t, NEGATION_MASK);

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

pub fn get_twiddle_dbls(domain: CircleDomain) -> Vec<Vec<i32>> {
    let mut coset = domain.half_coset;

    let mut res = vec![];
    res.push(coset.iter().map(|p| (p.y.0 * 2) as i32).collect::<Vec<_>>());
    bit_reverse(res.last_mut().unwrap());
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

/// # Safety
pub unsafe fn fft3(
    values: *mut i32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [i32; 4],
    twiddles_dbl1: [i32; 2],
    twiddles_dbl2: [i32; 1],
) {
    let log_u32_step = log_step;
    // load
    let mut val0 = _mm512_load_epi32(values.add((offset + (0 << log_u32_step)) << 4).cast_const());
    let mut val1 = _mm512_load_epi32(values.add((offset + (1 << log_u32_step)) << 4).cast_const());
    let mut val2 = _mm512_load_epi32(values.add((offset + (2 << log_u32_step)) << 4).cast_const());
    let mut val3 = _mm512_load_epi32(values.add((offset + (3 << log_u32_step)) << 4).cast_const());
    let mut val4 = _mm512_load_epi32(values.add((offset + (4 << log_u32_step)) << 4).cast_const());
    let mut val5 = _mm512_load_epi32(values.add((offset + (5 << log_u32_step)) << 4).cast_const());
    let mut val6 = _mm512_load_epi32(values.add((offset + (6 << log_u32_step)) << 4).cast_const());
    let mut val7 = _mm512_load_epi32(values.add((offset + (7 << log_u32_step)) << 4).cast_const());

    (val0, val4) = avx_butterfly(val0, val4, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val1, val5) = avx_butterfly(val1, val5, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val2, val6) = avx_butterfly(val2, val6, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val3, val7) = avx_butterfly(val3, val7, _mm512_set1_epi32(twiddles_dbl2[0]));

    (val0, val2) = avx_butterfly(val0, val2, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val1, val3) = avx_butterfly(val1, val3, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val4, val6) = avx_butterfly(val4, val6, _mm512_set1_epi32(twiddles_dbl1[1]));
    (val5, val7) = avx_butterfly(val5, val7, _mm512_set1_epi32(twiddles_dbl1[1]));

    (val0, val1) = avx_butterfly(val0, val1, _mm512_set1_epi32(twiddles_dbl0[0]));
    (val2, val3) = avx_butterfly(val2, val3, _mm512_set1_epi32(twiddles_dbl0[1]));
    (val4, val5) = avx_butterfly(val4, val5, _mm512_set1_epi32(twiddles_dbl0[2]));
    (val6, val7) = avx_butterfly(val6, val7, _mm512_set1_epi32(twiddles_dbl0[3]));

    // store
    _mm512_store_epi32(values.add((offset + (0 << log_u32_step)) << 4), val0);
    _mm512_store_epi32(values.add((offset + (1 << log_u32_step)) << 4), val1);
    _mm512_store_epi32(values.add((offset + (2 << log_u32_step)) << 4), val2);
    _mm512_store_epi32(values.add((offset + (3 << log_u32_step)) << 4), val3);
    _mm512_store_epi32(values.add((offset + (4 << log_u32_step)) << 4), val4);
    _mm512_store_epi32(values.add((offset + (5 << log_u32_step)) << 4), val5);
    _mm512_store_epi32(values.add((offset + (6 << log_u32_step)) << 4), val6);
    _mm512_store_epi32(values.add((offset + (7 << log_u32_step)) << 4), val7);
}

/// # Safety
pub unsafe fn fft2(
    values: *mut i32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [i32; 2],
    twiddles_dbl1: [i32; 1],
) {
    let log_u32_step = log_step;
    // load
    let mut val0 = _mm512_load_epi32(values.add((offset + (0 << log_u32_step)) << 4).cast_const());
    let mut val1 = _mm512_load_epi32(values.add((offset + (1 << log_u32_step)) << 4).cast_const());
    let mut val2 = _mm512_load_epi32(values.add((offset + (2 << log_u32_step)) << 4).cast_const());
    let mut val3 = _mm512_load_epi32(values.add((offset + (3 << log_u32_step)) << 4).cast_const());

    (val0, val2) = avx_butterfly(val0, val2, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val1, val3) = avx_butterfly(val1, val3, _mm512_set1_epi32(twiddles_dbl1[0]));

    (val0, val1) = avx_butterfly(val0, val1, _mm512_set1_epi32(twiddles_dbl0[0]));
    (val2, val3) = avx_butterfly(val2, val3, _mm512_set1_epi32(twiddles_dbl0[1]));

    // store
    _mm512_store_epi32(values.add((offset + (0 << log_u32_step)) << 4), val0);
    _mm512_store_epi32(values.add((offset + (1 << log_u32_step)) << 4), val1);
    _mm512_store_epi32(values.add((offset + (2 << log_u32_step)) << 4), val2);
    _mm512_store_epi32(values.add((offset + (3 << log_u32_step)) << 4), val3);
}

/// # Safety
pub unsafe fn fft1(values: *mut i32, offset: usize, log_step: usize, twiddles_dbl0: [i32; 1]) {
    let log_u32_step = log_step;
    // load
    let mut val0 = _mm512_load_epi32(values.add((offset + (0 << log_u32_step)) << 4).cast_const());
    let mut val1 = _mm512_load_epi32(values.add((offset + (1 << log_u32_step)) << 4).cast_const());

    (val0, val1) = avx_butterfly(val0, val1, _mm512_set1_epi32(twiddles_dbl0[0]));

    // store
    _mm512_store_epi32(values.add((offset + (0 << log_u32_step)) << 4), val0);
    _mm512_store_epi32(values.add((offset + (1 << log_u32_step)) << 4), val1);
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

#[cfg(test)]
mod tests {
    use std::arch::x86_64::{_mm512_add_epi32, _mm512_setr_epi32};

    use super::*;
    use crate::core::backend::avx512::BaseFieldVec;
    use crate::core::backend::CPUBackend;
    use crate::core::fft::butterfly;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::Column;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain, CirclePoly};

    #[test]
    fn test_butterfly() {
        unsafe {
            let val0 = _mm512_setr_epi32(2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            let val1 = _mm512_setr_epi32(
                3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            );
            let twiddle = _mm512_setr_epi32(
                1177558791, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
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
            let mut values: Vec<[i32; 16]> = (0..8).map(|i| std::array::from_fn(|_| i)).collect();
            let twiddles0 = [32, 33, 34, 35];
            let twiddles1 = [36, 37];
            let twiddles2 = [38];
            let twiddles0_dbl = std::array::from_fn(|i| twiddles0[i] * 2);
            let twiddles1_dbl = std::array::from_fn(|i| twiddles1[i] * 2);
            let twiddles2_dbl = std::array::from_fn(|i| twiddles2[i] * 2);
            fft3(
                std::mem::transmute(values.as_mut_ptr()),
                0,
                0,
                twiddles0_dbl,
                twiddles1_dbl,
                twiddles2_dbl,
            );

            let actual: Vec<[BaseField; 16]> = std::mem::transmute(values);
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
                assert_eq!(actual[i][0], expected[i]);
            }
        }
    }

    fn ref_fft(domain: CircleDomain, values: Vec<BaseField>) -> Vec<BaseField> {
        let eval = CirclePoly::<CPUBackend, _>::new(values);
        eval.evaluate(domain).values
    }

    #[test]
    fn test_vecwise_butterflies_real() {
        let domain = CanonicCoset::new(5).circle_domain();
        let twiddle_dbls = get_twiddle_dbls(domain);
        assert_eq!(twiddle_dbls.len(), 5);
        let values0: [i32; 16] = std::array::from_fn(|i| i as i32);
        let values1: [i32; 16] = std::array::from_fn(|i| (i + 16) as i32);
        let result: [BaseField; 32] = unsafe {
            let val0 = std::mem::transmute(values0);
            let val1 = std::mem::transmute(values1);

            let (val0, val1) = avx_butterfly(val0, val1, _mm512_set1_epi32(twiddle_dbls[4][0]));
            let (val0, val1) = vecwise_butterflies(
                val0,
                val1,
                twiddle_dbls[1].clone().try_into().unwrap(),
                twiddle_dbls[2].clone().try_into().unwrap(),
                twiddle_dbls[3].clone().try_into().unwrap(),
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
        for log_size in 5..=10 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let values = (0..domain.size())
                .map(|i| BaseField::from_u32_unchecked(i as u32))
                .collect::<Vec<_>>();
            let expected_coeffs = ref_fft(domain, values.clone());

            // Compute.
            let mut values = BaseFieldVec::from_iter(values);
            let twiddle_dbls = get_twiddle_dbls(domain);

            unsafe {
                fft_lower_with_vecwise(
                    std::mem::transmute(values.data.as_mut_ptr()),
                    &twiddle_dbls[1..]
                        .iter()
                        .map(|x| x.as_ref())
                        .collect::<Vec<_>>(),
                    (log_size) as usize - VECS_LOG_SIZE,
                    (log_size) as usize - VECS_LOG_SIZE,
                );

                // Compare.
                assert_eq!(values.to_vec(), expected_coeffs);
            }
        }
    }

    #[test]
    fn test_transposed_fft() {
        for log_size in (CACHED_FFT_LOG_SIZE + 1)..(CACHED_FFT_LOG_SIZE + 4) {
            let domain = CanonicCoset::new(log_size as u32).circle_domain();
            let values = (0..domain.size())
                .map(|i| BaseField::from_u32_unchecked(i as u32))
                .collect::<Vec<_>>();
            let expected_coeffs = ref_fft(domain, values.clone());

            // Compute.
            let mut values = BaseFieldVec::from_iter(values);
            let twiddle_dbls = get_twiddle_dbls(domain);

            unsafe {
                transpose_vecs(
                    std::mem::transmute(values.data.as_mut_ptr()),
                    log_size - VECS_LOG_SIZE,
                );
                fft(
                    std::mem::transmute(values.data.as_mut_ptr()),
                    &twiddle_dbls[1..]
                        .iter()
                        .map(|x| x.as_ref())
                        .collect::<Vec<_>>(),
                    log_size,
                );
            }

            // Compare.
            assert_eq!(values.to_vec(), expected_coeffs);
        }
    }
}
