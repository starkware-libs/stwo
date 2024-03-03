use std::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_min_epu32, _mm512_mul_epu32, _mm512_permutex2var_epi32,
    _mm512_srli_epi64, _mm512_sub_epi32,
};

/// L is an input to _mm512_permutex2var_epi32, and is used to interleave the even words of a
/// with the even words of b.
const L: __m512i = unsafe {
    core::mem::transmute([
        0b00000, 0b10000, 0b00010, 0b10010, 0b00100, 0b10100, 0b00110, 0b10110, 0b01000, 0b11000,
        0b01010, 0b11010, 0b01100, 0b11100, 0b01110, 0b11110,
    ])
};
/// H is an input to _mm512_permutex2var_epi32, and is used to interleave the odd words of a
/// with the odd words of b.
const H: __m512i = unsafe {
    core::mem::transmute([
        0b00001, 0b10001, 0b00011, 0b10011, 0b00101, 0b10101, 0b00111, 0b10111, 0b01001, 0b11001,
        0b01011, 0b11011, 0b01101, 0b11101, 0b01111, 0b11111,
    ])
};
const P: __m512i = unsafe { core::mem::transmute([(1u32 << 31) - 1; 16]) };

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

    // To compute prod = val1 * twiddle start by multipling
    // val1_e/o by twiddle_dbl_e/o.
    let prod_e_dbl = _mm512_mul_epu32(val1_e, twiddle_dbl_e);
    let prod_o_dbl = _mm512_mul_epu32(val1_o, twiddle_dbl_o);

    // The result of a multiplication holds val1*twiddle_dbl in as 64-bits.
    // Each 64b-bit word looks like this:
    //               1    31       31    1
    // prod_e_dbl - |0|prod_e_h|prod_e_l|0|
    // prod_o_dbl - |0|prod_o_h|prod_o_l|0|

    // Interleave the even words of prod_e_dbl with the even words of prod_o_dbl:
    let prod_ls = _mm512_permutex2var_epi32(prod_e_dbl, L, prod_o_dbl);
    // prod_ls -    |prod_o_l|0|prod_e_l|0|

    // Divide by 2:
    let prod_ls = _mm512_srli_epi64(prod_ls, 1);
    // prod_ls -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
    let prod_hs = _mm512_permutex2var_epi32(prod_e_dbl, H, prod_o_dbl);
    // prod_hs -    |0|prod_o_h|0|prod_e_h|

    let prod = add_mod_p(prod_ls, prod_hs);

    let r0 = add_mod_p(val0, prod);
    let r1 = sub_mod_p(val0, prod);

    (r0, r1)
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
    use crate::core::fft::butterfly;
    use crate::core::fields::m31::BaseField;

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

            let val0: [u32; 16] = std::mem::transmute(val0);
            let val1: [u32; 16] = std::mem::transmute(val1);
            let twiddle: [u32; 16] = std::mem::transmute(twiddle);
            let r0: [u32; 16] = std::mem::transmute(r0);
            let r1: [u32; 16] = std::mem::transmute(r1);

            for i in 0..16 {
                let mut x = BaseField::from_u32_unchecked(val0[i]);
                let mut y = BaseField::from_u32_unchecked(val1[i]);
                let twiddle = BaseField::from_u32_unchecked(twiddle[i]);
                butterfly(&mut x, &mut y, twiddle);
                assert_eq!(x, BaseField::from_u32_unchecked(r0[i]));
                assert_eq!(y, BaseField::from_u32_unchecked(r1[i]));
            }
        }
    }
}
