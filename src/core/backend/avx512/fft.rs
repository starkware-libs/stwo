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
    let prod_ls = _mm512_permutex2var_epi32(prod_e_dbl, L, prod_o_dbl);
    // prod_ls -    |prod_o_l|0|prod_e_l|0|

    // Divide by 2:
    let prod_ls = _mm512_srli_epi64(prod_ls, 1);
    // prod_ls -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
    let prod_hs = _mm512_permutex2var_epi32(prod_e_dbl, H, prod_o_dbl);
    // prod_hs -    |0|prod_o_h|0|prod_e_h|

    // Add prod_ls and prod_hs mod P.
    let unreduced_prod = _mm512_add_epi32(prod_ls, prod_hs);
    // unreduced_prod -     |prod_o_l+prod_o_h|prod_e_l+prod_e_h|
    let unreduced_prod_m_p = _mm512_sub_epi32(unreduced_prod, P);
    let prod = _mm512_min_epu32(unreduced_prod, unreduced_prod_m_p);

    // Add val0 + prod mod P.
    let a0 = _mm512_add_epi32(val0, prod);
    let a0_m_p = _mm512_sub_epi32(a0, P);
    let r0 = _mm512_min_epu32(a0, a0_m_p);

    // Subtract val0 - prod mod P.
    let a1 = _mm512_sub_epi32(val0, prod);
    let a1_p_p = _mm512_add_epi32(a1, P);
    let r1 = _mm512_min_epu32(a1_p_p, a1);

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
    // Add val0 + val1 mod P.
    let a0 = _mm512_add_epi32(val0, val1);
    let a0_m_p = _mm512_sub_epi32(a0, P);
    let r0 = _mm512_min_epu32(a0, a0_m_p);

    // Subtract val0 - val1 mod P.
    let a1 = _mm512_sub_epi32(val0, val1);
    let a1_p_p = _mm512_add_epi32(a1, P);
    let r1 = _mm512_min_epu32(a1_p_p, a1);

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
    let prod_ls = _mm512_permutex2var_epi32(prod_e_dbl, L, prod_o_dbl);
    // prod_ls -    |prod_o_l|0|prod_e_l|0|

    // Divide by 2:
    let prod_ls = _mm512_srli_epi64(prod_ls, 1);
    // prod_ls -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
    let prod_hs = _mm512_permutex2var_epi32(prod_e_dbl, H, prod_o_dbl);
    // prod_hs -    |0|prod_o_h|0|prod_e_h|

    // Add prod_ls and prod_hs mod P.
    let unreduced_prod = _mm512_add_epi32(prod_ls, prod_hs);
    // unreduced_prod -     |prod_o_l+prod_o_h|prod_e_l+prod_e_h|
    let unreduced_prod_m_p = _mm512_sub_epi32(unreduced_prod, P);
    let prod = _mm512_min_epu32(unreduced_prod, unreduced_prod_m_p);

    (r0, prod)
}

#[cfg(test)]
mod tests {
    use std::arch::x86_64::_mm512_setr_epi32;

    use super::*;
    use crate::core::fft::{butterfly, ibutterfly};
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

    #[test]
    fn test_ibutterfly() {
        unsafe {
            let val0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            let val1 = _mm512_setr_epi32(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            );
            let twiddle = _mm512_setr_epi32(
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            );
            let twiddle_dbl = _mm512_add_epi32(twiddle, twiddle);
            let (r0, r1) = avx_ibutterfly(val0, val1, twiddle_dbl);

            let val0: [u32; 16] = std::mem::transmute(val0);
            let val1: [u32; 16] = std::mem::transmute(val1);
            let twiddle: [u32; 16] = std::mem::transmute(twiddle);
            let r0: [u32; 16] = std::mem::transmute(r0);
            let r1: [u32; 16] = std::mem::transmute(r1);

            for i in 0..16 {
                let mut x = BaseField::from_u32_unchecked(val0[i]);
                let mut y = BaseField::from_u32_unchecked(val1[i]);
                let twiddle = BaseField::from_u32_unchecked(twiddle[i]);
                ibutterfly(&mut x, &mut y, twiddle);
                assert_eq!(x, BaseField::from_u32_unchecked(r0[i]));
                assert_eq!(y, BaseField::from_u32_unchecked(r1[i]));
            }
        }
    }
}
