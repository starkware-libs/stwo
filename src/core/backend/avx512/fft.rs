use std::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_min_epu32, _mm512_mul_epi32, _mm512_permutex2var_epi32,
    _mm512_srli_epi64, _mm512_sub_epi32,
};

const L: __m512i = unsafe {
    core::mem::transmute([
        0b00000, 0b10000, 0b00010, 0b10010, 0b00100, 0b10100, 0b00110, 0b10110, 0b01000, 0b11000,
        0b01010, 0b11010, 0b01100, 0b11100, 0b01110, 0b11110,
    ])
};
const H: __m512i = unsafe {
    core::mem::transmute([
        0b00001, 0b10001, 0b00011, 0b10011, 0b00101, 0b10101, 0b00111, 0b10111, 0b01001, 0b11001,
        0b01011, 0b11011, 0b01101, 0b11101, 0b01111, 0b11111,
    ])
};
const P: __m512i = unsafe { core::mem::transmute([(1u32 << 31) - 1; 16]) };

/// # Safety
pub unsafe fn avx_butterfly(
    val0: __m512i,
    val1: __m512i,
    twiddle_dbl: __m512i,
) -> (__m512i, __m512i) {
    let val1_e = val1;
    let val1_o = _mm512_srli_epi64(val1, 32);
    let m_e_dbl = _mm512_mul_epi32(val1_e, twiddle_dbl);
    let m_o_dbl = _mm512_mul_epi32(val1_o, _mm512_srli_epi64(twiddle_dbl, 32));

    let rm_l = _mm512_srli_epi64(_mm512_permutex2var_epi32(m_e_dbl, L, m_o_dbl), 1);
    let rm_h = _mm512_permutex2var_epi32(m_e_dbl, H, m_o_dbl);

    let rm = _mm512_add_epi32(rm_l, rm_h);
    let rm_m_p = _mm512_sub_epi32(rm, P);
    let rrm = _mm512_min_epu32(rm, rm_m_p);

    let a0 = _mm512_add_epi32(val0, rrm);
    let a0_m_p = _mm512_sub_epi32(a0, P);
    let r0 = _mm512_min_epu32(a0, a0_m_p);

    let a1 = _mm512_sub_epi32(val0, rrm);
    let a1_p_p = _mm512_add_epi32(a1, P);
    let r1 = _mm512_min_epu32(a1_p_p, a1);

    (r0, r1)
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
