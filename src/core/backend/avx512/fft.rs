use std::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_broadcast_i32x4, _mm512_broadcast_i64x4, _mm512_load_epi32,
    _mm512_min_epu32, _mm512_mul_epu32, _mm512_permutex2var_epi32, _mm512_permutexvar_epi32,
    _mm512_set1_epi32, _mm512_set1_epi64, _mm512_srli_epi64, _mm512_store_epi32, _mm512_sub_epi32,
    _mm512_xor_epi32,
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

const L1: __m512i = unsafe {
    core::mem::transmute([
        0b00000, 0b00010, 0b00100, 0b00110, 0b01000, 0b01010, 0b01100, 0b01110, 0b10000, 0b10010,
        0b10100, 0b10110, 0b11000, 0b11010, 0b11100, 0b11110,
    ])
};
const H1: __m512i = unsafe {
    core::mem::transmute([
        0b00001, 0b00011, 0b00101, 0b00111, 0b01001, 0b01011, 0b01101, 0b01111, 0b10001, 0b10011,
        0b10101, 0b10111, 0b11001, 0b11011, 0b11101, 0b11111,
    ])
};

const L2: __m512i = unsafe {
    core::mem::transmute([
        0b00000, 0b10000, 0b00001, 0b10001, 0b00010, 0b10010, 0b00011, 0b10011, 0b00100, 0b10100,
        0b00101, 0b10101, 0b00110, 0b10110, 0b00111, 0b10111,
    ])
};
const H2: __m512i = unsafe {
    core::mem::transmute([
        0b01000, 0b11000, 0b01001, 0b11001, 0b01010, 0b11010, 0b01011, 0b11011, 0b01100, 0b11100,
        0b01101, 0b11101, 0b01110, 0b11110, 0b01111, 0b11111,
    ])
};
const P: __m512i = unsafe { core::mem::transmute([(1u32 << 31) - 1; 16]) };

// TODO(spapini): FFTs return a redundant representation, that can get the value P. need to reduce
// it somewhere.

/// # Safety
pub unsafe fn ifft(values: *mut i32, twiddle_dbl: &[Vec<i32>], log_n_elements: usize) {
    assert!(log_n_elements >= 4);
    if log_n_elements <= 1 {
        // 16 {
        ifft_lower(
            values,
            Some(&twiddle_dbl[..3]),
            &twiddle_dbl[3..],
            log_n_elements - 4,
            log_n_elements - 4,
        );
        return;
    }
    let log_n_vecs = log_n_elements - 4;
    let log_n_fft_vecs0 = log_n_vecs / 2;
    let log_n_fft_vecs1 = (log_n_vecs + 1) / 2;
    ifft_lower(
        values,
        Some(&twiddle_dbl[..3]),
        &twiddle_dbl[3..(3 + log_n_fft_vecs1)],
        log_n_elements - 4,
        log_n_fft_vecs1,
    );
    // TODO(spapini): better transpose.
    transpose_vecs(values, log_n_elements - 4);
    ifft_lower(
        values,
        None,
        &twiddle_dbl[(3 + log_n_fft_vecs1)..],
        log_n_elements - 4,
        log_n_fft_vecs0,
    );
}

// TODO(spapini): This is inefficient. Optimize.
/// # Safety
pub unsafe fn transpose_vecs(values: *mut i32, log_n_vecs: usize) {
    let half = log_n_vecs / 2;
    for b in 0..(1 << (log_n_vecs & 1)) {
        for a in 0..(1 << half) {
            for c in 0..(1 << half) {
                let i = (a << (log_n_vecs - half)) | (b << half) | c;
                let j = (c << (log_n_vecs - half)) | (b << half) | a;
                if i >= j {
                    continue;
                }
                let val0 = _mm512_load_epi32(values.add(i << 4).cast_const());
                let val1 = _mm512_load_epi32(values.add(j << 4).cast_const());
                _mm512_store_epi32(values.add(i << 4), val1);
                _mm512_store_epi32(values.add(j << 4), val0);
            }
        }
    }
}

/// # Safety
pub unsafe fn ifft_lower(
    values: *mut i32,
    vecwise_twiddle_dbl: Option<&[Vec<i32>]>,
    twiddle_dbl: &[Vec<i32>],
    log_n_vecs: usize,
    fft_bits: usize,
) {
    assert!(fft_bits >= 1);
    if let Some(vecwise_twiddle_dbl) = vecwise_twiddle_dbl {
        assert_eq!(vecwise_twiddle_dbl[0].len(), 1 << (log_n_vecs + 2));
        assert_eq!(vecwise_twiddle_dbl[1].len(), 1 << (log_n_vecs + 1));
        assert_eq!(vecwise_twiddle_dbl[2].len(), 1 << log_n_vecs);
    }
    for h in 0..(1 << (log_n_vecs - fft_bits)) {
        // TODO(spapini):
        if let Some(vecwise_twiddle_dbl) = vecwise_twiddle_dbl {
            for l in 0..(1 << (fft_bits - 1)) {
                // TODO(spapini): modulo for twiddles on the iters.
                let index = (h << (fft_bits - 1)) + l;
                let mut val0 = _mm512_load_epi32(values.add(index * 32).cast_const());
                let mut val1 = _mm512_load_epi32(values.add(index * 32 + 16).cast_const());
                (val0, val1) = vecwise_ibutterflies(
                    val0,
                    val1,
                    std::array::from_fn(|i| *vecwise_twiddle_dbl[0].get_unchecked(index * 8 + i)),
                    std::array::from_fn(|i| *vecwise_twiddle_dbl[1].get_unchecked(index * 4 + i)),
                    std::array::from_fn(|i| *vecwise_twiddle_dbl[2].get_unchecked(index * 2 + i)),
                );
                _mm512_store_epi32(values.add(index * 32), val0);
                _mm512_store_epi32(values.add(index * 32 + 16), val1);
                // TODO(spapini): do a fifth layer here.
            }
        }
        for bit_i in (0..fft_bits).step_by(3) {
            if bit_i + 3 > fft_bits {
                todo!();
            }
            for m in 0..(1 << (fft_bits - 3 - bit_i)) {
                let twid_index = (h << (fft_bits - 3 - bit_i)) + m;
                for l in 0..(1 << bit_i) {
                    ifft3(
                        values,
                        (h << fft_bits) + (m << (bit_i + 3)) + l,
                        bit_i,
                        std::array::from_fn(|i| {
                            *twiddle_dbl[bit_i].get_unchecked(
                                (twid_index * 4 + i) & (twiddle_dbl[bit_i].len() - 1),
                            )
                        }),
                        std::array::from_fn(|i| {
                            *twiddle_dbl[bit_i + 1].get_unchecked(
                                (twid_index * 2 + i) & (twiddle_dbl[bit_i + 1].len() - 1),
                            )
                        }),
                        std::array::from_fn(|i| {
                            *twiddle_dbl[bit_i + 2].get_unchecked(
                                (twid_index + i) & (twiddle_dbl[bit_i + 2].len() - 1),
                            )
                        }),
                    );
                }
            }
        }
    }
}

/// # Safety
pub unsafe fn avx_butterfly(
    val0: __m512i,
    val1: __m512i,
    twiddle_dbl: __m512i,
) -> (__m512i, __m512i) {
    let val1_e = val1;
    let twiddle_dbl_e = twiddle_dbl;
    let val1_o = _mm512_srli_epi64(val1, 32);
    let twiddle_dbl_o = _mm512_srli_epi64(twiddle_dbl, 32);
    let m_e_dbl = _mm512_mul_epu32(val1_e, twiddle_dbl_e);
    let m_o_dbl = _mm512_mul_epu32(val1_o, twiddle_dbl_o);

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

/// # Safety
pub unsafe fn avx_ibutterfly(
    val0: __m512i,
    val1: __m512i,
    twiddle_dbl: __m512i,
) -> (__m512i, __m512i) {
    let a0 = _mm512_add_epi32(val0, val1);
    let a0_m_p = _mm512_sub_epi32(a0, P);
    let r0 = _mm512_min_epu32(a0, a0_m_p);

    let a1 = _mm512_sub_epi32(val0, val1);
    let a1_p_p = _mm512_add_epi32(a1, P);
    let r1 = _mm512_min_epu32(a1_p_p, a1);

    // mul
    let r1_e = r1;
    let twiddle_dbl_e = twiddle_dbl;
    let r1_o = _mm512_srli_epi64(r1, 32);
    let twiddle_dbl_o = _mm512_srli_epi64(twiddle_dbl, 32);
    let m_e_dbl = _mm512_mul_epu32(r1_e, twiddle_dbl_e);
    let m_o_dbl = _mm512_mul_epu32(r1_o, twiddle_dbl_o);

    let rm_l = _mm512_srli_epi64(_mm512_permutex2var_epi32(m_e_dbl, L, m_o_dbl), 1);
    let rm_h = _mm512_permutex2var_epi32(m_e_dbl, H, m_o_dbl);

    let rm = _mm512_add_epi32(rm_l, rm_h);
    let rm_m_p = _mm512_sub_epi32(rm, P);
    let rrm = _mm512_min_epu32(rm, rm_m_p);

    (r0, rrm)
}

/// # Safety
pub unsafe fn vecwise_butterflies(
    mut val0: __m512i,
    mut val1: __m512i,
    twiddle0_dbl: [i32; 16],
    twiddle1_dbl: [i32; 8],
    twiddle2_dbl: [i32; 4],
    twiddle3_dbl: [i32; 2],
) -> (__m512i, __m512i) {
    // TODO(spapini): The permute can be fused with the _mm512_srli_epi64 inside the butterfly.
    let t = _mm512_set1_epi64(std::mem::transmute(twiddle3_dbl));
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, L2, val1),
        _mm512_permutex2var_epi32(val0, H2, val1),
    );
    (val0, val1) = avx_butterfly(val0, val1, t);

    let t = _mm512_broadcast_i32x4(std::mem::transmute(twiddle2_dbl));
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, L2, val1),
        _mm512_permutex2var_epi32(val0, H2, val1),
    );
    (val0, val1) = avx_butterfly(val0, val1, t);

    let t = _mm512_broadcast_i64x4(std::mem::transmute(twiddle1_dbl));
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, L2, val1),
        _mm512_permutex2var_epi32(val0, H2, val1),
    );
    (val0, val1) = avx_butterfly(val0, val1, t);

    let t: __m512i = std::mem::transmute(twiddle0_dbl);
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, L2, val1),
        _mm512_permutex2var_epi32(val0, H2, val1),
    );
    (val0, val1) = avx_butterfly(val0, val1, t);

    (
        _mm512_permutex2var_epi32(val0, L2, val1),
        _mm512_permutex2var_epi32(val0, H2, val1),
    )
}

/// # Safety
pub unsafe fn vecwise_ibutterflies(
    mut val0: __m512i,
    mut val1: __m512i,
    twiddle1_dbl: [i32; 8],
    twiddle2_dbl: [i32; 4],
    twiddle3_dbl: [i32; 2],
) -> (__m512i, __m512i) {
    // TODO(spapini): The permute can be fused with the _mm512_srli_epi64 inside the butterfly.
    let t1 = _mm512_broadcast_i64x4(std::mem::transmute(twiddle1_dbl));
    const A: __m512i = unsafe {
        core::mem::transmute([
            0b0001, 0b0001, 0b0000, 0b0000, 0b0011, 0b0011, 0b0010, 0b0010, 0b0101, 0b0101, 0b0100,
            0b0100, 0b0111, 0b0111, 0b0110, 0b0110,
        ])
    };
    const M: __m512i = unsafe {
        core::mem::transmute([0i32, -2, -2, 0, 0, -2, -2, 0, 0, -2, -2, 0, 0, -2, -2, 0])
    };
    let t = _mm512_permutexvar_epi32(A, t1);
    let t = _mm512_xor_epi32(t, M);
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, L1, val1),
        _mm512_permutex2var_epi32(val0, H1, val1),
    );
    (val0, val1) = avx_ibutterfly(val0, val1, t);

    let t = t1;
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, L1, val1),
        _mm512_permutex2var_epi32(val0, H1, val1),
    );
    (val0, val1) = avx_ibutterfly(val0, val1, t);

    let t = _mm512_broadcast_i32x4(std::mem::transmute(twiddle2_dbl));
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, L1, val1),
        _mm512_permutex2var_epi32(val0, H1, val1),
    );
    (val0, val1) = avx_ibutterfly(val0, val1, t);

    let t = _mm512_set1_epi64(std::mem::transmute(twiddle3_dbl));
    (val0, val1) = (
        _mm512_permutex2var_epi32(val0, L1, val1),
        _mm512_permutex2var_epi32(val0, H1, val1),
    );
    (val0, val1) = avx_ibutterfly(val0, val1, t);

    (
        _mm512_permutex2var_epi32(val0, L1, val1),
        _mm512_permutex2var_epi32(val0, H1, val1),
    )
}

/// # Safety
pub unsafe fn ifft3(
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

    (val0, val1) = avx_ibutterfly(val0, val1, _mm512_set1_epi32(twiddles_dbl0[0]));
    (val2, val3) = avx_ibutterfly(val2, val3, _mm512_set1_epi32(twiddles_dbl0[1]));
    (val4, val5) = avx_ibutterfly(val4, val5, _mm512_set1_epi32(twiddles_dbl0[2]));
    (val6, val7) = avx_ibutterfly(val6, val7, _mm512_set1_epi32(twiddles_dbl0[3]));

    (val0, val2) = avx_ibutterfly(val0, val2, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val1, val3) = avx_ibutterfly(val1, val3, _mm512_set1_epi32(twiddles_dbl1[0]));
    (val4, val6) = avx_ibutterfly(val4, val6, _mm512_set1_epi32(twiddles_dbl1[1]));
    (val5, val7) = avx_ibutterfly(val5, val7, _mm512_set1_epi32(twiddles_dbl1[1]));

    (val0, val4) = avx_ibutterfly(val0, val4, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val1, val5) = avx_ibutterfly(val1, val5, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val2, val6) = avx_ibutterfly(val2, val6, _mm512_set1_epi32(twiddles_dbl2[0]));
    (val3, val7) = avx_ibutterfly(val3, val7, _mm512_set1_epi32(twiddles_dbl2[0]));

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

#[cfg(test)]
mod tests {
    use std::arch::x86_64::_mm512_setr_epi32;

    use super::*;
    use crate::core::backend::avx512::BaseFieldVec;
    use crate::core::backend::{CPUBackend, ColumnTrait};
    use crate::core::fft::{butterfly, ibutterfly};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::Field;
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
    fn test_ifft3() {
        unsafe {
            let mut values: Vec<[i32; 16]> = (0..8).map(|i| std::array::from_fn(|_| i)).collect();
            let twiddles0 = [32, 33, 34, 35];
            let twiddles1 = [36, 37];
            let twiddles2 = [38];
            let twiddles0_dbl = std::array::from_fn(|i| twiddles0[i] * 2);
            let twiddles1_dbl = std::array::from_fn(|i| twiddles1[i] * 2);
            let twiddles2_dbl = std::array::from_fn(|i| twiddles2[i] * 2);
            ifft3(
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
                assert_eq!(actual[i][0], expected[i]);
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

    #[test]
    fn test_twiddle_relation() {
        let ts = get_itwiddle_dbls(CanonicCoset::new(5).circle_domain());
        let t0 = ts[0]
            .iter()
            .copied()
            .map(|x| BaseField::from_u32_unchecked((x as u32) / 2))
            .collect::<Vec<_>>();
        let t1 = ts[1]
            .iter()
            .copied()
            .map(|x| BaseField::from_u32_unchecked((x as u32) / 2))
            .collect::<Vec<_>>();

        for i in 0..t0.len() / 4 {
            assert_eq!(t0[i * 4], t1[i * 2 + 1]);
            assert_eq!(t0[i * 4 + 1], -t1[i * 2 + 1]);
            assert_eq!(t0[i * 4 + 2], -t1[i * 2]);
            assert_eq!(t0[i * 4 + 3], t1[i * 2]);
        }
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
    fn test_ifft_lower() {
        let log_size = 4 + 3 + 3;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = (0..domain.size())
            .map(|i| BaseField::from_u32_unchecked(i as u32))
            .collect::<Vec<_>>();
        let expected_coeffs = ref_ifft(domain, values.clone());

        // Compute.
        let mut values = BaseFieldVec::from_vec(values);
        let twiddle_dbls = get_itwiddle_dbls(domain);

        unsafe {
            ifft_lower(
                std::mem::transmute(values.data.as_mut_ptr()),
                Some(&twiddle_dbls[1..4]),
                &twiddle_dbls[4..],
                (log_size - 4) as usize,
                (log_size - 4) as usize,
            );

            // Compare.
            for i in 0..expected_coeffs.len() {
                assert_eq!(values[i], expected_coeffs[i]);
            }
        }
    }

    fn run_ifft_full_test(log_size: u32) {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = (0..domain.size())
            .map(|i| BaseField::from_u32_unchecked(i as u32))
            .collect::<Vec<_>>();
        let expected_coeffs = ref_ifft(domain, values.clone());

        // Compute.
        let mut values = BaseFieldVec::from_vec(values);
        let twiddle_dbls = get_itwiddle_dbls(domain);

        unsafe {
            ifft(
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddle_dbls[1..],
                log_size as usize,
            );
            transpose_vecs(
                std::mem::transmute(values.data.as_mut_ptr()),
                (log_size - 4) as usize,
            );

            // Compare.
            for i in 0..expected_coeffs.len() {
                assert_eq!(values[i], expected_coeffs[i]);
            }
        }
    }

    #[test]
    fn test_ifft_full_10() {
        run_ifft_full_test(3 + 3 + 4);
    }
}
