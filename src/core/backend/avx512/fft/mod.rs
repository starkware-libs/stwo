pub mod ifft;

use std::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_broadcast_i32x4, _mm512_broadcast_i64x4, _mm512_load_epi32,
    _mm512_min_epu32, _mm512_mul_epu32, _mm512_permutex2var_epi32, _mm512_permutexvar_epi32,
    _mm512_set1_epi64, _mm512_srli_epi64, _mm512_store_epi32, _mm512_sub_epi32, _mm512_xor_epi32,
};

/// An input to _mm512_permutex2var_epi32, and is used to interleave the even words of a
/// L is an input to _mm512_permutex2var_epi32, and is used to interleave the even words of a
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

/// Transposes the AVX vectors in the given array.
/// Swaps the bit index abc <-> cba, where |a|=|c| and |b| = 0 or 1, according to the parity of
/// `log_n_vecs`.
/// When log_n_vecs is odd, transforms the index abc <-> cba, w
///
/// # Safety
/// This function is unsafe because it takes a raw pointer to i32 values.
/// `values` must be aligned to 64 bytes.
///
/// # Arguments
/// * `values`: A mutable pointer to the values that are to be transposed.
/// * `log_n_vecs`: The log of the number of AVX vectors in the `values` array.
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
    use std::arch::x86_64::{_mm512_set1_epi32, _mm512_setr_epi32};

    use self::ifft::get_itwiddle_dbls;
    use super::*;
    use crate::core::backend::cpu::CPUCirclePoly;
    use crate::core::fft::butterfly;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain};
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
    fn test_vecwise_butterflies_real() {
        let domain = CanonicCoset::new(5).circle_domain();
        let twiddle_dbls = get_twiddle_dbls(domain);
        assert_eq!(twiddle_dbls.len(), 5);
        let values0: [i32; 16] = std::array::from_fn(|i| i as i32);
        let values1: [i32; 16] = std::array::from_fn(|i| (i + 16) as i32);
        let result: [BaseField; 32] = unsafe {
            let (val0, val1) = avx_butterfly(
                std::mem::transmute(values0),
                std::mem::transmute(values1),
                _mm512_set1_epi32(twiddle_dbls[4][0]),
            );
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

    fn get_twiddle_dbls(domain: CircleDomain) -> Vec<Vec<i32>> {
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

    fn ref_fft(domain: CircleDomain, mut values: Vec<BaseField>) -> Vec<BaseField> {
        bit_reverse(&mut values);
        let poly = CPUCirclePoly::new(values);
        let mut expected_values = poly.evaluate(domain).values;
        bit_reverse(&mut expected_values);
        expected_values
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
}
