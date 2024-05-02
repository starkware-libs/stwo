use std::mem::transmute;
use std::ptr;
use std::simd::{i32x16, simd_swizzle, u32x16, u32x8, Simd, Swizzle};

use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::utils::{LoEvensInterleaveHiEvens, LoOddsInterleaveHiOdds};
use crate::core::backend::simd::PackedBaseField;

pub mod ifft;
pub mod rfft;

pub const CACHED_FFT_LOG_SIZE: u32 = 16;

pub const MIN_FFT_LOG_SIZE: u32 = 5;

// TODO(spapini): FFTs return a redundant representation, that can get the value P. need to reduce
// it somewhere.

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
    for b in 0..1 << (log_n_vecs & 1) {
        for a in 0..1 << half {
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
    }
}

/// Computes the twiddles for the first fft layer from the second, and loads both to SIMD registers.
///
/// Returns the twiddles for the first layer and the twiddles for the second layer.
pub fn compute_first_twiddles(twiddle1_dbl: u32x8) -> (u32x16, u32x16) {
    // Start by loading the twiddles for the second layer (layer 1):
    // The twiddles for layer 1 are replicated in the following pattern:
    //   0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
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
    struct IndicesFromT1;

    impl Swizzle<16> for IndicesFromT1 {
        const INDEX: [usize; 16] = [
            0b0001, 0b0001, 0b0000, 0b0000, 0b0011, 0b0011, 0b0010, 0b0010, 0b0101, 0b0101, 0b0100,
            0b0100, 0b0111, 0b0111, 0b0110, 0b0110,
        ];
    }

    // Xoring a double twiddle with 2^32-2 transforms it to the double of it negation.
    // Note that this keeps the values as a double of a value in the range [0, P].
    const NEGATION_MASK: u32x16 = unsafe {
        transmute(i32x16::from_array([
            0, -2, -2, 0, 0, -2, -2, 0, 0, -2, -2, 0, 0, -2, -2, 0,
        ]))
    };

    let t0 = IndicesFromT1::swizzle(t1) ^ NEGATION_MASK;

    (t0, t1)
}

unsafe fn load(mem_addr: *const u32) -> u32x16 {
    ptr::read(mem_addr as *const u32x16)
}

unsafe fn store(mem_addr: *mut u32, a: u32x16) {
    ptr::write(mem_addr as *mut u32x16, a);
}

#[cfg(target_arch = "aarch64")]
fn _mul_twiddle_neon(a: PackedBaseField, twiddle_dbl: u32x16) -> PackedBaseField {
    use core::arch::aarch64::{uint32x2_t, vmull_u32};
    use std::simd::u32x4;

    let [a0, a1, a2, a3, a4, a5, a6, a7]: [uint32x2_t; 8] = unsafe { transmute(a) };
    let [b0, b1, b2, b3, b4, b5, b6, b7]: [uint32x2_t; 8] = unsafe { transmute(twiddle_dbl) };

    // Each c_i contains |0|prod_lo|prod_hi|0|0|prod_lo|prod_hi|0|
    let c0: u32x4 = unsafe { transmute(vmull_u32(a0, b0)) };
    let c1: u32x4 = unsafe { transmute(vmull_u32(a1, b1)) };
    let c2: u32x4 = unsafe { transmute(vmull_u32(a2, b2)) };
    let c3: u32x4 = unsafe { transmute(vmull_u32(a3, b3)) };
    let c4: u32x4 = unsafe { transmute(vmull_u32(a4, b4)) };
    let c5: u32x4 = unsafe { transmute(vmull_u32(a5, b5)) };
    let c6: u32x4 = unsafe { transmute(vmull_u32(a6, b6)) };
    let c7: u32x4 = unsafe { transmute(vmull_u32(a7, b7)) };

    // *_lo contain `|prod_lo|0|prod_lo|0|prod_lo0|0|prod_lo|0|`.
    // *_hi contain `|0|prod_hi|0|prod_hi|0|prod_hi|0|prod_hi|`.
    let (mut c0_c1_lo, c0_c1_hi) = c0.deinterleave(c1);
    let (mut c2_c3_lo, c2_c3_hi) = c2.deinterleave(c3);
    let (mut c4_c5_lo, c4_c5_hi) = c4.deinterleave(c5);
    let (mut c6_c7_lo, c6_c7_hi) = c6.deinterleave(c7);

    // *_lo contain `|0|prod_lo|0|prod_lo|0|prod_lo|0|prod_lo|`.
    c0_c1_lo >>= 1;
    c2_c3_lo >>= 1;
    c4_c5_lo >>= 1;
    c6_c7_lo >>= 1;

    let lo: PackedBaseField = unsafe { transmute([c0_c1_lo, c2_c3_lo, c4_c5_lo, c6_c7_lo]) };
    let hi: PackedBaseField = unsafe { transmute([c0_c1_hi, c2_c3_hi, c4_c5_hi, c6_c7_hi]) };

    lo + hi
}

#[cfg(target_arch = "wasm32")]
fn _mul_twiddle_wasm(a: PackedBaseField, twiddle_dbl: u32x16) -> PackedBaseField {
    use core::arch::wasm32::{i64x2_extmul_high_u32x4, i64x2_extmul_low_u32x4, v128};
    use std::simd::u32x4;

    let [a0, a1, a2, a3]: [v128; 4] = unsafe { transmute(a) };
    let [b_dbl0, b_dbl1, b_dbl2, b_dbl3]: [v128; 4] = unsafe { transmute(twiddle_dbl) };

    let c0_lo: u32x4 = unsafe { transmute(i64x2_extmul_low_u32x4(a0, b_dbl0)) };
    let c0_hi: u32x4 = unsafe { transmute(i64x2_extmul_high_u32x4(a0, b_dbl0)) };
    let c1_lo: u32x4 = unsafe { transmute(i64x2_extmul_low_u32x4(a1, b_dbl1)) };
    let c1_hi: u32x4 = unsafe { transmute(i64x2_extmul_high_u32x4(a1, b_dbl1)) };
    let c2_lo: u32x4 = unsafe { transmute(i64x2_extmul_low_u32x4(a2, b_dbl2)) };
    let c2_hi: u32x4 = unsafe { transmute(i64x2_extmul_high_u32x4(a2, b_dbl2)) };
    let c3_lo: u32x4 = unsafe { transmute(i64x2_extmul_low_u32x4(a3, b_dbl3)) };
    let c3_hi: u32x4 = unsafe { transmute(i64x2_extmul_high_u32x4(a3, b_dbl3)) };

    let (mut c0_even, c0_odd) = c0_lo.deinterleave(c0_hi);
    let (mut c1_even, c1_odd) = c1_lo.deinterleave(c1_hi);
    let (mut c2_even, c2_odd) = c2_lo.deinterleave(c2_hi);
    let (mut c3_even, c3_odd) = c3_lo.deinterleave(c3_hi);

    c0_even >>= 1;
    c1_even >>= 1;
    c2_even >>= 1;
    c3_even >>= 1;

    let even: PackedBaseField = unsafe { transmute([c0_even, c1_even, c2_even, c3_even]) };
    let odd: PackedBaseField = unsafe { transmute([c0_odd, c1_odd, c2_odd, c3_odd]) };

    even + odd
}

#[cfg(target_arch = "x86_64")]
fn _mul_twiddle_avx512(a: PackedBaseField, twiddle_dbl: u32x16) -> PackedBaseField {
    use std::arch::x86_64::{__m512i, _mm512_mul_epu32, _mm512_srli_epi64};

    let a: __m512i = unsafe { transmute(a) };
    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
    // the first operand.
    let a_e = a;
    // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of
    // the first operand.
    let a_o = unsafe { _mm512_srli_epi64(a, 32) };

    let b_dbl = unsafe { transmute(twiddle_dbl) };
    let b_dbl_e = b_dbl;
    let b_dbl_o = unsafe { _mm512_srli_epi64(b_dbl, 32) };

    // To compute prod = a * b start by multiplying a_e/odd by b_dbl_e/odd.
    let prod_dbl_e: u32x16 = unsafe { transmute(_mm512_mul_epu32(a_e, b_dbl_e)) };
    let prod_dbl_o: u32x16 = unsafe { transmute(_mm512_mul_epu32(a_o, b_dbl_o)) };

    // The result of a multiplication holds a*b in as 64-bits.
    // Each 64b-bit word looks like this:
    //               1    31       31    1
    // prod_dbl_e - |0|prod_e_h|prod_e_l|0|
    // prod_dbl_o - |0|prod_o_h|prod_o_l|0|

    // Interleave the even words of prod_dbl_e with the even words of prod_dbl_o:
    let mut prod_lo = LoEvensInterleaveHiEvens::concat_swizzle(prod_dbl_e, prod_dbl_o);
    // prod_lo -    |prod_dbl_o_l|0|prod_dbl_e_l|0|
    // Divide by 2:
    prod_lo >>= 1;
    // prod_lo -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_dbl_e with the odd words of prod_dbl_o:
    let prod_hi = LoOddsInterleaveHiOdds::concat_swizzle(prod_dbl_e, prod_dbl_o);
    // prod_hi -    |0|prod_o_h|0|prod_e_h|

    unsafe {
        PackedBaseField::from_simd_unchecked(prod_lo)
            + PackedBaseField::from_simd_unchecked(prod_hi)
    }
}

#[cfg(target_arch = "x86_64")]
fn _mul_twiddle_avx2(a: PackedBaseField, twiddle_dbl: u32x16) -> PackedBaseField {
    use std::arch::x86_64::{__m256i, _mm256_mul_epu32, _mm256_srli_epi64};

    let [a0, a1]: [__m256i; 2] = unsafe { transmute(a) };
    let [b0_dbl, b1_dbl]: [__m256i; 2] = unsafe { transmute(twiddle_dbl) };

    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
    // the first operand.
    let a0_e = a0;
    let a1_e = a1;
    // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of
    // the first operand.
    let a0_o = unsafe { _mm256_srli_epi64(a0, 32) };
    let a1_o = unsafe { _mm256_srli_epi64(a1, 32) };

    let b0_dbl_e = b0_dbl;
    let b1_dbl_e = b1_dbl;
    let b0_dbl_o = unsafe { _mm256_srli_epi64(b0_dbl, 32) };
    let b1_dbl_o = unsafe { _mm256_srli_epi64(b1_dbl, 32) };

    // To compute prod = a * b start by multiplying a0/1_e/odd by b0/1_e/odd.
    let prod0_dbl_e = unsafe { _mm256_mul_epu32(a0_e, b0_dbl_e) };
    let prod0_dbl_o = unsafe { _mm256_mul_epu32(a0_o, b0_dbl_o) };
    let prod1_dbl_e = unsafe { _mm256_mul_epu32(a1_e, b1_dbl_e) };
    let prod1_dbl_o = unsafe { _mm256_mul_epu32(a1_o, b1_dbl_o) };

    let prod_dbl_e: u32x16 = unsafe { transmute([prod0_dbl_e, prod1_dbl_e]) };
    let prod_dbl_o: u32x16 = unsafe { transmute([prod0_dbl_o, prod1_dbl_o]) };

    // The result of a multiplication holds a*b in as 64-bits.
    // Each 64b-bit word looks like this:
    //               1    31       31    1
    // prod_dbl_e - |0|prod_e_h|prod_e_l|0|
    // prod_dbl_o - |0|prod_o_h|prod_o_l|0|

    // Interleave the even words of prod_dbl_e with the even words of prod_dbl_o:
    let mut prod_lo = LoEvensInterleaveHiEvens::concat_swizzle(prod_dbl_e, prod_dbl_o);
    // prod_lo -    |prod_dbl_o_l|0|prod_dbl_e_l|0|
    // Divide by 2:
    prod_lo >>= 1;
    // prod_lo -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_dbl_e with the odd words of prod_dbl_o:
    let prod_hi = LoOddsInterleaveHiOdds::concat_swizzle(prod_dbl_e, prod_dbl_o);
    // prod_hi -    |0|prod_o_h|0|prod_e_h|

    unsafe {
        PackedBaseField::from_simd_unchecked(prod_lo)
            + PackedBaseField::from_simd_unchecked(prod_hi)
    }
}

// Should only be used in the absence of a platform specific implementation.
fn _mul_twiddle_simd(a: PackedBaseField, twiddle_dbl: u32x16) -> PackedBaseField {
    const MASK_EVENS: Simd<u64, { N_LANES / 2 }> = Simd::from_array([0xFFFFFFFF; { N_LANES / 2 }]);

    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
    // the first operand.
    let a_e = unsafe { transmute::<_, Simd<u64, { N_LANES / 2 }>>(a.into_simd()) & MASK_EVENS };
    // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of
    // the first operand.
    let a_o = unsafe { transmute::<_, Simd<u64, { N_LANES / 2 }>>(a) >> 32 };

    let b_dbl_e = unsafe { transmute::<_, Simd<u64, { N_LANES / 2 }>>(twiddle_dbl) & MASK_EVENS };
    let b_dbl_o = unsafe { transmute::<_, Simd<u64, { N_LANES / 2 }>>(twiddle_dbl) >> 32 };

    // To compute prod = a * b start by multiplying
    // a_e/o by b_dbl_e/o.
    let prod_e_dbl = a_e * b_dbl_e;
    let prod_o_dbl = a_o * b_dbl_o;

    // The result of a multiplication holds a*b in as 64-bits.
    // Each 64b-bit word looks like this:
    //               1    31       31    1
    // prod_e_dbl - |0|prod_e_h|prod_e_l|0|
    // prod_o_dbl - |0|prod_o_h|prod_o_l|0|

    // Interleave the even words of prod_e_dbl with the even words of prod_o_dbl:
    // prod_ls -    |prod_o_l|0|prod_e_l|0|
    let mut prod_lows = LoEvensInterleaveHiEvens::concat_swizzle(
        unsafe { transmute::<_, Simd<u32, N_LANES>>(prod_e_dbl) },
        unsafe { transmute::<_, Simd<u32, N_LANES>>(prod_o_dbl) },
    );
    // Divide by 2:
    prod_lows >>= 1;
    // prod_ls -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
    let prod_highs = LoOddsInterleaveHiOdds::concat_swizzle(
        unsafe { transmute::<_, Simd<u32, N_LANES>>(prod_e_dbl) },
        unsafe { transmute::<_, Simd<u32, N_LANES>>(prod_o_dbl) },
    );

    // prod_hs -    |0|prod_o_h|0|prod_e_h|
    unsafe {
        PackedBaseField::from_simd_unchecked(prod_lows)
            + PackedBaseField::from_simd_unchecked(prod_highs)
    }
}
