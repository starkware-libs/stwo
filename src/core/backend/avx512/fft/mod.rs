pub mod ifft;
pub mod rfft;

use std::arch::x86_64::{__m512i, _mm512_load_epi32, _mm512_store_epi32};

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

pub const CACHED_FFT_LOG_SIZE: usize = 16;
pub const MIN_FFT_LOG_SIZE: usize = 5;

// TODO(spapini): FFTs return a redundant representation, that can get the value P. need to reduce
// it somewhere.

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
