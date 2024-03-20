//! Based on bla bla's impl. TODO: add the link.

use std::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_or_si512, _mm512_permutex2var_epi32, _mm512_set1_epi32,
    _mm512_slli_epi32, _mm512_srli_epi32, _mm512_xor_si512,
};

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const SIGMA: [[u8; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// # Safety
#[inline(always)]
pub unsafe fn set1(iv: i32) -> __m512i {
    _mm512_set1_epi32(iv)
}

#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi32(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m512i, b: __m512i) -> __m512i {
    _mm512_xor_si512(a, b)
}

#[inline(always)]
unsafe fn rot16(x: __m512i) -> __m512i {
    _mm512_or_si512(_mm512_srli_epi32(x, 16), _mm512_slli_epi32(x, 32 - 16))
}

#[inline(always)]
unsafe fn rot12(x: __m512i) -> __m512i {
    _mm512_or_si512(_mm512_srli_epi32(x, 12), _mm512_slli_epi32(x, 32 - 12))
}

#[inline(always)]
unsafe fn rot8(x: __m512i) -> __m512i {
    _mm512_or_si512(_mm512_srli_epi32(x, 8), _mm512_slli_epi32(x, 32 - 8))
}

#[inline(always)]
unsafe fn rot7(x: __m512i) -> __m512i {
    _mm512_or_si512(_mm512_srli_epi32(x, 7), _mm512_slli_epi32(x, 32 - 7))
}

#[inline(always)]
unsafe fn round(v: &mut [__m512i; 16], m: &[__m512i; 16], r: usize) {
    v[0] = add(v[0], m[SIGMA[r][0] as usize]);
    v[1] = add(v[1], m[SIGMA[r][2] as usize]);
    v[2] = add(v[2], m[SIGMA[r][4] as usize]);
    v[3] = add(v[3], m[SIGMA[r][6] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[15] = rot16(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot12(v[4]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[0] = add(v[0], m[SIGMA[r][1] as usize]);
    v[1] = add(v[1], m[SIGMA[r][3] as usize]);
    v[2] = add(v[2], m[SIGMA[r][5] as usize]);
    v[3] = add(v[3], m[SIGMA[r][7] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[15] = rot8(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot7(v[4]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);

    v[0] = add(v[0], m[SIGMA[r][8] as usize]);
    v[1] = add(v[1], m[SIGMA[r][10] as usize]);
    v[2] = add(v[2], m[SIGMA[r][12] as usize]);
    v[3] = add(v[3], m[SIGMA[r][14] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot16(v[15]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[4] = rot12(v[4]);
    v[0] = add(v[0], m[SIGMA[r][9] as usize]);
    v[1] = add(v[1], m[SIGMA[r][11] as usize]);
    v[2] = add(v[2], m[SIGMA[r][13] as usize]);
    v[3] = add(v[3], m[SIGMA[r][15] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot8(v[15]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);
    v[4] = rot7(v[4]);
}

/// # Safety
pub unsafe fn transpose_msgs(mut data: [__m512i; 16]) -> [__m512i; 16] {
    //   abcd:0123 => 3abc:d012
    const L: __m512i = unsafe {
        core::mem::transmute([
            0b00000, 0b00010, 0b00100, 0b00110, 0b01000, 0b01010, 0b01100, 0b01110, 0b10000,
            0b10010, 0b10100, 0b10110, 0b11000, 0b11010, 0b11100, 0b11110,
        ])
    };
    const H: __m512i = unsafe {
        core::mem::transmute([
            0b00001, 0b00011, 0b00101, 0b00111, 0b01001, 0b01011, 0b01101, 0b01111, 0b10001,
            0b10011, 0b10101, 0b10111, 0b11001, 0b11011, 0b11101, 0b11111,
        ])
    };
    for _ in 0..4 {
        data = [
            _mm512_permutex2var_epi32(data[0], L, data[1]),
            _mm512_permutex2var_epi32(data[2], L, data[3]),
            _mm512_permutex2var_epi32(data[4], L, data[5]),
            _mm512_permutex2var_epi32(data[6], L, data[7]),
            _mm512_permutex2var_epi32(data[8], L, data[9]),
            _mm512_permutex2var_epi32(data[10], L, data[11]),
            _mm512_permutex2var_epi32(data[12], L, data[13]),
            _mm512_permutex2var_epi32(data[14], L, data[15]),
            _mm512_permutex2var_epi32(data[0], H, data[1]),
            _mm512_permutex2var_epi32(data[2], H, data[3]),
            _mm512_permutex2var_epi32(data[4], H, data[5]),
            _mm512_permutex2var_epi32(data[6], H, data[7]),
            _mm512_permutex2var_epi32(data[8], H, data[9]),
            _mm512_permutex2var_epi32(data[10], H, data[11]),
            _mm512_permutex2var_epi32(data[12], H, data[13]),
            _mm512_permutex2var_epi32(data[14], H, data[15]),
        ];
    }
    data
}

/// # Safety
pub unsafe fn transpose_states(mut states: [__m512i; 8]) -> [__m512i; 8] {
    // It's ordered as 012:abcd.
    // We need it as abc:d012.
    // We can apply 012:abcd => 12a:bcd0, 3 times.
    const L: __m512i = unsafe {
        core::mem::transmute([
            0b00000, 0b10000, 0b00001, 0b10001, 0b00010, 0b10010, 0b00011, 0b10011, 0b00100,
            0b10100, 0b00101, 0b10101, 0b00110, 0b10110, 0b00111, 0b10111,
        ])
    };
    const H: __m512i = unsafe {
        core::mem::transmute([
            0b01000, 0b11000, 0b01001, 0b11001, 0b01010, 0b11010, 0b01011, 0b11011, 0b01100,
            0b11100, 0b01101, 0b11101, 0b01110, 0b11110, 0b01111, 0b11111,
        ])
    };
    for _ in 0..3 {
        states = [
            _mm512_permutex2var_epi32(states[0], L, states[4]),
            _mm512_permutex2var_epi32(states[0], H, states[4]),
            _mm512_permutex2var_epi32(states[1], L, states[5]),
            _mm512_permutex2var_epi32(states[1], H, states[5]),
            _mm512_permutex2var_epi32(states[2], L, states[6]),
            _mm512_permutex2var_epi32(states[2], H, states[6]),
            _mm512_permutex2var_epi32(states[3], L, states[7]),
            _mm512_permutex2var_epi32(states[3], H, states[7]),
        ];
    }
    states
}

/// # Safety
pub unsafe fn compress16_transposed(
    h_vecs: &mut [__m512i; 8],
    msg_vecs: &[__m512i; 16],
    count_low: __m512i,
    count_high: __m512i,
    lastblock: __m512i,
    lastnode: __m512i,
) {
    let mut v = [
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        set1(IV[0] as i32),
        set1(IV[1] as i32),
        set1(IV[2] as i32),
        set1(IV[3] as i32),
        xor(set1(IV[4] as i32), count_low),
        xor(set1(IV[5] as i32), count_high),
        xor(set1(IV[6] as i32), lastblock),
        xor(set1(IV[7] as i32), lastnode),
    ];

    round(&mut v, msg_vecs, 0);
    round(&mut v, msg_vecs, 1);
    round(&mut v, msg_vecs, 2);
    round(&mut v, msg_vecs, 3);
    round(&mut v, msg_vecs, 4);
    round(&mut v, msg_vecs, 5);
    round(&mut v, msg_vecs, 6);
    round(&mut v, msg_vecs, 7);
    round(&mut v, msg_vecs, 8);
    round(&mut v, msg_vecs, 9);

    h_vecs[0] = xor(xor(h_vecs[0], v[0]), v[8]);
    h_vecs[1] = xor(xor(h_vecs[1], v[1]), v[9]);
    h_vecs[2] = xor(xor(h_vecs[2], v[2]), v[10]);
    h_vecs[3] = xor(xor(h_vecs[3], v[3]), v[11]);
    h_vecs[4] = xor(xor(h_vecs[4], v[4]), v[12]);
    h_vecs[5] = xor(xor(h_vecs[5], v[5]), v[13]);
    h_vecs[6] = xor(xor(h_vecs[6], v[6]), v[14]);
    h_vecs[7] = xor(xor(h_vecs[7], v[7]), v[15]);
}
