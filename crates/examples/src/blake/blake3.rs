//! A SIMD implementation of the BLAKE3 compression function.
//! Based on BLAKE3 specification: <https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf>

use std::simd::u32x16;

// Blake3 uses the same message schedule for all rounds
// Ref: https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs#L19
pub const MSG_SCHEDULE: [u8; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Applies [`u32::rotate_right(N)`] to each element of the vector
///
/// [`u32::rotate_right(N)`]: u32::rotate_right
#[inline(always)]
fn rotate<const N: u32>(x: u32x16) -> u32x16 {
    (x >> N) | (x << (u32::BITS - N))
}

// `inline(always)` can cause code parsing errors for wasm: "locals exceed maximum".
#[cfg_attr(not(target_arch = "wasm32"), inline(always))]
pub fn round(v: &mut [u32x16; 16], m: [u32x16; 16], _r: usize) {
    // Blake3: m should be pre-permuted according to MSG_SCHEDULE before calling round()
    // Unlike Blake2s, Blake3 uses the same permutation for all rounds
    v[0] += m[0];
    v[1] += m[2];
    v[2] += m[4];
    v[3] += m[6];
    v[0] += v[4];
    v[1] += v[5];
    v[2] += v[6];
    v[3] += v[7];
    v[12] ^= v[0];
    v[13] ^= v[1];
    v[14] ^= v[2];
    v[15] ^= v[3];
    v[12] = rotate::<16>(v[12]);
    v[13] = rotate::<16>(v[13]);
    v[14] = rotate::<16>(v[14]);
    v[15] = rotate::<16>(v[15]);
    v[8] += v[12];
    v[9] += v[13];
    v[10] += v[14];
    v[11] += v[15];
    v[4] ^= v[8];
    v[5] ^= v[9];
    v[6] ^= v[10];
    v[7] ^= v[11];
    v[4] = rotate::<12>(v[4]);
    v[5] = rotate::<12>(v[5]);
    v[6] = rotate::<12>(v[6]);
    v[7] = rotate::<12>(v[7]);
    v[0] += m[1];
    v[1] += m[3];
    v[2] += m[5];
    v[3] += m[7];
    v[0] += v[4];
    v[1] += v[5];
    v[2] += v[6];
    v[3] += v[7];
    v[12] ^= v[0];
    v[13] ^= v[1];
    v[14] ^= v[2];
    v[15] ^= v[3];
    v[12] = rotate::<8>(v[12]);
    v[13] = rotate::<8>(v[13]);
    v[14] = rotate::<8>(v[14]);
    v[15] = rotate::<8>(v[15]);
    v[8] += v[12];
    v[9] += v[13];
    v[10] += v[14];
    v[11] += v[15];
    v[4] ^= v[8];
    v[5] ^= v[9];
    v[6] ^= v[10];
    v[7] ^= v[11];
    v[4] = rotate::<7>(v[4]);
    v[5] = rotate::<7>(v[5]);
    v[6] = rotate::<7>(v[6]);
    v[7] = rotate::<7>(v[7]);

    v[0] += m[8];
    v[1] += m[10];
    v[2] += m[12];
    v[3] += m[14];
    v[0] += v[5];
    v[1] += v[6];
    v[2] += v[7];
    v[3] += v[4];
    v[15] ^= v[0];
    v[12] ^= v[1];
    v[13] ^= v[2];
    v[14] ^= v[3];
    v[15] = rotate::<16>(v[15]);
    v[12] = rotate::<16>(v[12]);
    v[13] = rotate::<16>(v[13]);
    v[14] = rotate::<16>(v[14]);
    v[10] += v[15];
    v[11] += v[12];
    v[8] += v[13];
    v[9] += v[14];
    v[5] ^= v[10];
    v[6] ^= v[11];
    v[7] ^= v[8];
    v[4] ^= v[9];
    v[5] = rotate::<12>(v[5]);
    v[6] = rotate::<12>(v[6]);
    v[7] = rotate::<12>(v[7]);
    v[4] = rotate::<12>(v[4]);
    v[0] += m[9];
    v[1] += m[11];
    v[2] += m[13];
    v[3] += m[15];
    v[0] += v[5];
    v[1] += v[6];
    v[2] += v[7];
    v[3] += v[4];
    v[15] ^= v[0];
    v[12] ^= v[1];
    v[13] ^= v[2];
    v[14] ^= v[3];
    v[15] = rotate::<8>(v[15]);
    v[12] = rotate::<8>(v[12]);
    v[13] = rotate::<8>(v[13]);
    v[14] = rotate::<8>(v[14]);
    v[10] += v[15];
    v[11] += v[12];
    v[8] += v[13];
    v[9] += v[14];
    v[5] ^= v[10];
    v[6] ^= v[11];
    v[7] ^= v[8];
    v[4] ^= v[9];
    v[5] = rotate::<7>(v[5]);
    v[6] = rotate::<7>(v[6]);
    v[7] = rotate::<7>(v[7]);
    v[4] = rotate::<7>(v[4]);
}
