//! SIMD-parallel Keccak-f\[1600\] permutation.
//!
//! Operates on `N_LANES_KECCAK = 8` parallel Keccak states packed lane-wise into
//! `[u64x8; 25]`. Lane `i` of the result is identical to running scalar
//! Keccak-f\[1600\] on lane `i` of the input. Width matches the 512-bit lane used
//! elsewhere in the SIMD backend (e.g. blake2s with `u32x16`): native on AVX-512,
//! splits to 2× AVX2 ops on AVX2 hosts.

use std::simd::u64x8;

/// Number of parallel Keccak states processed by [`keccak_f1600x8`].
pub const N_LANES_KECCAK: usize = 8;

/// Number of `u64` words in a Keccak-f\[1600\] state (5×5 state array).
pub const PLEN: usize = 25;

/// Number of rounds in Keccak-f\[1600\].
const KECCAK_ROUNDS: usize = 24;

const RHO: [u32; 24] =
    [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];

const PI: [usize; 24] =
    [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];

const RC: [u64x8; KECCAK_ROUNDS] = [
    u64x8::splat(0x0000000000000001),
    u64x8::splat(0x0000000000008082),
    u64x8::splat(0x800000000000808a),
    u64x8::splat(0x8000000080008000),
    u64x8::splat(0x000000000000808b),
    u64x8::splat(0x0000000080000001),
    u64x8::splat(0x8000000080008081),
    u64x8::splat(0x8000000000008009),
    u64x8::splat(0x000000000000008a),
    u64x8::splat(0x0000000000000088),
    u64x8::splat(0x0000000080008009),
    u64x8::splat(0x000000008000000a),
    u64x8::splat(0x000000008000808b),
    u64x8::splat(0x800000000000008b),
    u64x8::splat(0x8000000000008089),
    u64x8::splat(0x8000000000008003),
    u64x8::splat(0x8000000000008002),
    u64x8::splat(0x8000000000000080),
    u64x8::splat(0x000000000000800a),
    u64x8::splat(0x800000008000000a),
    u64x8::splat(0x8000000080008081),
    u64x8::splat(0x8000000000008080),
    u64x8::splat(0x0000000080000001),
    u64x8::splat(0x8000000080008008),
];

const C_INIT: [u64x8; 5] = [u64x8::splat(0); 5];

#[inline(always)]
fn rotate_left(v: u64x8, n: u32) -> u64x8 {
    // Safe because every RHO value is in 1..=62.
    debug_assert!((1..=63).contains(&n));
    v << u64x8::splat(n as u64) | v >> u64x8::splat(64 - n as u64)
}

/// In-place Keccak-f\[1600\] on eight parallel states packed as `[u64x8; 25]`.
///
/// For every lane `i` in `0..N_LANES_KECCAK`, on return `state[j].as_array()[i]` equals
/// scalar Keccak-f\[1600\] applied to the input `[u64; 25]` formed by
/// `input[j].as_array()[i]` for `j in 0..PLEN`.
pub fn keccak_f1600x8(state: &mut [u64x8; PLEN]) {
    for &rc in &RC {
        // Theta.
        let mut c = C_INIT;
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ rotate_left(c[(x + 1) % 5], 1);
            for y_step in 0..5 {
                state[5 * y_step + x] ^= d;
            }
        }

        // Rho and Pi.
        let mut last = state[1];
        for x in 0..24 {
            let tmp = state[PI[x]];
            state[PI[x]] = rotate_left(last, RHO[x]);
            last = tmp;
        }

        // Chi.
        for y_step in 0..5 {
            let y = 5 * y_step;
            let row: [u64x8; 5] =
                [state[y], state[y + 1], state[y + 2], state[y + 3], state[y + 4]];
            for x in 0..5 {
                state[y + x] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
            }
        }

        // Iota.
        state[0] ^= rc;
    }
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::simd::u64x8;

    use keccak::Keccak;

    use super::{N_LANES_KECCAK, PLEN, keccak_f1600x8};

    /// Apply scalar `keccak::f1600` to `state` in place.
    fn scalar_f1600(state: &mut [u64; PLEN]) {
        Keccak::default().with_f1600(|f| f(state));
    }

    fn pack(states: [[u64; PLEN]; N_LANES_KECCAK]) -> [u64x8; PLEN] {
        array::from_fn(|j| u64x8::from_array(array::from_fn(|i| states[i][j])))
    }

    fn unpack(state: [u64x8; PLEN]) -> [[u64; PLEN]; N_LANES_KECCAK] {
        array::from_fn(|i| array::from_fn(|j| state[j].as_array()[i]))
    }

    #[test]
    fn zero_state_matches_scalar() {
        let mut simd_state = pack([[0; PLEN]; N_LANES_KECCAK]);
        keccak_f1600x8(&mut simd_state);

        let mut expected = [0u64; PLEN];
        scalar_f1600(&mut expected);

        for lane in unpack(simd_state) {
            assert_eq!(lane, expected);
        }
    }

    #[test]
    fn single_state_matches_scalar() {
        // All lanes carry the same state — every output lane must equal the scalar reference.
        let seed: [u64; PLEN] = array::from_fn(|j| (j as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let mut simd_state = pack([seed; N_LANES_KECCAK]);
        keccak_f1600x8(&mut simd_state);

        let mut expected = seed;
        scalar_f1600(&mut expected);

        for lane in unpack(simd_state) {
            assert_eq!(lane, expected);
        }
    }

    #[test]
    fn eight_distinct_states_match_scalar() {
        let inputs: [[u64; PLEN]; N_LANES_KECCAK] = array::from_fn(|i| {
            array::from_fn(|j| {
                ((i + 1) as u64)
                    .wrapping_mul(0xDEADBEEFCAFEBABE)
                    .wrapping_add((j as u64).wrapping_mul(0x9E3779B97F4A7C15))
            })
        });
        let mut simd_state = pack(inputs);
        keccak_f1600x8(&mut simd_state);

        let expected = inputs.map(|mut state| {
            scalar_f1600(&mut state);
            state
        });

        assert_eq!(unpack(simd_state), expected);
    }

    #[test]
    fn two_step_matches_scalar() {
        // Run the permutation twice through both implementations to catch bugs in how the state
        // is read back / forwarded into the next round.
        let inputs: [[u64; PLEN]; N_LANES_KECCAK] =
            array::from_fn(|i| array::from_fn(|j| (i as u64) << 32 | (j as u64)));
        let mut simd_state = pack(inputs);
        keccak_f1600x8(&mut simd_state);
        keccak_f1600x8(&mut simd_state);

        let expected = inputs.map(|mut state| {
            scalar_f1600(&mut state);
            scalar_f1600(&mut state);
            state
        });

        assert_eq!(unpack(simd_state), expected);
    }
}
