use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;
use std::simd::u32x16;

use bytemuck::cast_slice;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::SimdBackend;
use crate::core::backend::simd::blake2s::hash_16;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::channel::Blake2sChannel;
#[cfg(not(target_arch = "wasm32"))]
use crate::core::channel::{Channel, Poseidon252Channel};
use crate::core::proof_of_work::GrindOps;

// Note: GRIND_LOW_BITS is a cap on how much extra time we need to wait for all threads to finish.
const GRIND_LOW_BITS: u32 = 20;
const GRIND_HI_BITS: u32 = 64 - GRIND_LOW_BITS;

impl GrindOps<Blake2sChannel> for SimdBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        // TODO(first): support more than 32 bits.
        assert!(pow_bits <= 32, "pow_bits > 32 is not supported");
        let digest = channel.digest();
        let digest: &[u32] = cast_slice(&digest.0[..]);

        #[cfg(not(feature = "parallel"))]
        let res = (0..=(1 << GRIND_HI_BITS))
            .find_map(|hi| grind_blake(digest, hi, pow_bits))
            .expect("Grind failed to find a solution.");

        #[cfg(feature = "parallel")]
        let res = (0..=(1 << GRIND_HI_BITS))
            .into_par_iter()
            .find_map_first(|hi| grind_blake(digest, hi, pow_bits))
            .expect("Grind failed to find a solution.");

        res
    }
}

fn grind_blake(digest: &[u32], hi: u64, pow_bits: u32) -> Option<u64> {
    const DIGEST_SIZE: usize = std::mem::size_of::<[u32; 8]>();
    const NONCE_SIZE: usize = std::mem::size_of::<u64>();
    let zero: u32x16 = u32x16::splat(0);
    let offsets_vec = u32x16::from(std::array::from_fn(|i| i as u32));
    let pow_bits = u32x16::splat(pow_bits);

    let state: [_; 8] = std::array::from_fn(|i| u32x16::splat(digest[i]));

    let mut attempt_low = u32x16::splat((hi << GRIND_LOW_BITS) as u32) + offsets_vec;
    let attempt_high = u32x16::splat((hi >> (32 - GRIND_LOW_BITS)) as u32);
    for low in (0..(1 << GRIND_LOW_BITS)).step_by(N_LANES) {
        let msgs = std::array::from_fn(|i| match i {
            0..=7 => state[i],
            8 => attempt_low,
            9 => attempt_high,
            _ => zero,
        });
        let res = hash_16(msgs, (DIGEST_SIZE + NONCE_SIZE) as u64);
        let success_mask = res[0].trailing_zeros().simd_ge(pow_bits);
        if success_mask.any() {
            let i = success_mask.to_array().iter().position(|&x| x).unwrap();
            return Some((hi << GRIND_LOW_BITS) + low as u64 + i as u64);
        }
        attempt_low += u32x16::splat(N_LANES as u32);
    }
    None
}

// TODO(shahars): This is a naive implementation. Optimize it.
#[cfg(not(target_arch = "wasm32"))]
impl GrindOps<Poseidon252Channel> for SimdBackend {
    fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
        let mut nonce = 0;
        loop {
            let mut channel = channel.clone();
            channel.mix_u64(nonce);
            if channel.trailing_zeros() >= pow_bits {
                return nonce;
            }
            nonce += 1;
        }
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "parallel")]
    #[test]
    fn test_grind_blake_is_determinstic() {
        use itertools::Itertools;

        use super::*;

        let pow_bits = 2;
        let n_attempts = 1000;
        let mut channel = Blake2sChannel::default();
        channel.mix_u64(0);

        let results = (0..n_attempts)
            .map(|_| SimdBackend::grind(&channel, pow_bits))
            .collect_vec();

        assert!(results.iter().all(|r| r == &results[0]));
    }
}
