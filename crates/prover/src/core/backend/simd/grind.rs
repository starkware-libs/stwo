use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;
use std::simd::u32x16;

use bytemuck::cast_slice;

use super::blake2s::compress16;
use super::SimdBackend;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::channel::Blake2sChannel;
#[cfg(not(target_arch = "wasm32"))]
use crate::core::channel::{Channel, Poseidon252Channel};
use crate::core::proof_of_work::GrindOps;

impl GrindOps<Blake2sChannel> for SimdBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        assert!(pow_bits <= 32, "pow_bits >= 32 is not supported");
        let zero: u32x16 = u32x16::default();
        let pow_bits = u32x16::splat(pow_bits);

        // TODO(spapini): Parallelize.
        let digest = channel.digest();
        let digest: &[u32] = cast_slice(&digest.0[..]);
        let state: [u32x16; 8] = std::array::from_fn(|i| u32x16::splat(digest[i]));

        let mut attempt = [zero; 16];
        for hi in 0..=u32::MAX {
            attempt[1] = u32x16::splat(hi);
            attempt[0] = u32x16::from(std::array::from_fn(|i| i as u32));
            for low in (0..=u32::MAX).step_by(N_LANES) {
                let res = compress16(state, attempt, zero, zero, zero, zero);
                let success_mask = res[0].trailing_zeros().simd_ge(pow_bits);
                if success_mask.any() {
                    let i = success_mask.to_array().iter().position(|&x| x).unwrap();
                    return ((hi as u64) << 32) + low as u64 + i as u64;
                }
                attempt[0] += u32x16::splat(N_LANES as u32);
            }
        }
        panic!("Grind failed to find a solution in a 64bit space.");
    }
}

// TODO(spapini): This is a naive implementation. Optimize it.
#[cfg(not(target_arch = "wasm32"))]
impl GrindOps<Poseidon252Channel> for SimdBackend {
    fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
        let mut nonce = 0;
        loop {
            let mut channel = channel.clone();
            channel.mix_nonce(nonce);
            if channel.trailing_zeros() >= pow_bits {
                return nonce;
            }
            nonce += 1;
        }
    }
}
