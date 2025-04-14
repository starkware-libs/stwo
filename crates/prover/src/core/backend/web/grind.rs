use super::WebBackend;
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Blake2sChannel;
#[cfg(not(target_arch = "wasm32"))]
use crate::core::channel::{Channel, Poseidon252Channel};
use crate::core::proof_of_work::GrindOps;

impl GrindOps<Blake2sChannel> for WebBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        SimdBackend::grind(channel, pow_bits)
    }
}

// TODO(shahars): This is a naive implementation. Optimize it.
#[cfg(not(target_arch = "wasm32"))]
impl GrindOps<Poseidon252Channel> for WebBackend {
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
