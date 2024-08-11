use super::SimdBackend;
use crate::core::channel::Channel;
use crate::core::proof_of_work::GrindOps;

impl<C: Channel> GrindOps<C> for SimdBackend {
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        // TODO(spark): This is a naive implementation. Optimize it.
        let mut nonce = 0;
        loop {
            let mut channel = channel.clone();
            channel.mix_nonce(nonce);
            if channel.leading_zeros() >= pow_bits {
                return nonce;
            }
            nonce += 1;
        }
    }
}
