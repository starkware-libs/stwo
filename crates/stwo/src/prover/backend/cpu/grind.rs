use super::CpuBackend;
use crate::core::channel::Channel;
use crate::core::proof_of_work::GrindOps;

impl<C: Channel> GrindOps<C> for CpuBackend {
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        let mut nonce = 0;
        loop {
            let channel = channel.clone();
            if channel.verify_pow_nonce(pow_bits, nonce) {
                return nonce;
            }
            nonce += 1;
        }
    }
}
