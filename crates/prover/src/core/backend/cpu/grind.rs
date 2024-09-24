#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::CpuBackend;
use crate::core::channel::Channel;
use crate::core::proof_of_work::GrindOps;

impl<C> GrindOps<C> for CpuBackend
where
    C: Channel + std::marker::Sync,
{
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        // TODO(spapini): This is a naive implementation. Optimize it.

        let check_nonce = |nonce: &u64| {
            let mut channel = channel.clone();
            channel.mix_u64(*nonce);
            channel.trailing_zeros() >= pow_bits
        };

        {
            let range = 0..u64::MAX;
            #[cfg(not(feature = "parallel"))]
            {
                range.into_iter().find(check_nonce)
            }

            #[cfg(feature = "parallel")]
            {
                range.into_par_iter().find_any(check_nonce)
            }
        }
        .expect("Grind failed to find a solution.")
    }
}
