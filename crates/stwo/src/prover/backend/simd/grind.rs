use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;
use std::simd::u32x16;

use bytemuck::cast_slice;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::{span, Level};

use super::SimdBackend;
use crate::core::channel::Blake2sChannel;
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::prover::backend::simd::blake2s::hash_16;
use crate::prover::backend::simd::m31::N_LANES;

// Note: GRIND_LOW_BITS is a cap on how much extra time we need to wait for all threads to finish.
const GRIND_LOW_BITS: u32 = 20;

impl GrindOps<Blake2sChannel> for SimdBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        let _span = span!(Level::TRACE, "Simd Blake2s Grind", class = "Blake2s Grind");

        // TODO(first): support more than 32 bits.
        assert!(pow_bits <= 32, "pow_bits > 32 is not supported");
        let digest = channel.digest();

        let mut hasher = Blake2sHasher::default();
        hasher.update(&Blake2sChannel::POW_PREFIX.to_le_bytes());
        hasher.update(&digest.0[..]);
        hasher.update(&pow_bits.to_le_bytes());
        let prefixed_digest = hasher.finalize();
        let prefixed_digest: &[u32] = cast_slice(&prefixed_digest.0[..]);

        #[cfg(not(feature = "parallel"))]
        let res = (0..)
            .find_map(|hi| grind_blake(prefixed_digest, hi, pow_bits))
            .expect("Grind failed to find a solution.");

        #[cfg(feature = "parallel")]
        let res = parallel_grind(prefixed_digest, pow_bits, grind_blake);

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

// Deterministically finds the smallest nonce that satisfies:
// `hash(digest, nonce).trailing_zeros() >= pow_bits`.
#[cfg(feature = "parallel")]
fn parallel_grind<GRIND, DIGEST>(digest: DIGEST, pow_bits: u32, grind: GRIND) -> u64
where
    GRIND: Fn(DIGEST, u64, u32) -> Option<u64> + Send + Sync,
    DIGEST: Send + Sync + Copy,
{
    use core::sync::atomic::{AtomicU64, Ordering};

    let n_workers = rayon::current_num_threads() as u64;
    let next_chunk = AtomicU64::new(n_workers);
    let smallest_good_chunk = AtomicU64::new(u64::MAX);
    let found = (0..n_workers)
        .into_par_iter()
        .filter_map(|thread_id| {
            let mut chunk_id = thread_id;
            loop {
                if let Some(found) = grind(digest, chunk_id, pow_bits) {
                    // Signal higher chunk handlers to stop.
                    let current_smallest_chunk = smallest_good_chunk.load(Ordering::Relaxed);
                    if chunk_id < current_smallest_chunk {
                        // If fails, it means that another thread found a solution.
                        // Every thread that found an answer returns it, the results are compared.
                        let _ = smallest_good_chunk.compare_exchange(
                            current_smallest_chunk,
                            chunk_id,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        );
                    }
                    return Some(found);
                }
                // Assign the next chunk to this thread.
                chunk_id = next_chunk.fetch_add(1, Ordering::Relaxed);
                if chunk_id >= smallest_good_chunk.load(Ordering::Relaxed) {
                    break;
                }
            }
            None
        })
        .min();

    found.expect("Grind failed to find a solution.")
}

#[cfg(not(target_arch = "wasm32"))]
pub mod poseidon252 {
    use starknet_crypto::poseidon_hash_many;
    use starknet_ff::FieldElement as FieldElement252;

    use super::*;
    use crate::core::channel::Poseidon252Channel;

    const GRIND_LOW_BITS: u32 = 14;

    impl GrindOps<Poseidon252Channel> for SimdBackend {
        fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
            let digest = channel.digest();
            let prefixed_digest = poseidon_hash_many(&[
                Poseidon252Channel::POW_PREFIX.into(),
                digest,
                pow_bits.into(),
            ]);
            #[cfg(not(feature = "parallel"))]
            let res = (0..)
                .find_map(|hi| grind_poseidon(digest, hi, pow_bits))
                .expect("Grind failed to find a solution.");

            #[cfg(feature = "parallel")]
            let res = parallel_grind(prefixed_digest, pow_bits, grind_poseidon);
            res
        }
    }

    fn grind_poseidon(digest: FieldElement252, chunk_id: u64, pow_bits: u32) -> Option<u64> {
        for low in 0..(1 << GRIND_LOW_BITS) {
            let nonce = low | (chunk_id << GRIND_LOW_BITS);
            let hash = starknet_crypto::poseidon_hash(digest, nonce.into());
            let trailing_zeros =
                u128::from_be_bytes(hash.to_bytes_be()[16..].try_into().unwrap()).trailing_zeros();
            if trailing_zeros >= pow_bits {
                return Some(nonce);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use crate::core::channel::Channel;

    #[cfg(all(feature = "parallel", feature = "slow-tests"))]
    #[test]
    fn test_parallel_grind_with_high_pow_bits() {
        let mut channel = Blake2sChannel::default();
        channel.mix_u64(0x1111222233334344);
        let pow_bits = 26;
        for _ in 0..10 {
            let res = SimdBackend::grind(&channel, pow_bits);
            assert!(channel.verify_pow_nonce(pow_bits, res));
            channel.mix_u64(res);
            channel.mix_u64(0x1111222233334344);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_grind_poseidon() {
        let pow_bits = 10;
        let mut channel = crate::core::channel::Poseidon252Channel::default();
        channel.mix_u64(0x1111222233334344);

        let nonce = SimdBackend::grind(&channel, pow_bits);
        assert!(channel.verify_pow_nonce(pow_bits, nonce));
    }

    fn test_grind_is_deterministic<C: Channel>()
    where
        SimdBackend: GrindOps<C>,
    {
        let pow_bits = 2;
        let n_attempts = 1000;
        let mut channel = C::default();
        channel.mix_u64(0);

        let results = (0..n_attempts)
            .map(|_| SimdBackend::grind(&channel, pow_bits))
            .collect_vec();

        assert!(results.iter().all(|r| r == &results[0]));
    }

    #[test]
    fn test_grind_blake_is_deterministic() {
        test_grind_is_deterministic::<Blake2sChannel>();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_grind_poseidon_is_deterministic() {
        test_grind_is_deterministic::<crate::core::channel::Poseidon252Channel>();
    }
}
