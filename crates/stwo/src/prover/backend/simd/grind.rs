use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;
use std::simd::u32x16;

use bytemuck::cast_slice;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::{span, Level};

use super::SimdBackend;
use crate::core::channel::Blake2sChannelGeneric;
use crate::core::fields::m31::P;
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::blake2_hash::Blake2sHasherGeneric;
use crate::prover::backend::simd::blake2s::hash_16;
use crate::prover::backend::simd::m31::{PackedM31, N_LANES};

// Note: GRIND_LOW_BITS is a cap on how much extra time we need to wait for all threads to finish.
// It must be <= 30 if we want to guarantee that the lowest 32 bits of the nonce are < 2^31 - 1.
const GRIND_LOW_BITS: u32 = 20;

impl<const IS_M31_OUTPUT: bool> GrindOps<Blake2sChannelGeneric<IS_M31_OUTPUT>> for SimdBackend {
    /// Outputs the smallest nonce of the form `(a << 32) | b`, where `0 <= a < 2^31 - 1` and
    /// `0 <= b < 2^GRIND_LOW_BITS`.
    fn grind(channel: &Blake2sChannelGeneric<IS_M31_OUTPUT>, pow_bits: u32) -> u64 {
        let _span = span!(Level::TRACE, "Simd Blake2s Grind", class = "Blake2s Grind");

        // TODO(first): support more than 32 bits.
        assert!(pow_bits <= 32, "pow_bits > 32 is not supported");
        let digest = channel.digest();

        // Compute the prefix digest H(POW_PREFIX, [0_u8; 12], digest, n_bits).
        let mut hasher = Blake2sHasherGeneric::<IS_M31_OUTPUT>::default();
        hasher.update(&Blake2sChannelGeneric::<IS_M31_OUTPUT>::POW_PREFIX.to_le_bytes());
        hasher.update(&[0_u8; 12]);
        hasher.update(&digest.0[..]);
        hasher.update(&pow_bits.to_le_bytes());
        let prefixed_digest = hasher.finalize();
        let prefixed_digest: &[u32] = cast_slice(&prefixed_digest.0[..]);

        #[cfg(not(feature = "parallel"))]
        let res = (0..)
            .find_map(|hi| grind_blake::<IS_M31_OUTPUT>(prefixed_digest, hi, pow_bits))
            .expect("Grind failed to find a solution.");

        #[cfg(feature = "parallel")]
        let res = parallel_grind(prefixed_digest, pow_bits, grind_blake::<IS_M31_OUTPUT>);

        assert!(
            ((res >> 32) as u32) < P,
            "The 32 high bits of the nonce are not reduced modulo the M31 prime."
        );
        assert!(
            (res as u32) < P,
            "The 32 low bits of the nonce are not reduced modulo the M31 prime."
        );
        res
    }
}

fn grind_blake<const IS_M31_OUTPUT: bool>(digest: &[u32], hi: u32, pow_bits: u32) -> Option<u64> {
    const DIGEST_SIZE: usize = std::mem::size_of::<[u32; 8]>();
    const NONCE_SIZE: usize = std::mem::size_of::<u64>();
    let zero: u32x16 = u32x16::splat(0);
    let offsets_vec = u32x16::from(std::array::from_fn(|i| i as u32));
    let pow_bits = u32x16::splat(pow_bits);

    let state: [_; 8] = std::array::from_fn(|i| u32x16::splat(digest[i]));

    let mut attempt_low = offsets_vec;
    let attempt_high = u32x16::splat(hi);
    for low in (0..(1 << GRIND_LOW_BITS)).step_by(N_LANES) {
        let msgs = std::array::from_fn(|i| match i {
            0..=7 => state[i],
            8 => attempt_low,
            9 => attempt_high,
            _ => zero,
        });
        let res = hash_16(msgs, (DIGEST_SIZE + NONCE_SIZE) as u64);
        let res0 = if IS_M31_OUTPUT {
            PackedM31::reduce_simd(res[0]).into_simd()
        } else {
            res[0]
        };
        let success_mask = res0.trailing_zeros().simd_ge(pow_bits);
        if success_mask.any() {
            let i = success_mask.to_array().iter().position(|&x| x).unwrap();
            return Some(((hi as u64) << 32) + low as u64 + i as u64);
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
    GRIND: Fn(DIGEST, u32, u32) -> Option<u64> + Send + Sync,
    DIGEST: Send + Sync + Copy,
{
    use core::sync::atomic::Ordering;
    use std::sync::atomic::AtomicU32;

    let n_workers = rayon::current_num_threads() as u32;
    let next_chunk = AtomicU32::new(n_workers);
    let smallest_good_chunk = AtomicU32::new(u32::MAX);
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
        /// Outputs the smallest nonce of the form `(a << 32) | b`, where `0 <= a < 2^31 - 1` and
        /// `0 <= b < 2^GRIND_LOW_BITS`.
        fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
            let digest = channel.digest();
            let prefixed_digest = poseidon_hash_many(&[
                Poseidon252Channel::POW_PREFIX.into(),
                digest,
                pow_bits.into(),
            ]);
            #[cfg(not(feature = "parallel"))]
            let res = (0..)
                .find_map(|hi| grind_poseidon(prefixed_digest, hi, pow_bits))
                .expect("Grind failed to find a solution.");

            #[cfg(feature = "parallel")]
            let res = parallel_grind(prefixed_digest, pow_bits, grind_poseidon);

            assert!(
                ((res >> 32) as u32) < P,
                "The 32 high bits of the solution are larger than the M31 prime."
            );
            res
        }
    }

    fn grind_poseidon(digest: FieldElement252, chunk_id: u32, pow_bits: u32) -> Option<u64> {
        for low in 0..(1 << GRIND_LOW_BITS) {
            let nonce = low | ((chunk_id as u64) << 32);
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
    use crate::core::channel::{Blake2sChannel, Channel};

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
