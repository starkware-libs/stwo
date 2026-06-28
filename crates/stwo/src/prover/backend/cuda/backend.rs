use bytemuck::cast_slice;
use serde::{Deserialize, Serialize};
use tracing::{span, Level};

use crate::core::channel::Blake2sChannelGeneric;
use crate::core::vcs::blake2_hash::Blake2sHasherGeneric;
use crate::prover::backend::{Backend, BackendForChannel, simd::SimdBackend};
use crate::core::{
    channel::{Blake2sChannel, Blake2sM31Channel, Poseidon252Channel},
    proof_of_work::GrindOps,
    vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sM31MerkleChannel},
    vcs_lifted::poseidon252_merkle::Poseidon252MerkleChannel,
};
use crate::stwo_cuda::bindings;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}

impl GrindOps<Blake2sChannel> for CudaBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        let _span = span!(Level::TRACE, "CUDA Blake2s Grind", class = "Blake2s Grind");

        assert!(pow_bits <= 32, "pow_bits > 32 is not supported");
        let digest = channel.digest();

        // Compute prefixed_digest = H(POW_PREFIX, [0; 12], digest, pow_bits) on CPU
        let mut hasher = Blake2sHasherGeneric::<false>::default();
        hasher.update(&Blake2sChannelGeneric::<false>::POW_PREFIX.to_le_bytes());
        hasher.update(&[0_u8; 12]);
        hasher.update(&digest.0[..]);
        hasher.update(&pow_bits.to_le_bytes());
        let prefixed_digest = hasher.finalize();
        let prefixed_digest_u32: &[u32] = cast_slice(&prefixed_digest.0[..]);

        // Call CUDA kernel for GPU-accelerated grinding
        unsafe { bindings::grind_blake2s(prefixed_digest_u32.as_ptr(), pow_bits) }
    }
}

impl BackendForChannel<Blake2sMerkleChannel> for CudaBackend {}

impl GrindOps<Blake2sM31Channel> for CudaBackend {
    fn grind(channel: &Blake2sM31Channel, pow_bits: u32) -> u64 {
        SimdBackend::grind(channel, pow_bits)
    }
}

impl BackendForChannel<Blake2sM31MerkleChannel> for CudaBackend {}

impl GrindOps<Poseidon252Channel> for CudaBackend {
    fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
        SimdBackend::grind(channel, pow_bits)
    }
}

impl BackendForChannel<Poseidon252MerkleChannel> for CudaBackend {}
