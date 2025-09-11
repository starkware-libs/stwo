use core::fmt::Debug;

use std_shims::Vec;

use super::fields::qm31::SecureField;
use super::vcs::MerkleHasher;

#[cfg(not(target_arch = "wasm32"))]
mod poseidon252;
#[cfg(not(target_arch = "wasm32"))]
pub use poseidon252::Poseidon252Channel;

mod blake2s;
pub use blake2s::Blake2sChannel;

pub const EXTENSION_FELTS_PER_HASH: usize = 2;

pub trait Channel: Default + Clone + Debug {
    const BYTES_PER_HASH: usize;

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool;

    // Mix functions.
    fn mix_u32s(&mut self, data: &[u32]);
    fn mix_felts(&mut self, felts: &[SecureField]);
    fn mix_u64(&mut self, value: u64);

    // Draw functions.
    fn draw_secure_felt(&mut self) -> SecureField;
    /// Generates a uniform random vector of SecureField elements.
    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField>;
    /// Returns a vector of random u32s.
    ///
    /// The length of this vector depends on the channel's hash function.
    /// For blake2s channel, the length of the returned vector is 8
    /// while for poseidon channel, the length is 7.
    fn draw_u32s(&mut self) -> Vec<u32>;
}

pub trait MerkleChannel: Default {
    type C: Channel;
    type H: MerkleHasher;
    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash);
}
