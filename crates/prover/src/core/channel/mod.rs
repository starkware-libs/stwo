use std::fmt::Debug;

use super::fields::qm31::SecureField;
use super::vcs::ops::MerkleHasher;

#[cfg(not(target_arch = "wasm32"))]
mod poseidon252;
#[cfg(not(target_arch = "wasm32"))]
pub use poseidon252::Poseidon252Channel;

mod blake2s;
pub use blake2s::Blake2sChannel;

pub mod logging_channel;

pub const EXTENSION_FELTS_PER_HASH: usize = 2;

#[derive(Clone, Default, Debug)]
pub struct ChannelTime {
    pub n_challenges: usize,
    n_sent: usize,
}

impl ChannelTime {
    const fn inc_sent(&mut self) {
        self.n_sent += 1;
    }

    const fn inc_challenges(&mut self) {
        self.n_challenges += 1;
        self.n_sent = 0;
    }
}

pub trait Channel: Default + Clone + Debug {
    const BYTES_PER_HASH: usize;

    fn trailing_zeros(&self) -> u32;

    // Mix functions.
    fn mix_u32s(&mut self, data: &[u32]);
    fn mix_felts(&mut self, felts: &[SecureField]);
    fn mix_u64(&mut self, value: u64);

    // Draw functions.
    fn draw_felt(&mut self) -> SecureField;
    /// Generates a uniform random vector of SecureField elements.
    fn draw_felts(&mut self, n_felts: usize) -> Vec<SecureField>;
    /// Returns a vector of random bytes of length `BYTES_PER_HASH`.
    fn draw_random_bytes(&mut self) -> Vec<u8>;
}

pub trait MerkleChannel: Default {
    type C: Channel;
    type H: MerkleHasher;
    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash);
}
