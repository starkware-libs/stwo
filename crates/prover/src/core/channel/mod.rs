use super::fields::qm31::SecureField;

mod blake2s;
#[cfg(not(target_arch = "wasm32"))]
mod poseidon252;

pub use blake2s::Blake2sChannel;

pub const EXTENSION_FELTS_PER_HASH: usize = 2;

#[derive(Default)]
pub struct ChannelTime {
    n_challenges: usize,
    n_sent: usize,
}

impl ChannelTime {
    fn inc_sent(&mut self) {
        self.n_sent += 1;
    }

    fn inc_challenges(&mut self) {
        self.n_challenges += 1;
        self.n_sent = 0;
    }
}

pub trait Channel {
    type Digest;

    const BYTES_PER_HASH: usize;

    fn new(digest: Self::Digest) -> Self;
    fn get_digest(&self) -> Self::Digest;

    // Mix functions.
    fn mix_digest(&mut self, digest: Self::Digest);
    fn mix_felts(&mut self, felts: &[SecureField]);
    fn mix_nonce(&mut self, nonce: u64);

    // Draw functions.
    fn draw_felt(&mut self) -> SecureField;
    /// Generates a uniform random vector of SecureField elements.
    fn draw_felts(&mut self, n_felts: usize) -> Vec<SecureField>;
    /// Returns a vector of random bytes of length `BYTES_PER_HASH`.
    fn draw_random_bytes(&mut self) -> Vec<u8>;
}
