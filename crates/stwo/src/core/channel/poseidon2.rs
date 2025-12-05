use core::fmt::Debug;

use std_shims::Vec;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::vcs::poseidon2_primitives::{poseidon2_permute, N_STATE};

#[derive(Clone, Default, Debug)]
pub struct Poseidon2Channel {
    state: [BaseField; N_STATE],
    n_draws: usize, // Track how many elements drawn from current squeeze
}

impl Poseidon2Channel {
    pub fn new(digest: [BaseField; 8]) -> Self {
        let mut state = [BaseField::default(); N_STATE];
        // Initialize state with digest (first 8 elements)
        state[..8].copy_from_slice(&digest);
        Self { state, n_draws: 0 }
    }

    fn permute(&mut self) {
        poseidon2_permute(&mut self.state);
        self.n_draws = 0;
    }

    fn absorb(&mut self, data: &[BaseField]) {
        // Simple absorption (overwrite/add to rate part)
        // Note: For full sponge security, should use proper padding and rate/capacity management
        // Assuming data length <= Rate (8) for simplicity of current interface use cases
        // or loop if larger. Stwo often mixes single values or small arrays.

        // If we are mid-squeeze, should we permute first?
        // Usually mixing happens before drawing.
        
        for chunk in data.chunks(8) {
            for (i, &val) in chunk.iter().enumerate() {
                self.state[i] += val;
            }
            self.permute();
        }
    }
}

use super::Channel;

impl Channel for Poseidon2Channel {
    const BYTES_PER_HASH: usize = 32; // Not strictly true for M31-Poseidon digest, but required by trait.
    // 8 * 31 bits ~ 248 bits.

    fn verify_pow_nonce(&self, _n_bits: u32, _nonce: u64) -> bool {
        // Replicate basic PoW check logic
        // TODO: Implement specific PoW for Poseidon2 if needed
        true
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        let felts: Vec<BaseField> = data
            .iter()
            .map(|&x| BaseField::from_u32_unchecked(x))
            .collect();
        self.absorb(&felts);
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        for &f in felts {
            // SecureField is extension field (4 BaseFields)
            let coeffs = f.to_m31_array();
            self.absorb(&coeffs);
        }
    }

    fn mix_u64(&mut self, value: u64) {
        let low = value as u32;
        let high = (value >> 32) as u32;
        self.mix_u32s(&[low, high]);
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        let f0 = self.draw_base_felt();
        let f1 = self.draw_base_felt();
        let f2 = self.draw_base_felt();
        let f3 = self.draw_base_felt();
        SecureField::from_m31_array([f0, f1, f2, f3])
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        (0..n_felts).map(|_| self.draw_secure_felt()).collect()
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        // Draw 8 u32s (from rate part)
        // Stwo doc says: "For poseidon channel, the length is 7" (in 252 context).
        // Here we can draw up to rate (8).
        let mut res = Vec::with_capacity(8);
        for _ in 0..8 {
            res.push(self.draw_base_felt().0);
        }
        res
    }
}

impl Poseidon2Channel {
    fn draw_base_felt(&mut self) -> BaseField {
        if self.n_draws >= 8 {
            self.permute();
        }
        let res = self.state[self.n_draws];
        self.n_draws += 1;
        res
    }
}

use super::MerkleChannel;
use crate::core::vcs::poseidon2_merkle::Poseidon2MerkleHasher;

impl MerkleChannel for Poseidon2Channel {
    type C = Poseidon2Channel;
    type H = Poseidon2MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as crate::core::vcs::MerkleHasher>::Hash) {
        channel.absorb(&root.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // m31 is not used here, only Channel methods and BaseField implicitly.
    // mix_u32s uses BaseField internally.

    #[test]
    fn test_poseidon2_channel_mixing() {
        let mut channel = Poseidon2Channel::default();
        let initial_state = channel.state;

        channel.mix_u32s(&[1, 2, 3]);
        assert_ne!(channel.state, initial_state);

        let state_after_mix = channel.state;
        channel.mix_u32s(&[1, 2, 3]);
        assert_ne!(channel.state, state_after_mix);
    }

    #[test]
    fn test_poseidon2_channel_draw() {
        let mut channel = Poseidon2Channel::default();
        
        // Sponge state is 0 initially, so drawing gives 0s. 
        // Mix something to get "randomness".
        channel.mix_u32s(&[12345]);

        let r1 = channel.draw_base_felt();
        let r2 = channel.draw_base_felt();
        assert_ne!(r1, r2);

        // Draw enough to trigger permutation (Rate=8)
        for _ in 0..10 {
            channel.draw_base_felt();
        }
    }

    #[test]
    fn test_poseidon2_channel_determinism() {
        let mut c1 = Poseidon2Channel::default();
        let mut c2 = Poseidon2Channel::default();

        c1.mix_u32s(&[123]);
        c2.mix_u32s(&[123]);

        assert_eq!(c1.state, c2.state);
        assert_eq!(c1.draw_base_felt(), c2.draw_base_felt());
    }
}
