use core::{array, iter};

use itertools::Itertools;
use std_shims::Vec;

use super::Channel;
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::vcs::keccak256_hash::{Keccak256Hash, Keccak256Hasher};

pub const KECCAK_BYTES_PER_HASH: usize = 32;
pub const FELTS_PER_HASH: usize = 8;

/// A Fiat-Shamir channel backed by `keccak256`.
///
/// All multi-byte integer inputs are big-endian, matching Solidity's
/// `abi.encodePacked` conventions so that an on-chain Solidity verifier can
/// reproduce the transcript without byte-order translation.
#[derive(Default, Clone, Debug)]
pub struct Keccak256Channel {
    digest: Keccak256Hash,
    n_draws: u32,
}

impl Keccak256Channel {
    pub const POW_PREFIX: u32 = 0x12345678;

    pub const fn digest(&self) -> Keccak256Hash {
        self.digest
    }

    pub const fn update_digest(&mut self, new_digest: Keccak256Hash) {
        self.digest = new_digest;
        self.n_draws = 0;
    }

    /// Generates a uniform random vector of BaseField elements via rejection
    /// sampling. Retry probability per round is ~2^-28.
    fn draw_base_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        loop {
            let u32s: [u32; FELTS_PER_HASH] = self.draw_u32s().try_into().unwrap();

            if u32s.iter().all(|x| *x < 2 * P) {
                return u32s
                    .into_iter()
                    .map(BaseField::partial_reduce)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
            }
        }
    }
}

impl Channel for Keccak256Channel {
    const BYTES_PER_HASH: usize = KECCAK_BYTES_PER_HASH;

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let felts_bytes = felts
            .iter()
            .flat_map(|qm31| qm31.to_m31_array())
            .flat_map(|m31| m31.0.to_be_bytes())
            .collect_vec();
        let mut hasher = Keccak256Hasher::new();
        hasher.update(self.digest.as_ref());
        hasher.update(&felts_bytes);

        self.update_digest(hasher.finalize());
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        let mut hasher = Keccak256Hasher::new();
        hasher.update(self.digest.as_ref());
        for word in data {
            hasher.update(&word.to_be_bytes());
        }

        self.update_digest(hasher.finalize());
    }

    fn mix_u64(&mut self, value: u64) {
        self.mix_u32s(&[(value >> 32) as u32, value as u32])
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        let felts: [BaseField; FELTS_PER_HASH] = self.draw_base_felts();
        SecureField::from_m31_array(felts[..SECURE_EXTENSION_DEGREE].try_into().unwrap())
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        let mut felts = iter::from_fn(|| Some(self.draw_base_felts())).flatten();
        let secure_felts = iter::from_fn(|| {
            Some(SecureField::from_m31_array([
                felts.next()?,
                felts.next()?,
                felts.next()?,
                felts.next()?,
            ]))
        });
        secure_felts.take(n_felts).collect()
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        let mut hash_input = self.digest.as_ref().to_vec();

        // Append BE counter bytes (4 bytes for u32).
        hash_input.extend_from_slice(&self.n_draws.to_be_bytes());

        // Domain separation byte between the draw and mix-single-u32 paths.
        hash_input.push(0_u8);

        self.n_draws += 1;
        Keccak256Hasher::hash(&hash_input)
            .0
            .chunks_exact(4)
            .map(|chunk| u32::from_be_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    /// Verifies that `H(H(POW_PREFIX_BE || [0u8;12] || digest || n_bits_BE) || nonce_BE)` has at
    /// least `n_bits` leading zero bits when its first 16 bytes are interpreted as a big-endian
    /// u128. Equivalent to Solidity's `uint256(hash) < 2**(256 - n_bits)` for `n_bits <= 128`.
    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        let digest = self.digest();
        let mut hasher = Keccak256Hasher::default();
        hasher.update(&Self::POW_PREFIX.to_be_bytes());
        hasher.update(&[0_u8; 12]);
        hasher.update(&digest.0[..]);
        hasher.update(&n_bits.to_be_bytes());
        let prefixed_digest = hasher.finalize();

        let mut hasher = Keccak256Hasher::default();
        hasher.update(prefixed_digest.as_ref());
        hasher.update(&nonce.to_be_bytes());
        let res = hasher.finalize();

        let n_zeros = u128::from_be_bytes(array::from_fn(|i| res.0[i])).leading_zeros();
        n_zeros >= n_bits
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use std_shims::BTreeSet;

    use crate::core::channel::keccak256::Keccak256Channel;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::m31;

    #[test]
    fn test_channel_draws() {
        let mut channel = Keccak256Channel::default();

        assert_eq!(channel.n_draws, 0);

        channel.draw_u32s();
        assert_eq!(channel.n_draws, 1);

        channel.draw_secure_felts(9);
        assert_eq!(channel.n_draws, 6);
    }

    #[test]
    fn test_draw_u32s() {
        let mut channel = Keccak256Channel::default();

        let first_random_words = channel.draw_u32s();

        assert_ne!(first_random_words, channel.draw_u32s());
    }

    #[test]
    pub fn test_draw_secure_felt() {
        let mut channel = Keccak256Channel::default();

        let first_random_felt = channel.draw_secure_felt();

        assert_ne!(first_random_felt, channel.draw_secure_felt());
    }

    #[test]
    pub fn test_draw_secure_felts() {
        let mut channel = Keccak256Channel::default();

        let mut random_felts = channel.draw_secure_felts(5);
        random_felts.extend(channel.draw_secure_felts(4));

        assert_eq!(
            random_felts.len(),
            random_felts.iter().collect::<BTreeSet<_>>().len()
        );
    }

    #[test]
    pub fn test_mix_felts() {
        let mut channel = Keccak256Channel::default();
        let initial_digest = channel.digest;
        let felts = (0..2)
            .map(|i| SecureField::from(m31!(i + 1923782)))
            .collect_vec();

        channel.mix_felts(felts.as_slice());

        assert_ne!(initial_digest, channel.digest);
    }

    /// Checks that `mix_u64(value)` equals
    /// `keccak256(abi.encodePacked(prev_digest, uint64(value)))`, where
    /// `prev_digest` is the all-zero default initial digest, so that an
    /// on-chain Solidity verifier produces the same transcript.
    #[test]
    pub fn test_mix_u64_solidity_equivalence() {
        let mut channel = Keccak256Channel::default();
        channel.mix_u64(0x1111222233334444);

        // Reproduce by directly hashing the equivalent Solidity input:
        //   bytes.concat(bytes32(0), bytes8(0x1111222233334444))
        use sha3::{Digest, Keccak256};
        let mut hasher = Keccak256::new();
        hasher.update([0u8; 32]); // initial digest
        hasher.update(0x1111222233334444u64.to_be_bytes());
        let expected: [u8; 32] = hasher.finalize().into();

        let actual: [u8; 32] = channel.digest.into();
        assert_eq!(actual, expected);
    }

    /// Checks that `mix_u32s(...)` equals
    /// `keccak256(abi.encodePacked(prev_digest, uint32(...), ...))`, so that
    /// an on-chain Solidity verifier produces the same transcript.
    #[test]
    pub fn test_mix_u32s_solidity_equivalence() {
        let mut channel = Keccak256Channel::default();
        channel.mix_u32s(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);

        use sha3::{Digest, Keccak256};
        let mut hasher = Keccak256::new();
        hasher.update([0u8; 32]);
        for w in [1u32, 2, 3, 4, 5, 6, 7, 8, 9] {
            hasher.update(w.to_be_bytes());
        }
        let expected: [u8; 32] = hasher.finalize().into();

        let actual: [u8; 32] = channel.digest.into();
        assert_eq!(actual, expected);
    }
}
