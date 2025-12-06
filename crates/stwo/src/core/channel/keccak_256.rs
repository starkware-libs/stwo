use core::{array, iter};

use itertools::Itertools;
use std_shims::Vec;

use super::Channel;
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::vcs::keccak_hash::{Keccak256Hash, Keccak256HasherGeneric};

pub const KECCAK_BYTES_PER_HASH: usize = 32;
pub const FELTS_PER_HASH: usize = 8;

pub type Keccak256Channel = Keccak256ChannelGeneric<false>;
/// Same as [Keccak256Channel], expect that the hash output is taken modulo M31::P.
pub type Keccak256M31Channel = Keccak256ChannelGeneric<true>;

/// A channel that can be used to draw random elements from a [Keccak256Hash] digest.
#[derive(Default, Clone, Debug)]
pub struct Keccak256ChannelGeneric<const IS_M31_OUTPUT: bool> {
    digest: Keccak256Hash,
    n_draws: u32,
}

impl<const IS_M31_OUTPUT: bool> Keccak256ChannelGeneric<IS_M31_OUTPUT> {
    pub const POW_PREFIX: u32 = 0x12345678;

    pub const fn digest(&self) -> Keccak256Hash {
        self.digest
    }
    pub const fn update_digest(&mut self, new_digest: Keccak256Hash) {
        self.digest = new_digest;
        self.n_draws = 0;
    }
    /// Generates a uniform random vector of BaseField elements.
    fn draw_base_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        // Repeats hashing with an increasing counter until getting a good result.
        // Retry probability for each round is ~ 2^(-28).
        loop {
            let u32s: [u32; FELTS_PER_HASH] = self.draw_u32s().try_into().unwrap();

            // Retry if not all the u32 are in the range [0, 2P).
            if u32s.iter().all(|x| *x < 2 * P) {
                return u32s
                    .into_iter()
                    .map(|x| BaseField::reduce(x as u64))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
            }
        }
    }
}

impl<const IS_M31_OUTPUT: bool> Channel for Keccak256ChannelGeneric<IS_M31_OUTPUT> {
    const BYTES_PER_HASH: usize = KECCAK_BYTES_PER_HASH;

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let felts_bytes = felts
            .iter()
            .flat_map(|qm31| qm31.to_m31_array())
            .flat_map(|m31| m31.0.to_le_bytes())
            .collect_vec();
        let mut hasher = Keccak256HasherGeneric::<IS_M31_OUTPUT>::new();
        hasher.update(self.digest.as_ref());
        hasher.update(&felts_bytes);

        self.update_digest(hasher.finalize());
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        let mut hasher = Keccak256HasherGeneric::<IS_M31_OUTPUT>::new();
        hasher.update(self.digest.as_ref());
        for word in data {
            hasher.update(&word.to_le_bytes());
        }

        self.update_digest(hasher.finalize());
    }

    fn mix_u64(&mut self, value: u64) {
        self.mix_u32s(&[value as u32, (value >> 32) as u32])
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

        // Append counter bytes directly (4 bytes for u32).
        let counter_bytes = self.n_draws.to_le_bytes();
        hash_input.extend_from_slice(&counter_bytes);

        // Append a zero byte for domain separation between generating randomness and mixing a
        // single u32.
        hash_input.push(0_u8);

        self.n_draws += 1;
        Keccak256HasherGeneric::<IS_M31_OUTPUT>::hash(&hash_input)
            .0
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    /// Verifies that `H(H(POW_PREFIX, [0_u8; 12], digest, n_bits), nonce)` has at least `n_bits`
    /// many leading zeros.
    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        let digest = self.digest();
        // Compute H(POW_PREFIX, [0_u8; 12], digest, n_bits).
        let mut hasher = Keccak256HasherGeneric::<IS_M31_OUTPUT>::default();
        hasher.update(&Self::POW_PREFIX.to_le_bytes());
        hasher.update(&[0_u8; 12]);
        hasher.update(&digest.0[..]);
        hasher.update(&n_bits.to_le_bytes());
        let prefixed_digest = hasher.finalize();
        // Compute `H(prefixed_digest, nonce)`.
        let mut hasher = Keccak256HasherGeneric::<IS_M31_OUTPUT>::default();
        hasher.update(prefixed_digest.as_ref());
        hasher.update(&nonce.to_le_bytes());
        let res = hasher.finalize();
        let n_zeros = u128::from_le_bytes(array::from_fn(|i| res.0[i])).trailing_zeros();
        n_zeros >= n_bits
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use std_shims::BTreeSet;

    use crate::core::channel::keccak_256::Keccak256Channel;
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

        // Assert that next random words are different.
        assert_ne!(first_random_words, channel.draw_u32s());
    }

    #[test]
    pub fn test_draw_secure_felt() {
        let mut channel = Keccak256Channel::default();

        let first_random_felt = channel.draw_secure_felt();

        // Assert that next random felt is different.
        assert_ne!(first_random_felt, channel.draw_secure_felt());
    }

    #[test]
    pub fn test_draw_secure_felts() {
        let mut channel = Keccak256Channel::default();

        let mut random_felts = channel.draw_secure_felts(5);
        random_felts.extend(channel.draw_secure_felts(4));

        // Assert that all the random felts are unique.
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

    #[test]
    pub fn test_mix_u64() {
        let mut channel = Keccak256Channel::default();
        channel.mix_u64(0x1111222233334444);
        let digest_64 = channel.digest;

        let mut channel = Keccak256Channel::default();
        channel.mix_u32s(&[0x33334444, 0x11112222]);

        assert_eq!(digest_64, channel.digest);
        let digest_bytes: [u8; 32] = digest_64.into();
        assert_eq!(
            digest_bytes,
            [
                0x77, 0xa5, 0x13, 0x97, 0xdd, 0x41, 0x26, 0x43, 0x4a, 0xc7, 0x0d, 0x65, 0x6d, 0xf6,
                0x4a, 0xef, 0xdd, 0xd7, 0x88, 0xbc, 0x47, 0xe7, 0xf3, 0xcc, 0x1d, 0x76, 0xd0, 0xa7,
                0x49, 0x22, 0x36, 0x88
            ]
        )
    }

    #[test]
    pub fn test_mix_u32s() {
        let mut channel = Keccak256Channel::default();
        channel.mix_u32s(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let digest: [u8; 32] = channel.digest.into();
        assert_eq!(
            digest,
            [
                0x1f, 0xc4, 0x72, 0x11, 0x17, 0xdb, 0x54, 0x22, 0xb8, 0x4d, 0x79, 0x96, 0x90, 0xe2,
                0xe4, 0x8b, 0xc8, 0x26, 0x6f, 0xfd, 0x74, 0x0a, 0x6b, 0xf5, 0xd5, 0x54, 0x7f, 0xdc,
                0x49, 0x1c, 0x86, 0x8c
            ]
        );
    }
}

