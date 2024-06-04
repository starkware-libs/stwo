use std::iter;

use super::{Channel, ChannelTime};
use crate::core::fields::m31::{BaseField, N_BYTES_FELT, P};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::fields::IntoSlice;
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::core::vcs::hasher::Hasher;

pub const BLAKE_BYTES_PER_HASH: usize = 32;
pub const FELTS_PER_HASH: usize = 8;

/// A channel that can be used to draw random elements from a [Blake2sHash] digest.
pub struct Blake2sChannel {
    digest: Blake2sHash,
    channel_time: ChannelTime,
}

impl Blake2sChannel {
    /// Generates a uniform random vector of BaseField elements.
    fn draw_base_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        // Repeats hashing with an increasing counter until getting a good result.
        // Retry probability for each round is ~ 2^(-28).
        loop {
            let u32s: [u32; FELTS_PER_HASH] = self
                .draw_random_bytes()
                .chunks_exact(N_BYTES_FELT) // 4 bytes per u32.
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

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

impl Channel for Blake2sChannel {
    type Digest = Blake2sHash;
    const BYTES_PER_HASH: usize = BLAKE_BYTES_PER_HASH;

    fn new(digest: Self::Digest) -> Self {
        Blake2sChannel {
            digest,
            channel_time: ChannelTime::default(),
        }
    }

    fn get_digest(&self) -> Self::Digest {
        self.digest
    }

    fn mix_digest(&mut self, digest: Self::Digest) {
        self.digest = Blake2sHasher::concat_and_hash(&self.digest, &digest);
        self.channel_time.inc_challenges();
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let mut hasher = Blake2sHasher::new();
        hasher.update(self.digest.as_ref());
        hasher.update(IntoSlice::<u8>::into_slice(felts));

        self.digest = hasher.finalize();
        self.channel_time.inc_challenges();
    }

    fn mix_nonce(&mut self, nonce: u64) {
        // Copy the elements from the original array to the new array
        let mut padded_nonce = vec![0; BLAKE_BYTES_PER_HASH];
        padded_nonce[..8].copy_from_slice(&nonce.to_le_bytes());

        self.digest =
            Blake2sHasher::concat_and_hash(&self.digest, &Blake2sHash::from(padded_nonce));
        self.channel_time.inc_challenges();
    }

    fn draw_felt(&mut self) -> SecureField {
        let felts: [BaseField; FELTS_PER_HASH] = self.draw_base_felts();
        SecureField::from_m31_array(felts[..SECURE_EXTENSION_DEGREE].try_into().unwrap())
    }

    fn draw_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
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

    fn draw_random_bytes(&mut self) -> Vec<u8> {
        let mut hash_input = self.digest.as_ref().to_vec();

        // Pad the counter to 32 bytes.
        let mut padded_counter = [0; BLAKE_BYTES_PER_HASH];
        let counter_bytes = self.channel_time.n_sent.to_le_bytes();
        padded_counter[0..counter_bytes.len()].copy_from_slice(&counter_bytes);

        hash_input.extend_from_slice(&padded_counter);

        // TODO(spapini): Are we worried about this drawing hash colliding with mix_digest?

        self.channel_time.inc_sent();
        Blake2sHasher::hash(&hash_input).into()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::core::channel::blake2s::Blake2sChannel;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::core::vcs::blake2_hash::Blake2sHash;
    use crate::m31;

    #[test]
    fn test_initialize_channel() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let channel = Blake2sChannel::new(initial_digest);

        // Assert that the channel is initialized correctly.
        assert_eq!(channel.digest, initial_digest);
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 0);
    }

    #[test]
    fn test_channel_time() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 0);

        channel.draw_random_bytes();
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 1);

        channel.draw_felts(9);
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 6);

        channel.mix_digest(Blake2sHash::from(vec![1; 32]));
        assert_eq!(channel.channel_time.n_challenges, 1);
        assert_eq!(channel.channel_time.n_sent, 0);

        channel.draw_felt();
        assert_eq!(channel.channel_time.n_challenges, 1);
        assert_eq!(channel.channel_time.n_sent, 1);
        assert_ne!(channel.digest, initial_digest);
    }

    #[test]
    fn test_draw_random_bytes() {
        let initial_digest = Blake2sHash::from(vec![1; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        let first_random_bytes = channel.draw_random_bytes();

        // Assert that next random bytes are different.
        assert_ne!(first_random_bytes, channel.draw_random_bytes());
    }

    #[test]
    pub fn test_draw_felt() {
        let initial_digest = Blake2sHash::from(vec![2; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        let first_random_felt = channel.draw_felt();

        // Assert that next random felt is different.
        assert_ne!(first_random_felt, channel.draw_felt());
    }

    #[test]
    pub fn test_draw_felts() {
        let initial_digest = Blake2sHash::from(vec![2; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        let mut random_felts = channel.draw_felts(5);
        random_felts.extend(channel.draw_felts(4));

        // Assert that all the random felts are unique.
        assert_eq!(
            random_felts.len(),
            random_felts.iter().collect::<BTreeSet<_>>().len()
        );
    }

    #[test]
    pub fn test_mix_digest() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        for _ in 0..10 {
            channel.draw_random_bytes();
            channel.draw_felt();
        }

        // Reseed channel and check the digest was changed.
        channel.mix_digest(Blake2sHash::from(vec![1; 32]));
        assert_ne!(initial_digest, channel.digest);
    }

    #[test]
    pub fn test_mix_felts() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);
        let felts: Vec<SecureField> = (0..2)
            .map(|i| SecureField::from(m31!(i + 1923782)))
            .collect();

        channel.mix_felts(felts.as_slice());

        assert_ne!(initial_digest, channel.digest);
    }
}
