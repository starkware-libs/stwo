use super::fields::m31::{BaseField, N_BYTES_FELT, P};
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::hasher::Hasher;

pub const FELTS_PER_HASH: usize = 8;

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

/// A channel that can be used to draw random elements from a [Blake2sHash] digest.
pub struct Blake2sChannel {
    digest: Blake2sHash,
    channel_time: ChannelTime,
}

impl Blake2sChannel {
    pub fn new(digest: Blake2sHash) -> Self {
        Blake2sChannel {
            digest,
            channel_time: ChannelTime::default(),
        }
    }

    pub fn mix_with_seed(&mut self, seed: Blake2sHash) {
        self.digest = Blake2sHasher::concat_and_hash(&self.digest, &seed);
        self.channel_time.inc_challenges();
    }
    /// Generates uniform 32 bytes and increases the channel counter by 1.
    pub fn draw_random_bytes(&mut self) -> [u8; 32] {
        let mut hash_input = self.digest.as_ref().to_vec();

        // Pad the counter to 32 bytes.
        let mut padded_counter = [0; 32];
        let counter_bytes = self.channel_time.n_sent.to_le_bytes();
        padded_counter[0..counter_bytes.len()].copy_from_slice(&counter_bytes);
        hash_input.extend_from_slice(&padded_counter);

        self.channel_time.inc_sent();
        Blake2sHasher::hash(&hash_input).into()
    }

    /// Generates a uniform random vector of BaseField elements.
    /// Repeats hashing with an increasing counter until getting a good result.
    /// Retry probablity for each round is ~ 2^(-28).
    pub fn draw_random_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        loop {
            let random_bytes: [u32; FELTS_PER_HASH] = self
                .draw_random_bytes()
                .chunks_exact(N_BYTES_FELT) // 4 bytes per u32.
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            // Retry if not all the u32 are in the range [0, 2P).
            if random_bytes.iter().all(|x| *x < 2 * P) {
                return random_bytes
                    .into_iter()
                    .map(|x| BaseField::reduce(x as u64))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::channel::Blake2sChannel;

    #[test]
    fn test_initliaze_channel() {
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

        // Assert that the channel time is initialized correctly.
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 0);

        // Assert that the channel time is updated correctly.
        channel.draw_random_bytes();
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 1);

        channel.mix_with_seed(Blake2sHash::from(vec![1; 32]));
        assert_eq!(channel.channel_time.n_challenges, 1);
        assert_eq!(channel.channel_time.n_sent, 0);

        // Assert that the channel time is updated correctly.
        channel.draw_random_felts();
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

        // Assert that the digest is not changed.
        assert_eq!(channel.digest, initial_digest);
    }

    #[test]
    pub fn test_get_random_felts() {
        let initial_digest = Blake2sHash::from(vec![2; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        let first_random_felts = channel.draw_random_felts();

        // Assert that the next random felts are different.
        assert_ne!(first_random_felts, channel.draw_random_felts());

        // Assert that the digest is not changed.
        assert_eq!(channel.digest, initial_digest);
    }

    #[test]
    pub fn test_mix_with_seed() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        // Reseed channel and check the digest was changed.
        channel.mix_with_seed(Blake2sHash::from(vec![1; 32]));

        // Assert that the digest is not changed.
        assert_ne!(channel.digest, initial_digest);
    }
}
