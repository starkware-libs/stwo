use super::fields::m31::{BaseField, N_BYTES_FELT, P};
use super::fields::qm31::{QM31, QM31_EXTENSION_DEGREE};
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::hasher::Hasher;

pub const BLAKE_BYTES_PER_HASH: usize = 32;
pub const FELTS_PER_HASH: usize = 8;
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
    type ChannelHasher: Hasher;
    const BYTES_PER_HASH: usize;

    fn new(digest: <Self::ChannelHasher as Hasher>::Hash) -> Self;
    fn get_digest(&self) -> <Self::ChannelHasher as Hasher>::Hash;
    fn mix_with_seed(&mut self, seed: <Self::ChannelHasher as Hasher>::Hash);
    fn draw_random_felts(&mut self) -> [BaseField; FELTS_PER_HASH];
    /// Returns a vector of random bytes of length `BYTES_PER_HASH`.
    fn draw_random_bytes(&mut self) -> Vec<u8>;
    /// Generates a uniform random vector of QM31 elements.
    fn draw_random_extension_felts(&mut self) -> [QM31; EXTENSION_FELTS_PER_HASH] {
        let felts: [BaseField; FELTS_PER_HASH] = self.draw_random_felts();
        [
            QM31::from_m31_array(felts[..QM31_EXTENSION_DEGREE].try_into().unwrap()),
            QM31::from_m31_array(felts[QM31_EXTENSION_DEGREE..].try_into().unwrap()),
        ]
    }
}

/// A channel that can be used to draw random elements from a [Blake2sHash] digest.
pub struct Blake2sChannel {
    digest: Blake2sHash,
    channel_time: ChannelTime,
}

impl Channel for Blake2sChannel {
    type ChannelHasher = Blake2sHasher;
    const BYTES_PER_HASH: usize = BLAKE_BYTES_PER_HASH;

    fn new(digest: <Self::ChannelHasher as Hasher>::Hash) -> Self {
        Blake2sChannel {
            digest,
            channel_time: ChannelTime::default(),
        }
    }

    fn get_digest(&self) -> <Self::ChannelHasher as Hasher>::Hash {
        self.digest
    }

    fn mix_with_seed(&mut self, seed: <Self::ChannelHasher as Hasher>::Hash) {
        self.digest = Self::ChannelHasher::concat_and_hash(&self.digest, &seed);
        self.channel_time.inc_challenges();
    }

    /// Generates a uniform random vector of BaseField elements.
    fn draw_random_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        // Repeats hashing with an increasing counter until getting a good result.
        // Retry probability for each round is ~ 2^(-28).
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

    fn draw_random_bytes(&mut self) -> Vec<u8> {
        let mut hash_input = self.digest.as_ref().to_vec();

        // Pad the counter to 32 bytes.
        let mut padded_counter = [0; BLAKE_BYTES_PER_HASH];
        let counter_bytes = self.channel_time.n_sent.to_le_bytes();
        padded_counter[0..counter_bytes.len()].copy_from_slice(&counter_bytes);

        hash_input.extend_from_slice(&padded_counter);

        self.channel_time.inc_sent();
        Self::ChannelHasher::hash(&hash_input).into()
    }
}

#[cfg(test)]
mod tests {

    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::channel::{Blake2sChannel, Channel};

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

        channel.draw_random_felts();
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 2);

        channel.mix_with_seed(Blake2sHash::from(vec![1; 32]));
        assert_eq!(channel.channel_time.n_challenges, 1);
        assert_eq!(channel.channel_time.n_sent, 0);

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
    }

    #[test]
    pub fn test_draw_random_felts() {
        let initial_digest = Blake2sHash::from(vec![2; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        let first_random_felts = channel.draw_random_felts();

        // Assert that the next random felts are different.
        assert_ne!(first_random_felts, channel.draw_random_felts());
    }

    #[test]
    pub fn test_mix_with_seed() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        for _ in 0..10 {
            channel.draw_random_bytes();
            channel.draw_random_felts();
        }

        // Reseed channel and check the digest was changed.
        channel.mix_with_seed(Blake2sHash::from(vec![1; 32]));
        assert_ne!(initial_digest, channel.digest);
    }
}
