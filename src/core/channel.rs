use super::fields::m31::{BaseField, N_BYTES_FELT, P};
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::hasher::Hasher;

pub const FELTS_PER_HASH: usize = 8;

/// A channel that can be used to draw random elements from a [Blake2sHash] digest.
pub struct Blake2sChannel {
    digest: Blake2sHash,
    counter: u32,
}

impl Blake2sChannel {
    pub fn new(digest: Blake2sHash) -> Self {
        Blake2sChannel { digest, counter: 0 }
    }

    pub fn mix_with_seed(&mut self, seed: Blake2sHash) {
        self.digest = Blake2sHasher::concat_and_hash(&self.digest, &seed);
        self.counter = 0;
    }
    /// Generates uniform 32 bytes and increases the channel counter by 1.
    pub fn draw_random_bytes(&mut self) -> [u8; 32] {
        let mut hash_input = self.digest.as_ref().to_vec();

        // Pad the counter to 32 bytes.
        let mut padded_counter = [0; 32];
        let counter_bytes = self.counter.to_le_bytes();
        padded_counter[0..counter_bytes.len()].copy_from_slice(&counter_bytes);
        hash_input.extend_from_slice(&padded_counter);

        self.counter += 1;
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
        assert_eq!(channel.counter, 0);
    }

    #[test]
    fn test_counter() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);
        assert_eq!(channel.counter, 0);

        channel.draw_random_bytes();
        assert_eq!(channel.counter, 1);

        channel.draw_random_felts();
        assert_eq!(channel.counter, 2);

        channel.mix_with_seed(Blake2sHash::from(vec![1; 32]));
        assert_eq!(channel.counter, 0);
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
