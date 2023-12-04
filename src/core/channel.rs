use super::fields::m31::{BaseField, P};
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher, FELTS_PER_HASH};
use crate::commitment_scheme::hasher::Hasher;

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

    /// Generates a uniform random [Blake2sHash] and increase the channel counter by 1.
    pub fn draw_random_bytes(&mut self) -> Blake2sHash {
        let mut padded_counter = [0; 32];

        // Pad the counter to 32 bytes.
        let counter_bytes = self.counter.to_le_bytes();
        padded_counter[..counter_bytes.len()].copy_from_slice(&counter_bytes);
        self.counter += 1;

        Blake2sHasher::concat_and_hash(&self.digest, &Blake2sHash::from(&padded_counter[..]))
    }

    /// Generates a uniform random vector of BaseField elements.
    /// Repeats hashing with an increasing counter until getting a good result.
    /// Retry probablity for each round is ~ 2^(-28).
    pub fn draw_random_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        loop {
            let random_bytes: [u32; FELTS_PER_HASH] = self.draw_random_bytes().into();

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
    use rand::Rng;

    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::channel::Blake2sChannel;

    #[test]
    fn test_channel() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let mut channel = Blake2sChannel::new(initial_digest);

        // Assert that the channel is initialized correctly.
        assert_eq!(channel.counter, 0);

        // Draw random elements.
        let x = channel.draw_random_bytes();
        assert_eq!(channel.counter, 1);
        channel.draw_random_felts();
        assert_eq!(channel.counter, 2);

        assert_eq!(
            x.to_string(),
            "ae09db7cd54f42b490ef09b6bc541af688e4959bb8c53f359a6f56e38ab454a3"
        );
        assert_eq!(channel.digest, initial_digest);

        // Reseed channel and check the digest was changed.
        channel.mix_with_seed(Blake2sHash::from(vec![1; 32]));
        assert_ne!(initial_digest.to_string(), channel.digest.to_string());
        assert_eq!(channel.counter, 0);
    }

    #[test]
    fn test_uniform_hash_success() {
        let mut rng = rand::thread_rng();
        let mut channel = Blake2sChannel::new(Blake2sHash::from(
            (0..32).map(|_| rng.gen::<u8>()).collect::<Vec<_>>(),
        ));
        for _ in 0..1000 {
            channel.draw_random_felts();
        }
    }
}
