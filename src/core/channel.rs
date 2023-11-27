use super::fields::m31::{BaseField, P};
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher, FELTS_PER_HASH};
use crate::commitment_scheme::hasher::Hasher;

const N_RETRIES: usize = 10;

pub trait ChannelHash {
    fn hash_with_digest(&mut self, digest: Self);
    fn hash_with_counter(&self, counter: u32) -> Self;

    // Returns random felts and the number of iterations it took to get them.
    fn extract_uniform_felts(&self, counter: u32) -> (usize, [BaseField; FELTS_PER_HASH]);
}

impl ChannelHash for Blake2sHash {
    fn hash_with_digest(&mut self, digest: Blake2sHash) {
        *self = Blake2sHasher::concat_and_hash(self, &digest);
    }

    fn hash_with_counter(&self, counter: u32) -> Self {
        // Pad the counter to 32 bytes.
        let mut padded_counter = [0; 32];
        let counter_bytes = counter.to_le_bytes();
        padded_counter[..counter_bytes.len()].copy_from_slice(&counter_bytes);

        Blake2sHasher::concat_and_hash(self, &Blake2sHash::from(&padded_counter[..]))
    }

    /// Generates a uniform random vector of BaseField elements.
    /// Repeats hashing until getting a good result, such that the function has a
    /// negligible failure probability ~ 2^(-28*N_RETRIES) = 2^-280.
    ///
    /// # Panic
    ///
    /// Panics if it didn't get a good hash result in N_RETRIES rounds.
    fn extract_uniform_felts(&self, counter: u32) -> (usize, [BaseField; FELTS_PER_HASH]) {
        for i in 0..N_RETRIES {
            let hash_res = self.hash_with_counter(counter + i as u32);
            let random_bytes: [u32; FELTS_PER_HASH] = hash_res.into();

            // Retry if not all the u32 are in the range [0, 2P).
            if random_bytes.iter().all(|x| *x < 2 * P) {
                return (
                    i + 1,
                    random_bytes
                        .into_iter()
                        .map(|x| BaseField::reduce(x as u64))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                );
            }
        }
        panic!("Could not find uniform randomness over the base field M31!");
    }
}

pub struct Channel<T: ChannelHash> {
    digest: T,
    counter: u32,
}

impl<T: ChannelHash> Channel<T> {
    pub fn new(digest: T) -> Self {
        Channel { digest, counter: 0 }
    }

    pub fn mix_with_digest(&mut self, seed: T) {
        self.digest.hash_with_digest(seed);
        self.counter = 0;
    }

    pub fn draw_random(&mut self) -> T {
        let rand = self.digest.hash_with_counter(self.counter);
        self.counter += 1;
        rand
    }

    pub fn draw_random_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        let (n_iterations, felts) = self.digest.extract_uniform_felts(self.counter);
        self.counter += n_iterations as u32;
        felts
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::Channel;
    use crate::commitment_scheme::blake2_hash::Blake2sHash;

    #[test]
    fn test_channel() {
        let initial_digest = Blake2sHash::from(vec![0; 32]);
        let mut channel = Channel::new(initial_digest);

        // Assert that the channel is initialized correctly.
        assert_eq!(channel.counter, 0);

        // Draw random elements.
        let x = channel.draw_random();
        assert_eq!(channel.counter, 1);
        channel.draw_random_felts();
        assert!(channel.counter > 1);

        assert_eq!(
            x.to_string(),
            "ae09db7cd54f42b490ef09b6bc541af688e4959bb8c53f359a6f56e38ab454a3"
        );
        assert_eq!(channel.digest, initial_digest);

        // Reseed channel and check the digest was changed.
        channel.mix_with_digest(Blake2sHash::from(vec![1; 32]));
        assert_ne!(initial_digest.to_string(), channel.digest.to_string());
        assert_eq!(channel.counter, 0);
    }

    #[test]
    fn test_uniform_hash() {
        let mut rng = rand::thread_rng();
        let mut channel = Channel::new(Blake2sHash::from(
            (0..32).map(|_| rng.gen::<u8>()).collect::<Vec<_>>(),
        ));
        for _ in 0..1000 {
            channel.draw_random_felts();
        }
    }
}
