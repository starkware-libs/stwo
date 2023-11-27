use super::fields::m31::{BaseField, P};
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::hasher::Hasher;

pub trait ChannelHash {
    fn hash_with_digest(&mut self, digest: Self);
    fn hash_with_counter(&self, counter: u32) -> Self;
    fn hash_uniform_felts(&mut self, counter: u32) -> Vec<BaseField>;
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
    /// negligible failure probability.
    ///
    /// # Panic
    ///
    /// Panics if it didn't got a good hash result in 10 rounds.
    fn hash_uniform_felts(&mut self, counter: u32) -> Vec<BaseField> {
        let mut hash_res = self.hash_with_counter(counter);
        for _ in 0..10 {
            let random_bytes: Vec<u32> = hash_res
                .as_ref()
                .chunks_exact(4) // 4 bytes per u32.
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            // Retry if not all the u32 are in the range [0, 2P).
            if random_bytes.iter().all(|x| *x < 2 * P) {
                return random_bytes
                    .iter()
                    .map(|x| BaseField::reduce(*x as u64))
                    .collect();
            }
            hash_res = hash_res.hash_with_counter(counter);
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

    pub fn draw_random_felts(&mut self) -> Vec<BaseField> {
        let felts = self.digest.hash_uniform_felts(self.counter);
        self.counter += 1;
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
        let mut channel = Channel::new(Blake2sHash::from(vec![0; 32]));

        // Assert that the channel is initialized correctly.
        assert_eq!(channel.counter, 0);
        assert_eq!(channel.digest.as_ref(), vec![0; 32]);

        let x = channel.draw_random();
        assert_eq!(
            x.to_string(),
            "ae09db7cd54f42b490ef09b6bc541af688e4959bb8c53f359a6f56e38ab454a3"
        );
        assert_eq!(channel.counter, 1);

        // Reseed channel and check the digest was changed.
        let old_digest = channel.digest;
        channel.mix_with_digest(Blake2sHash::from(vec![1; 32]));
        assert_ne!(old_digest.to_string(), channel.digest.to_string());
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
