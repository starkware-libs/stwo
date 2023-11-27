use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::hasher::Hasher;

pub trait ChannelHash {
    fn hash_with_digest(&self, digest: Self) -> Self;
    fn hash_with_counter(&self, counter: u32) -> Self;
}

impl ChannelHash for Blake2sHash {
    fn hash_with_digest(&self, digest: Blake2sHash) -> Self {
        Blake2sHasher::concat_and_hash(self, &digest)
    }

    fn hash_with_counter(&self, counter: u32) -> Self {
        // Pad the counter to 32 bytes.
        let mut padded_counter = [0; 32];
        let counter_bytes = counter.to_le_bytes();
        padded_counter[..counter_bytes.len()].copy_from_slice(&counter_bytes);

        Blake2sHasher::concat_and_hash(self, &Blake2sHash::from(&padded_counter[..]))
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
        let new_digest = self.digest.hash_with_digest(seed);
        self.digest = new_digest;
        self.counter = 0;
    }

    pub fn draw_random(&mut self) -> T {
        let rand = self.digest.hash_with_counter(self.counter);
        self.counter += 1;
        rand
    }
}

#[cfg(test)]
mod tests {
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
}
