use std::iter;

use starknet_crypto::poseidon_hash;
use starknet_ff::FieldElement as FieldElement252;

use super::{Channel, ChannelTime};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;

pub const BYTES_PER_FELT252: usize = 31;
pub const FELTS_PER_HASH: usize = 8;

/// A channel that can be used to draw random elements from a Poseidon252 hash.
pub struct Poseidon252Channel {
    digest: FieldElement252,
    channel_time: ChannelTime,
}

impl Poseidon252Channel {
    fn draw_felt252(&mut self) -> FieldElement252 {
        let res = poseidon_hash(self.digest, self.channel_time.n_sent.into());
        self.channel_time.inc_sent();
        res
    }

    // TODO(spapini): Understand if we really need uniformity here.
    /// Generates a close-to uniform random vector of BaseField elements.
    fn draw_base_felts(&mut self) -> [BaseField; 8] {
        let shift = (1u64 << 31).into();

        let mut cur = self.draw_felt252();
        let u32s: [u32; 8] = std::array::from_fn(|_| {
            let next = cur.floor_div(shift);
            let res = cur - next * shift;
            cur = next;
            res.try_into().unwrap()
        });

        u32s.into_iter()
            .map(|x| BaseField::reduce(x as u64))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

impl Channel for Poseidon252Channel {
    type Digest = FieldElement252;
    const BYTES_PER_HASH: usize = BYTES_PER_FELT252;

    fn new(digest: Self::Digest) -> Self {
        Poseidon252Channel {
            digest,
            channel_time: ChannelTime::default(),
        }
    }

    fn get_digest(&self) -> Self::Digest {
        self.digest
    }

    fn mix_digest(&mut self, digest: Self::Digest) {
        self.digest = poseidon_hash(self.digest, digest);
        self.channel_time.inc_challenges();
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let shift = (1u64 << 31).into();
        let mut cur = FieldElement252::default();
        let mut in_chunk = 0;
        for x in felts {
            for y in x.to_m31_array() {
                cur = cur * shift + y.0.into();
            }
            in_chunk += 1;
            if in_chunk == 2 {
                self.digest = poseidon_hash(self.digest, cur);
                cur = FieldElement252::default();
                in_chunk = 0;
            }
        }
        if in_chunk > 0 {
            self.digest = poseidon_hash(self.digest, cur);
        }

        // TODO(spapini): do we need length padding?
        self.channel_time.inc_challenges();
    }

    fn mix_nonce(&mut self, nonce: u64) {
        self.mix_digest(nonce.into())
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
        let shift = (1u64 << 8).into();
        let mut cur = self.draw_felt252();
        let bytes: [u8; 31] = std::array::from_fn(|_| {
            let next = cur.floor_div(shift);
            let res = cur - next * shift;
            cur = next;
            res.try_into().unwrap()
        });
        bytes.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use starknet_ff::FieldElement as FieldElement252;

    use crate::core::channel::poseidon252::Poseidon252Channel;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::m31;

    #[test]
    fn test_initialize_channel() {
        let initial_digest = FieldElement252::default();
        let channel = Poseidon252Channel::new(initial_digest);

        // Assert that the channel is initialized correctly.
        assert_eq!(channel.digest, initial_digest);
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 0);
    }

    #[test]
    fn test_channel_time() {
        let initial_digest = FieldElement252::default();
        let mut channel = Poseidon252Channel::new(initial_digest);

        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 0);

        channel.draw_random_bytes();
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 1);

        channel.draw_felts(9);
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 6);

        channel.mix_digest(FieldElement252::default());
        assert_eq!(channel.channel_time.n_challenges, 1);
        assert_eq!(channel.channel_time.n_sent, 0);

        channel.draw_felt();
        assert_eq!(channel.channel_time.n_challenges, 1);
        assert_eq!(channel.channel_time.n_sent, 1);
        assert_ne!(channel.digest, initial_digest);
    }

    #[test]
    fn test_draw_random_bytes() {
        let initial_digest = FieldElement252::default();
        let mut channel = Poseidon252Channel::new(initial_digest);

        let first_random_bytes = channel.draw_random_bytes();

        // Assert that next random bytes are different.
        assert_ne!(first_random_bytes, channel.draw_random_bytes());
    }

    #[test]
    pub fn test_draw_felt() {
        let initial_digest = FieldElement252::default();
        let mut channel = Poseidon252Channel::new(initial_digest);

        let first_random_felt = channel.draw_felt();

        // Assert that next random felt is different.
        assert_ne!(first_random_felt, channel.draw_felt());
    }

    #[test]
    pub fn test_draw_felts() {
        let initial_digest = FieldElement252::default();
        let mut channel = Poseidon252Channel::new(initial_digest);

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
        let initial_digest = FieldElement252::default();
        let mut channel = Poseidon252Channel::new(initial_digest);

        for _ in 0..10 {
            channel.draw_random_bytes();
            channel.draw_felt();
        }

        // Reseed channel and check the digest was changed.
        channel.mix_digest(FieldElement252::default());
        assert_ne!(initial_digest, channel.digest);
    }

    #[test]
    pub fn test_mix_felts() {
        let initial_digest = FieldElement252::default();
        let mut channel = Poseidon252Channel::new(initial_digest);
        let felts: Vec<SecureField> = (0..2)
            .map(|i| SecureField::from(m31!(i + 1923782)))
            .collect();

        channel.mix_felts(felts.as_slice());

        assert_ne!(initial_digest, channel.digest);
    }
}
