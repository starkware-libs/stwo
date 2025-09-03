use core::{array, iter};

use itertools::Itertools;
use starknet_crypto::{poseidon_hash, poseidon_hash_many, poseidon_permute_comp};
use starknet_ff::FieldElement as FieldElement252;
use std_shims::{vec, Vec};

use super::{Channel, ChannelTime};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::vcs::utils::add_length_padding;

// Number of bytes that fit into a felt252.
pub const BYTES_PER_FELT252: usize = 252 / 8;
pub const FELTS_PER_HASH: usize = 8;

/// A channel that can be used to draw random elements from a Poseidon252 hash.
#[derive(Clone, Default, Debug)]
pub struct Poseidon252Channel {
    digest: FieldElement252,
    pub channel_time: ChannelTime,
}

impl Poseidon252Channel {
    pub const POW_PREFIX: u32 = 0x12345678;

    pub const fn digest(&self) -> FieldElement252 {
        self.digest
    }
    pub const fn update_digest(&mut self, new_digest: FieldElement252) {
        self.digest = new_digest;
        self.channel_time.inc_challenges();
    }

    fn draw_secure_felt252(&mut self) -> FieldElement252 {
        // We call `poseidon_permute_comp` here with `FieldElement252::THREE` to ensure domain
        // separation between the draw and mix operations. In all mix functions, the constant used
        // is either ZERO or TWO, so using THREE here distinguishes this context.
        let mut state = [
            self.digest,
            self.channel_time.n_sent.into(),
            FieldElement252::THREE,
        ];
        poseidon_permute_comp(&mut state);
        let res = state[0];
        self.channel_time.inc_sent();
        res
    }

    // TODO(shahars): Understand if we really need uniformity here.
    /// Generates a close-to uniform random vector of BaseField elements.
    fn draw_base_felts(&mut self) -> [BaseField; 8] {
        let shift = (1u64 << 31).into();

        let mut cur = self.draw_secure_felt252();
        let u32s: [u32; 8] = array::from_fn(|_| {
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
    const BYTES_PER_HASH: usize = BYTES_PER_FELT252;

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let shift = (1u64 << 31).into();
        let mut res = Vec::with_capacity(felts.len() / 2 + 2);
        res.push(self.digest);
        for chunk in felts.chunks(2) {
            res.push(
                chunk
                    .iter()
                    .flat_map(|x| x.to_m31_array())
                    .fold(FieldElement252::ONE, |cur, y| cur * shift + y.0.into()),
            );
        }

        // TODO(shahars): do we need length padding?
        self.update_digest(poseidon_hash_many(&res));
    }

    /// Mix a slice of u32s in chunks of 7 representing big endian felt252s.
    fn mix_u32s(&mut self, data: &[u32]) {
        let shift = (1u64 << 32).into();
        let padding_len = 6 - ((data.len() + 6) % 7);
        let mut felts = data
            .iter()
            .chain(iter::repeat_n(&0, padding_len))
            .chunks(7)
            .into_iter()
            .map(|chunk| {
                chunk.fold(FieldElement252::default(), |cur, y| {
                    cur * shift + (*y).into()
                })
            })
            .collect_vec();
        // If `data.len() % 7 != 0`, inject it into the bits [248:251] of the last
        // felt252.
        if padding_len != 0 {
            let last = felts.last_mut().unwrap();
            add_length_padding(last, 7 - padding_len);
        }
        self.update_digest(poseidon_hash_many(&[vec![self.digest], felts].concat()));
    }

    fn mix_u64(&mut self, value: u64) {
        self.update_digest(poseidon_hash(self.digest, value.into()));
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

    fn draw_random_bytes(&mut self) -> Vec<u8> {
        let shift = (1u64 << 8).into();
        let mut cur = self.draw_secure_felt252();
        let bytes: [u8; 31] = array::from_fn(|_| {
            let next = cur.floor_div(shift);
            let res = cur - next * shift;
            cur = next;
            res.try_into().unwrap()
        });
        bytes.to_vec()
    }

    /// Verifies that `H(H(POW_PREFIX, digest, n_bits), nonce)` has at least `n_bits` many
    /// leading zeros.
    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        let prefixed_digest =
            poseidon_hash_many(&[Self::POW_PREFIX.into(), self.digest, n_bits.into()]);
        let hash = poseidon_hash(prefixed_digest, nonce.into());
        let bytes = hash.to_bytes_be();
        let n_zeros = u128::from_be_bytes(bytes[16..].try_into().unwrap()).trailing_zeros();
        n_zeros >= n_bits
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use starknet_ff::FieldElement as FieldElement252;
    use std_shims::BTreeSet;

    use crate::core::channel::poseidon252::Poseidon252Channel;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::m31;

    #[test]
    fn test_channel_time() {
        let mut channel = Poseidon252Channel::default();

        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 0);

        channel.draw_random_bytes();
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 1);

        channel.draw_secure_felts(9);
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 6);
    }

    #[test]
    fn test_draw_random_bytes() {
        let mut channel = Poseidon252Channel::default();

        let first_random_bytes = channel.draw_random_bytes();

        // Assert that next random bytes are different.
        assert_ne!(first_random_bytes, channel.draw_random_bytes());
    }

    #[test]
    pub fn test_draw_secure_felt() {
        let mut channel = Poseidon252Channel::default();

        let first_random_felt = channel.draw_secure_felt();

        // Assert that next random felt is different.
        assert_ne!(first_random_felt, channel.draw_secure_felt());
    }

    #[test]
    pub fn test_draw_secure_felts() {
        let mut channel = Poseidon252Channel::default();

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
        let mut channel = Poseidon252Channel::default();
        let initial_digest = channel.digest;
        let felts = (0..2)
            .map(|i| SecureField::from(m31!(i + 1923782)))
            .collect_vec();

        channel.mix_felts(felts.as_slice());

        assert_ne!(initial_digest, channel.digest);
    }

    #[test]
    pub fn test_mix_u64() {
        let mut channel = Poseidon252Channel::default();
        channel.mix_u64(0x1111222233334444);

        assert_eq!(
            channel.digest(),
            FieldElement252::from_hex_be(
                "0x07cecc0ee3d858c843fe63165f038353f9f80f52dd8d32eead9f635e2f7d8b8e"
            )
            .unwrap()
        );
    }

    #[test]
    pub fn test_mix_u32s() {
        let mut channel = Poseidon252Channel::default();
        channel.mix_u32s(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            channel.digest,
            FieldElement252::from_hex_be(
                "0x06c7fc11690eb272bcc81115e801ad52de4e6271ddff3f97a2b75315e3572ced"
            )
            .unwrap()
        );
    }
}
