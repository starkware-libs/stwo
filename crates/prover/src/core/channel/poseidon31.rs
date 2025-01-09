use std::iter;

use num_traits::Zero;

use crate::core::channel::{Channel, ChannelTime};
use crate::core::fields::m31::{BaseField, M31, P};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::vcs::poseidon31_ref::poseidon2_permute;

pub const POSEIDON31_BYTES_PER_HASH: usize = 32;
pub const FELTS_PER_HASH: usize = 8;

/// A channel that can be used to draw random elements from a Poseidon31 hash.
#[derive(Clone, Default)]
pub struct Poseidon31Channel {
    digest: [M31; 8],
    pub channel_time: ChannelTime,
}

impl Poseidon31Channel {
    pub const fn digest(&self) -> [M31; 8] {
        self.digest
    }
    pub fn update_digest(&mut self, new_digest: [M31; 8]) {
        self.digest = new_digest;
        self.channel_time.inc_challenges();
    }

    fn draw_base_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        assert!((self.channel_time.n_sent as u64) < (P as u64));
        let zero = M31::zero();
        let mut state = [
            M31::from(self.channel_time.n_sent),
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            self.digest[0],
            self.digest[1],
            self.digest[2],
            self.digest[3],
            self.digest[4],
            self.digest[5],
            self.digest[6],
            self.digest[7],
        ];

        poseidon2_permute(&mut state);

        self.channel_time.inc_sent();

        // extract elements from the first 8 elements, not the last 8 elements
        state.first_chunk::<8>().unwrap().clone()
    }
}

impl Channel for Poseidon31Channel {
    const BYTES_PER_HASH: usize = POSEIDON31_BYTES_PER_HASH;

    fn trailing_zeros(&self) -> u32 {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.digest[0].0.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.digest[1].0.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.digest[2].0.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.digest[3].0.to_le_bytes());

        u128::from_le_bytes(bytes).trailing_zeros()
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let zero = M31::zero();
        let mut state = [
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            self.digest[0],
            self.digest[1],
            self.digest[2],
            self.digest[3],
            self.digest[4],
            self.digest[5],
            self.digest[6],
            self.digest[7],
        ];

        for chunk in felts.chunks(2) {
            let felts = chunk[0].to_m31_array();
            for i in 0..4 {
                state[i] = felts[i];
            }

            if chunk.len() == 2 {
                let felts = chunk[1].to_m31_array();
                for i in 0..4 {
                    state[i + 4] = felts[i];
                }
            } else {
                for i in 0..4 {
                    state[i + 4] = zero;
                }
            }

            poseidon2_permute(&mut state);
        }

        let new_digest = state.last_chunk::<8>().unwrap();
        self.update_digest(*new_digest);
    }

    fn mix_u64(&mut self, value: u64) {
        let zero = M31::zero();
        let n1 = value % ((1 << 22) - 1); // 22 bits
        let n2 = (value >> 22) & ((1 << 21) - 1); // 21 bits
        let n3 = (value >> 43) & ((1 << 21) - 1); // 21 bits

        let mut state = [
            M31::from(n1 as u32),
            M31::from(n2 as u32),
            M31::from(n3 as u32),
            zero,
            zero,
            zero,
            zero,
            zero,
            self.digest[0],
            self.digest[1],
            self.digest[2],
            self.digest[3],
            self.digest[4],
            self.digest[5],
            self.digest[6],
            self.digest[7],
        ];
        poseidon2_permute(&mut state);

        let new_digest = state.last_chunk::<8>().unwrap();
        self.update_digest(*new_digest);
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
        // the implementation here is based on the assumption that the only place draw_random_bytes
        // will be used is in generating the queries, where only the lowest n bits of every 4 bytes
        // slice would be used.

        let felts: [BaseField; FELTS_PER_HASH] = self.draw_base_felts();
        let mut bytes = Vec::with_capacity(FELTS_PER_HASH * 4);
        for i in 0..FELTS_PER_HASH {
            // important: only le bytes
            bytes.extend(felts[i].0.to_le_bytes());
        }

        bytes
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::core::channel::{Channel, Poseidon31Channel};
    use crate::core::fields::qm31::SecureField;
    use crate::m31;

    #[test]
    fn test_channel_time() {
        let mut channel = Poseidon31Channel::default();

        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 0);

        channel.draw_random_bytes();
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 1);

        channel.draw_felts(9);
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 6);
    }

    #[test]
    fn test_draw_random_bytes() {
        let mut channel = Poseidon31Channel::default();

        let first_random_bytes = channel.draw_random_bytes();

        // Assert that next random bytes are different.
        assert_ne!(first_random_bytes, channel.draw_random_bytes());
    }

    #[test]
    pub fn test_draw_felt() {
        let mut channel = Poseidon31Channel::default();

        let first_random_felt = channel.draw_felt();

        // Assert that next random felt is different.
        assert_ne!(first_random_felt, channel.draw_felt());
    }

    #[test]
    pub fn test_draw_felts() {
        let mut channel = Poseidon31Channel::default();

        let mut random_felts = channel.draw_felts(5);
        random_felts.extend(channel.draw_felts(4));

        // Assert that all the random felts are unique.
        assert_eq!(
            random_felts.len(),
            random_felts.iter().collect::<BTreeSet<_>>().len()
        );
    }

    #[test]
    pub fn test_mix_felts() {
        let mut channel = Poseidon31Channel::default();
        let initial_digest = channel.digest;
        let felts: Vec<SecureField> = (0..2)
            .map(|i| SecureField::from(m31!(i + 1923782)))
            .collect();

        channel.mix_felts(felts.as_slice());

        assert_ne!(initial_digest, channel.digest);
    }
}
