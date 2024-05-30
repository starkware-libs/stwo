use itertools::Itertools;
use num_traits::FromPrimitive;
use starknet_types_core::felt::Felt as felt252;
use starknet_types_core::hash::{Poseidon, StarkHash};

use super::channel::{Channel, ChannelTime};
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;

#[derive(Clone, Copy, PartialEq, Default, Eq)]
pub struct PoseidonHash {
    hash: felt252,
}

impl PoseidonHash {
    pub fn new(hash: felt252) -> Self {
        Self { hash }
    }
}

/// A channel that can be used to draw random elements from a [PoseidonHash] digest.
pub struct PoseidonChannel {
    digest: PoseidonHash,
    channel_time: ChannelTime,
}

impl PoseidonChannel {
    fn draw_felt252(&mut self) -> felt252 {
        let mut hash_input = vec![self.get_digest().hash];
        hash_input.push(felt252::from_u64(self.channel_time.n_sent as u64).unwrap());
        self.channel_time.inc_sent();
        Poseidon::hash_array(&hash_input)
    }
}

impl Channel for PoseidonChannel {
    type Digest = PoseidonHash;
    const BYTES_PER_HASH: usize = 16;

    fn new(digest: PoseidonHash) -> Self {
        Self {
            digest,
            channel_time: ChannelTime::default(),
        }
    }

    fn get_digest(&self) -> PoseidonHash {
        self.digest
    }

    fn mix_digest(&mut self, digest: PoseidonHash) {
        self.digest = PoseidonHash::new(Poseidon::hash(&self.get_digest().hash, &digest.hash));
        self.channel_time.inc_challenges();
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let secure_felts_as_felt252: Vec<felt252> = felts
            .iter()
            .map(|felt| secure_field_to_felt252(*felt))
            .collect();
        let hash_input = vec![self.get_digest().hash]
            .into_iter()
            .chain(secure_felts_as_felt252.iter().cloned())
            .collect::<Vec<_>>();
        self.digest = PoseidonHash::new(Poseidon::hash_array(&hash_input));
        self.channel_time.inc_challenges();
    }

    fn mix_nonce(&mut self, nonce: u64) {
        let hash_input = vec![self.get_digest().hash, felt252::from_u64(nonce).unwrap()];
        self.digest = PoseidonHash::new(Poseidon::hash_array(&hash_input));
        self.channel_time.inc_challenges();
    }

    fn draw_felt(&mut self) -> SecureField {
        let felt = self.draw_felt252();
        felt252_to_secure_field(felt)
    }

    fn draw_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        (0..n_felts).map(|_| self.draw_felt()).collect()
    }

    fn draw_random_bytes(&mut self) -> Vec<u8> {
        self.draw_felt252().to_bytes_le()[..16].to_vec()
    }
}

fn secure_field_to_felt252(felt: SecureField) -> felt252 {
    felt252::from_raw(
        felt.to_m31_array()
            .iter()
            .map(|x| x.0 as u64)
            .collect_vec()
            .try_into()
            .unwrap(),
    )
}

fn felt252_to_secure_field(felt: felt252) -> SecureField {
    let u32_array = felt.to_bytes_le()[..16]
        .chunks(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect_vec();
    let m31_array = u32_array
        .iter()
        .map(|x| BaseField::reduce(*x as u64))
        .collect_vec();
    SecureField::from_m31_array(m31_array.try_into().unwrap())
}
