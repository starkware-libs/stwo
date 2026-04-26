use core::fmt;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std_shims::Vec;

#[repr(C, align(32))]
#[derive(Clone, Copy, PartialEq, Default, Eq, Pod, Zeroable, Deserialize, Serialize)]
pub struct Keccak256Hash(pub [u8; 32]);

impl From<Keccak256Hash> for Vec<u8> {
    fn from(value: Keccak256Hash) -> Self {
        Vec::from(value.0)
    }
}

impl From<Vec<u8>> for Keccak256Hash {
    fn from(value: Vec<u8>) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting Vec<u8> to Keccak256Hash type"),
        )
    }
}

impl From<&[u8]> for Keccak256Hash {
    fn from(value: &[u8]) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting &[u8] to Keccak256Hash type"),
        )
    }
}

impl AsRef<[u8]> for Keccak256Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<Keccak256Hash> for [u8; 32] {
    fn from(val: Keccak256Hash) -> Self {
        val.0
    }
}

impl fmt::Display for Keccak256Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for Keccak256Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Keccak256Hash as fmt::Display>::fmt(self, f)
    }
}

impl super::hash::Hash for Keccak256Hash {}

#[derive(Clone, Debug, Default)]
pub struct Keccak256Hasher {
    state: Keccak256,
}

impl Keccak256Hasher {
    pub fn new() -> Self {
        Self {
            state: Keccak256::new(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        Digest::update(&mut self.state, data);
    }

    pub fn finalize(self) -> Keccak256Hash {
        Keccak256Hash(self.state.finalize().into())
    }

    pub fn concat_and_hash(v1: &Keccak256Hash, v2: &Keccak256Hash) -> Keccak256Hash {
        let mut hasher = Self::new();
        hasher.update(v1.as_ref());
        hasher.update(v2.as_ref());
        hasher.finalize()
    }

    pub fn hash(data: &[u8]) -> Keccak256Hash {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

#[cfg(test)]
mod tests {
    use std_shims::ToString;

    use super::{Keccak256Hash, Keccak256Hasher};

    impl Keccak256Hasher {
        fn finalize_reset(&mut self) -> Keccak256Hash {
            use sha3::Digest;
            let state = core::mem::take(&mut self.state);
            Keccak256Hash(state.finalize().into())
        }
    }

    #[test]
    fn single_hash_test() {
        let hash_a = Keccak256Hasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "3ac225168df54212a25c1c01fd35bebfea408fdac2e31ddd6f80a4bbf9a5f1cb"
        );
    }

    #[test]
    fn hash_state_test() {
        let mut state = Keccak256Hasher::new();
        state.update(b"a");
        state.update(b"b");
        let hash = state.finalize_reset();
        let hash_empty = state.finalize();

        assert_eq!(hash.to_string(), Keccak256Hasher::hash(b"ab").to_string());
        assert_eq!(
            hash_empty.to_string(),
            Keccak256Hasher::hash(b"").to_string()
        );
    }
}
