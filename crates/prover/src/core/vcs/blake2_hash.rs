use std::fmt;

use blake2::{Blake2s256, Digest};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
// Wrapper for the blake2s hash type.
#[repr(C, align(32))]
#[derive(Clone, Copy, PartialEq, Default, Eq, Pod, Zeroable, Deserialize, Serialize)]
pub struct Blake2sHash([u8; 32]);

impl From<Blake2sHash> for Vec<u8> {
    fn from(value: Blake2sHash) -> Self {
        Vec::from(value.0)
    }
}

impl From<Vec<u8>> for Blake2sHash {
    fn from(value: Vec<u8>) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting Vec<u8> to Blake2Hash type"),
        )
    }
}

impl From<&[u8]> for Blake2sHash {
    fn from(value: &[u8]) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting &[u8] to Blake2sHash Type!"),
        )
    }
}

impl AsRef<[u8]> for Blake2sHash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<Blake2sHash> for [u8; 32] {
    fn from(val: Blake2sHash) -> Self {
        val.0
    }
}

impl fmt::Display for Blake2sHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for Blake2sHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Blake2sHash as fmt::Display>::fmt(self, f)
    }
}

impl super::hasher::Name for Blake2sHash {
    const NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("BLAKE2");
}

impl super::hasher::Hash<u8> for Blake2sHash {}

// Wrapper for the blake2s Hashing functionalities.
#[derive(Clone, Debug, Default)]
pub struct Blake2sHasher {
    state: Blake2s256,
}

impl super::hasher::Hasher for Blake2sHasher {
    type Hash = Blake2sHash;
    const BLOCK_SIZE: usize = 64;
    const OUTPUT_SIZE: usize = 32;
    type NativeType = u8;

    fn new() -> Self {
        Self {
            state: Blake2s256::new(),
        }
    }

    fn reset(&mut self) {
        blake2::Digest::reset(&mut self.state);
    }

    fn update(&mut self, data: &[u8]) {
        blake2::Digest::update(&mut self.state, data);
    }

    fn finalize(self) -> Blake2sHash {
        Blake2sHash(self.state.finalize().into())
    }

    fn finalize_reset(&mut self) -> Blake2sHash {
        Blake2sHash(self.state.finalize_reset().into())
    }
}

#[cfg(test)]
mod tests {
    use super::Blake2sHasher;
    use crate::core::vcs::blake2_hash;
    use crate::core::vcs::hasher::Hasher;

    #[test]
    fn single_hash_test() {
        let hash_a = blake2_hash::Blake2sHasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "4a0d129873403037c2cd9b9048203687f6233fb6738956e0349bd4320fec3e90"
        );
    }

    #[test]
    fn hash_state_test() {
        let mut state = Blake2sHasher::new();
        state.update(b"a");
        state.update(b"b");
        let hash = state.finalize_reset();
        let hash_empty = state.finalize();

        assert_eq!(hash.to_string(), Blake2sHasher::hash(b"ab").to_string());
        assert_eq!(hash_empty.to_string(), Blake2sHasher::hash(b"").to_string());
    }
}
