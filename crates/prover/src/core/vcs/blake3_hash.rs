use std::fmt;

use serde::{Deserialize, Serialize};

use crate::core::vcs::hash::Hash;

// Wrapper for the blake3 hash type.
#[derive(Clone, Copy, PartialEq, Default, Eq, Serialize, Deserialize)]
pub struct Blake3Hash([u8; 32]);

impl From<Blake3Hash> for Vec<u8> {
    fn from(value: Blake3Hash) -> Self {
        Vec::from(value.0)
    }
}

impl From<Vec<u8>> for Blake3Hash {
    fn from(value: Vec<u8>) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting Vec<u8> to Blake3Hash Type!"),
        )
    }
}

impl From<&[u8]> for Blake3Hash {
    fn from(value: &[u8]) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting &[u8] to Blake3Hash Type!"),
        )
    }
}

impl AsRef<[u8]> for Blake3Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for Blake3Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for Blake3Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Blake3Hash as fmt::Display>::fmt(self, f)
    }
}

impl Hash for Blake3Hash {}

// Wrapper for the blake3 Hashing functionalities.
#[derive(Clone, Default)]
pub struct Blake3Hasher {
    state: blake3::Hasher,
}

impl Blake3Hasher {
    pub fn new() -> Self {
        Self {
            state: blake3::Hasher::new(),
        }
    }
    pub fn update(&mut self, data: &[u8]) {
        self.state.update(data);
    }

    pub fn finalize(self) -> Blake3Hash {
        Blake3Hash(self.state.finalize().into())
    }

    pub fn concat_and_hash(v1: &Blake3Hash, v2: &Blake3Hash) -> Blake3Hash {
        let mut hasher = Self::new();
        hasher.update(v1.as_ref());
        hasher.update(v2.as_ref());
        hasher.finalize()
    }

    pub fn hash(data: &[u8]) -> Blake3Hash {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

#[cfg(test)]
impl Blake3Hasher {
    fn finalize_reset(&mut self) -> Blake3Hash {
        let res = Blake3Hash(self.state.finalize().into());
        self.state.reset();
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::core::vcs::blake3_hash::Blake3Hasher;

    #[test]
    fn single_hash_test() {
        let hash_a = Blake3Hasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
        );
    }

    #[test]
    fn hash_state_test() {
        let mut state = Blake3Hasher::new();
        state.update(b"a");
        state.update(b"b");
        let hash = state.finalize_reset();
        let hash_empty = state.finalize();

        assert_eq!(hash.to_string(), Blake3Hasher::hash(b"ab").to_string());
        assert_eq!(hash_empty.to_string(), Blake3Hasher::hash(b"").to_string())
    }
}
