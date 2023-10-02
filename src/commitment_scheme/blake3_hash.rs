use std::fmt;

use super::hasher::Name;

// Wrapper for the blake3 hash type.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Blake3Hash(pub blake3::Hash);

impl From<Blake3Hash> for Vec<u8> {
    fn from(value: Blake3Hash) -> Self {
        Vec::from(*(value.0.as_bytes()))
    }
}

impl From<Vec<u8>> for Blake3Hash {
    fn from(value: Vec<u8>) -> Self {
        Self(blake3::Hash::from_bytes(
            value
                .try_into()
                .expect("Failed slice -> [u8; 32]:Blake3Hash"),
        ))
    }
}

impl fmt::Display for Blake3Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = self.0.to_hex();
        let hex: &str = hex.as_str();
        f.write_str(hex)
    }
}

impl Name for Blake3Hash {
    const NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("BLAKE3");
}

// Wrapper for the blake3 Hashing functionalities.
#[derive(Clone)]
pub struct Blake3Hasher {}

impl super::hasher::Hasher for Blake3Hasher {
    type Hash = Blake3Hash;
    const BLOCK_SIZE: usize = 64;
    fn hash(val: &[u8]) -> Blake3Hash {
        Blake3Hash(blake3::hash(val))
    }

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Blake3Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(v1.0.as_bytes());
        hasher.update(v2.0.as_bytes());

        Blake3Hash(hasher.finalize())
    }
}
