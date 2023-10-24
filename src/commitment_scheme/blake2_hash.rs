use std::fmt;

use blake2::{Blake2s256, Digest};

// Wrapper for the blake2s hash type.
#[derive(Clone, Copy, PartialEq)]
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

impl fmt::Display for Blake2sHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl super::hasher::Name for Blake2sHash {
    const NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("BLAKE2");
}

// Wrapper for the blake2s Hashing functionalities.
#[derive(Clone)]
pub struct Blake2sHasher {}

impl super::hasher::Hasher for Blake2sHasher {
    type Hash = Blake2sHash;
    const BLOCK_SIZE_IN_BYTES: usize = 64;
    const OUTPUT_SIZE_IN_BYTES: usize = 32;

    fn hash(val: &[u8]) -> Self::Hash {
        let mut hasher = Blake2s256::new();
        hasher.update(val);
        Blake2sHash(hasher.finalize().into())
    }

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Blake2sHash {
        let mut hasher = Blake2s256::new();
        hasher.update(v1.0);
        hasher.update(v2.0);

        Blake2sHash(hasher.finalize().into())
    }
}
