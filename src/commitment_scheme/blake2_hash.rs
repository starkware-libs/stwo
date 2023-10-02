use std::fmt;

use blake2::{Blake2s256, Digest};

// Wrapper for the blake2 hash type
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
        // Formatting field as `&str` to reduce code size since the `Debug`
        // dynamic dispatch table for `&str` is likely needed elsewhere already,
        // but that for `ArrayString<[u8; 64]>` is not.
        let mut s = String::new();
        let table = b"0123456789abcdef";
        for &b in self.0.iter() {
            s.push(table[(b >> 4) as usize] as char);
            s.push(table[(b & 0xf) as usize] as char);
        }

        f.write_str(&s)
    }
}

// Wrapper for the blake 3 Hashing functionalities
#[derive(Clone)]
pub struct Blake2sHasher {}

impl super::hasher::Hasher for Blake2sHasher {
    type Hash = Blake2sHash;
    const BLOCK_SIZE: usize = 64;

    fn hash(val: &[u8]) -> Blake2sHash {
        let mut hasher = Blake2s256::new();
        hasher.update(val);
        let res = hasher.finalize();
        Blake2sHash(res.into())
    }

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Blake2sHash {
        let mut hasher = Blake2s256::new();
        hasher.update(v1.0);
        hasher.update(v2.0);

        Blake2sHash(hasher.finalize().into())
    }
}
