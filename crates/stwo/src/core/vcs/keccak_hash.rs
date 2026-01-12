use core::fmt;

use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std_shims::Vec;

/// Keccak256 hash wrapper for STWO compatibility  
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub struct KeccakHash(pub [u8; 32]);

impl From<[u8; 32]> for KeccakHash {
    fn from(value: [u8; 32]) -> Self {
        Self(value)
    }
}

impl From<KeccakHash> for [u8; 32] {
    fn from(value: KeccakHash) -> [u8; 32] {
        value.0
    }
}

impl From<KeccakHash> for Vec<u8> {
    fn from(value: KeccakHash) -> Self {
        Vec::from(value.0)
    }
}

impl From<Vec<u8>> for KeccakHash {
    fn from(value: Vec<u8>) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting Vec<u8> to KeccakHash type"),
        )
    }
}

impl From<&[u8]> for KeccakHash {
    fn from(value: &[u8]) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting &[u8] to KeccakHash Type!"),
        )
    }
}

impl AsRef<[u8]> for KeccakHash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for KeccakHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for KeccakHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <KeccakHash as fmt::Display>::fmt(self, f)
    }
}

impl super::hash::Hash for KeccakHash {}

/// Keccak256 hasher implementation
#[derive(Clone, Debug, Default)]
pub struct KeccakHasher {
    hasher: Keccak256,
}

impl KeccakHasher {
    /// Create a new hasher instance
    pub fn new() -> Self {
        Self {
            hasher: Keccak256::new(),
        }
    }

    /// Update hasher with data
    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// Finalize and return hash
    pub fn finalize(self) -> KeccakHash {
        KeccakHash(self.hasher.finalize().into())
    }

    /// One-shot hash function
    pub fn hash(data: &[u8]) -> KeccakHash {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }

    /// Concatenate and hash two hashes (for Merkle operations)
    pub fn concat_and_hash(left: &KeccakHash, right: &KeccakHash) -> KeccakHash {
        let mut hasher = Self::new();
        hasher.update(left.as_ref());
        hasher.update(right.as_ref());
        hasher.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keccak_hash_basic() {
        let data = b"hello world";
        let hash = KeccakHasher::hash(data);
        
        // Verify it's not all zeros
        assert_ne!(hash.0, [0u8; 32]);
        
        // Verify deterministic
        let hash2 = KeccakHasher::hash(data);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_keccak_hasher_update() {
        let mut hasher = KeccakHasher::new();
        hasher.update(b"hello");
        hasher.update(b" ");
        hasher.update(b"world");
        let hash1 = hasher.finalize();

        let hash2 = KeccakHasher::hash(b"hello world");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_concat_and_hash() {
        let hash1 = KeccakHasher::hash(b"first");
        let hash2 = KeccakHasher::hash(b"second");
        
        let combined = KeccakHasher::concat_and_hash(&hash1, &hash2);
        
        // Manual verification
        let mut hasher = KeccakHasher::new();
        hasher.update(&hash1.0);
        hasher.update(&hash2.0);
        let expected = hasher.finalize();
        
        assert_eq!(combined, expected);
    }
}