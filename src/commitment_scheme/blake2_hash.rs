use blake2::{Blake2s256, Digest};
use std::fmt;

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

    fn hash_many(inputs: &[Vec<u8>]) -> Vec<Self::Hash> {
        inputs.iter().map(|b| Self::hash(b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::hasher::Hasher;

    #[test]
    fn test_build_vec_from_blake() {
        let hash_a = super::Blake2sHasher::hash(b"a");
        let vec_a: Vec<u8> = hash_a.into();
        assert_eq!(
            hex::encode(&vec_a[..]),
            String::from("4a0d129873403037c2cd9b9048203687f6233fb6738956e0349bd4320fec3e90")
        );
    }

    #[test]
    fn single_hash() {
        let hash_a = super::Blake2sHasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "4a0d129873403037c2cd9b9048203687f6233fb6738956e0349bd4320fec3e90"
        );
    }
}
