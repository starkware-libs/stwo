use std::fmt;

use super::hasher::Name;

// Wrapper for the blake3 hash type.
#[derive(Clone, Copy, PartialEq, Debug)]
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

impl fmt::Display for Blake3Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
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
    const BLOCK_SIZE_IN_BYTES: usize = 64;
    const OUTPUT_SIZE_IN_BYTES: usize = 32;
    fn hash(val: &[u8]) -> Blake3Hash {
        Blake3Hash(*blake3::hash(val).as_bytes())
    }

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Blake3Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&v1.0);
        hasher.update(&v2.0);

        Blake3Hash(*hasher.finalize().as_bytes())
    }

    fn hash_many(data: &[Vec<u8>]) -> Vec<Self::Hash> {
        data.iter().map(|x| Self::hash(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::{blake3_hash, hasher::Hasher};

    #[test]
    fn single_hash_test() {
        let hash_a = blake3_hash::Blake3Hasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
        );
    }

    #[test]
    fn hash_many_test() {
        let input: Vec<Vec<u8>> = std::iter::repeat(b"a".to_vec()).take(3).collect();
        let hash_result = blake3_hash::Blake3Hasher::hash_many(&input);

        for h in hash_result {
            assert_eq!(
                h.to_string(),
                "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
            );
        }
    }
}
