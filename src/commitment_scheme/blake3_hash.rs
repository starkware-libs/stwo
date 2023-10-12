use super::hasher::Name;
use blake3;
use std::fmt;

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
    fn hash(val: &[u8]) -> Blake3Hash {
        Blake3Hash(*blake3::hash(val).as_bytes())
    }

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Blake3Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&v1.0);
        hasher.update(&v2.0);

        Blake3Hash(*hasher.finalize().as_bytes())
    }

    // TODO(Ohad): Implement SIMD parallelised hashing.
    fn hash_many(inputs: &[Vec<u8>]) -> Vec<Self::Hash> {
        inputs.iter().map(|b| Self::hash(b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::hasher::Hasher;

    #[test]
    fn test_build_vec_from_blake() {
        let hash_a = super::Blake3Hasher::hash(b"a");
        let vec_a: Vec<u8> = hash_a.into();
        assert_eq!(
            hex::encode(&vec_a[..]),
            String::from("17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f")
        );
    }

    #[test]
    fn single_hash() {
        let hash_a = super::Blake3Hasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
        );
    }
}
