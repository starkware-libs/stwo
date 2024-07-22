use std::borrow::Cow;
use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

pub trait Name {
    const NAME: Cow<'static, str>;
}

/// An API for hash functions that support incremental hashing.
/// Provides advanced hashing functionality.
///
/// # Example
///
/// ```
/// use stwo_prover::core::vcs::blake3_hash::Blake3Hasher;
///
/// use crate::stwo_prover::core::vcs::hasher::BlakeHasher;
///
/// let mut hasher = Blake3Hasher::new();
/// hasher.update(&[1, 2, 3]);
/// hasher.update(&[4, 5, 6]);
/// let hash = hasher.finalize();
///
/// assert_eq!(hash, Blake3Hasher::hash(&[1, 2, 3, 4, 5, 6]));
/// ```

pub trait BlakeHasher: Sized + Default {
    type Hash: Hash + AsRef<[u8]>;
    // Input size of the compression function.
    const BLOCK_SIZE: usize;
    const OUTPUT_SIZE: usize;

    fn new() -> Self;

    fn reset(&mut self);

    fn update(&mut self, data: &[u8]);

    fn finalize(self) -> Self::Hash;

    fn finalize_reset(&mut self) -> Self::Hash;

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Self::Hash {
        let mut hasher = Self::new();
        hasher.update(v1.as_ref());
        hasher.update(v2.as_ref());
        hasher.finalize()
    }

    fn hash(data: &[u8]) -> Self::Hash {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

pub trait Hash:
    Copy
    + Default
    + Display
    + Debug
    + Eq
    + self::Name
    + Send
    + Sync
    + 'static
    + Serialize
    + for<'de> Deserialize<'de>
{
}
