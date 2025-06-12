//! Vector commitment scheme (VCS) module.

pub mod blake2_hash;
pub mod blake2_merkle;
pub mod blake3_hash;
pub mod hash;
mod merkle_hasher;
pub use merkle_hasher::MerkleHasher;
#[cfg(not(target_arch = "wasm32"))]
pub mod poseidon252_merkle;
#[cfg(all(test, feature = "prover"))]
pub mod test_utils;
pub mod utils;
pub mod verifier;
