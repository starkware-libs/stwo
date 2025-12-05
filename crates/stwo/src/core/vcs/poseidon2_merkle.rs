use core::fmt;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std_shims::Vec;

use super::poseidon2_primitives::{poseidon2_permute, N_STATE};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::MerkleHasher;
use crate::core::vcs::hash::Hash;

const RATE_SIZE: usize = 8; 

#[derive(Copy, Clone, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Poseidon2Hash(pub [BaseField; 8]);

impl fmt::Display for Poseidon2Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poseidon2Hash(")?;
        for (i, val) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, ")")
    }
}

impl fmt::Debug for Poseidon2Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Poseidon2Hash as fmt::Display>::fmt(self, f)
    }
}

impl Hash for Poseidon2Hash {}

impl From<Poseidon2Hash> for Vec<u8> {
    fn from(val: Poseidon2Hash) -> Self {
        let mut res = Vec::with_capacity(32); // 8 * 4 bytes
        for elem in val.0 {
            // M31 is u32 basically (31 bits).
            let val_u32: u32 = elem.0; 
            res.extend_from_slice(&val_u32.to_le_bytes());
        }
        res
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Poseidon2MerkleHasher;

impl MerkleHasher for Poseidon2MerkleHasher {
    type Hash = Poseidon2Hash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let mut state = [BaseField::zero(); N_STATE];

        // If children exist, absorb them first
        if let Some((left, right)) = children_hashes {
            // Absorb left child
            for (i, &val) in left.0.iter().enumerate() {
                state[i] = val;
            }
            // Absorb right child
            for (i, &val) in right.0.iter().enumerate() {
                state[8 + i] = val;
            }
            poseidon2_permute(&mut state);
        }

        // Absorb column values
        for chunk in column_values.chunks(RATE_SIZE) {
            for (i, &val) in chunk.iter().enumerate() {
                state[i] += val;
            }
            poseidon2_permute(&mut state);
        }

        // Return first 8 elements as digest
        let mut digest = [BaseField::zero(); 8];
        digest.copy_from_slice(&state[..8]);
        Poseidon2Hash(digest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m31;

    #[test]
    fn test_poseidon2_hash_node_consistency() {
        // Test 1: Hash two children (consistency check)
        let left = Poseidon2Hash([m31!(1); 8]);
        let right = Poseidon2Hash([m31!(2); 8]);
        
        let hash1 = Poseidon2MerkleHasher::hash_node(Some((left, right)), &[]);
        let hash2 = Poseidon2MerkleHasher::hash_node(Some((left, right)), &[]);
        
        assert_eq!(hash1, hash2, "Hash must be deterministic");
        assert_ne!(hash1, left, "Hash result should differ from input");
    }

    #[test]
    fn test_poseidon2_hash_node_values() {
        // Test 2: Hash column values
        let val1 = m31!(100);
        let val2 = m31!(200);
        
        let hash1 = Poseidon2MerkleHasher::hash_node(None, &[val1, val2]);
        let hash2 = Poseidon2MerkleHasher::hash_node(None, &[val2, val1]);
        
        // Since we absorb chunks sequentially and simple addition in this implementation might commute within a chunk 
        // if not carefully permuted, but here we do `state[i] += val`.
        // Wait, `state[i] += val` in a loop over `chunk`.
        // If chunk is `[A, B]`: state[0]+=A, state[1]+=B. This is positional, so order matters.
        assert_ne!(hash1, hash2, "Order of values should matter");
    }

    #[test]
    fn test_poseidon2_hash_diff_children() {
        let h1 = Poseidon2MerkleHasher::hash_node(None, &[m31!(1)]);
        let h2 = Poseidon2MerkleHasher::hash_node(None, &[m31!(2)]);
        assert_ne!(h1, h2);
    }
}
