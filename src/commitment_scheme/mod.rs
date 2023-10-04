pub mod blake2_hash;
pub mod blake3_hash;
pub mod hasher;
pub mod merkle_tree;

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::hasher::Hasher;

    use super::blake3_hash::{self, Blake3Hasher};
    use super::merkle_tree::MerkleTree;



    #[test]
    fn test_build_vec_from_blake() {
        let hash_a = blake3_hash::Blake3Hasher::hash(b"a");
        let vec_a: Vec<u8> = hash_a.into();
        assert_eq!(hex::encode(&vec_a[..]), String::from("17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"));
    }

    #[test]
    fn single_hash() {
        let hash_a = blake3_hash::Blake3Hasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
        );
    }

    #[test]
    fn merkle_tree_building() {
        let leaves = [0; 64];
        let mut tree: MerkleTree<Blake3Hasher> = MerkleTree::commit(&leaves[..]);
        assert_eq!(tree.height, 3);
        assert_eq!(
            tree.data[0][0].to_string(),
            "4d006976636a8696d909a630a4081aad4d7c50f81afdee04020bf05086ab6a55"
        );
        assert_eq!(
            tree.root_hex(),
            "06253c52ed8536e4b07757d679c547fdb2051181a9cbd1e3516bfc71742936f7"
        )
    }
}
