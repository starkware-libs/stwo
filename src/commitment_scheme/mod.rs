pub mod blake2_hash;
pub mod blake3_hash;
pub mod hasher;
pub mod merkle_tree;

#[cfg(test)]
mod tests {
    use super::blake3_hash::{self, Blake3Hasher};
    use super::merkle_tree::MerkleTree;
    use blake3::platform::Platform;

    #[test]
    fn hash_works() {
        let hash_a = blake3_hash::Blake3Hash(blake3::hash(b"a"));
        assert_eq!(
            hash_a.to_string(),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
        );
    }

    #[test]
    fn hash_layer_in_place_works() {
        let hash_a = blake3_hash::Blake3Hash(blake3::hash(b"a"));
        let hash_b = blake3_hash::Blake3Hash(blake3::hash(b"b"));
        let mut layer = vec![hash_a, hash_b];
        MerkleTree::<blake3_hash::Blake3Hasher>::hash_layer_in_place(&mut layer);
        assert_eq!(
            layer[0].to_string(),
            "8912f1e49d6c94830787bc8765e92f409d6db9041739884a42e59f16388756b1"
        )
    }

    #[test]
    fn merkle_hash_works() {
        let leaves = vec![1, 2, 3, 4];
        let root_hex = format!(
            "{}",
            MerkleTree::<blake3_hash::Blake3Hasher>::root_without_caching(&leaves)
        );
        assert_eq!(
            root_hex,
            "e1104b50ab4d8c2ea9a47b7d8be80cc03302e523f0510dfcf383086467306897"
        )
    }

    #[test]
    fn platform_test() {
        println!("{:?}", Platform::detect());
    }

    #[test]
    fn merkle_tree_building() {
        let leaves = vec![1, 2, 3, 4];
        let mut tree: MerkleTree<Blake3Hasher> = MerkleTree::from_leaves(leaves);
        tree.commit();
        assert_eq!(
            tree.root_hex(),
            "e1104b50ab4d8c2ea9a47b7d8be80cc03302e523f0510dfcf383086467306897"
        )
    }
}
