use byteorder;
use byteorder::{BigEndian, ByteOrder};

use super::hasher::Hasher;
use super::N_BYTES_FELT;
use crate::math;

pub struct MerkleTree<T: Hasher> {
    pub data: Vec<Vec<T::Hash>>,
    pub height: usize,
}

impl<T: Hasher> MerkleTree<T> {
    /// Constructs a new merkle tree.
    /// Hashes recursively.
    // TODO(Ohad): deal with mixed degree, taking columns instead of a stream.
    pub fn commit(elems: &[u32]) -> Self {
        let elems_in_block = T::BLOCK_SIZE_IN_BYTES / N_BYTES_FELT;
        let bottom_layer_length = math::usize_div_ceil(elems.len(), elems_in_block);
        let tree_height = math::log2_ceil(bottom_layer_length) + 1;

        let mut data: Vec<Vec<T::Hash>> = Vec::with_capacity(tree_height);

        // Concatenate elements to T::BLOCK_SIZE byte blocks and hash.
        let mut bottom_layer: Vec<T::Hash> = Vec::with_capacity(bottom_layer_length);

        for i in (0..elems.len()).step_by(elems_in_block) {
            let slice_size = std::cmp::min(elems_in_block, elems.len() - i);
            let mut block: [u8; 64] = [0; 64];

            // elems.len() might not be a multiple of elems_in_block
            BigEndian::write_u32_into(&elems[i..i + slice_size], &mut block[..(slice_size * 4)]);
            bottom_layer.push(T::hash(&block));
        }
        data.push(bottom_layer);

        // Build rest of the tree, every layer is composed of the 2-to-1 result of a
        // pair of neighbors from the previous layer.
        for i in 1..tree_height {
            let new_layer = Self::hash_layer(&data[i - 1]);
            data.push(new_layer);
        }

        Self {
            data,
            height: tree_height,
        }
    }

    pub fn root_hex(&mut self) -> String {
        format!(
            "{}",
            self.data
                .last()
                .expect("Attempted access to uncomitted tree")[0]
        )
    }

    fn hash_layer(layer: &[T::Hash]) -> Vec<T::Hash> {
        let mut res = Vec::with_capacity(layer.len() >> 1);
        for i in 0..(layer.len() >> 1) {
            res.push(T::concat_and_hash(&layer[i * 2], &layer[i * 2 + 1]));
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::merkle_tree::MerkleTree;

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
