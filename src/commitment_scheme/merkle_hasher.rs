use std::slice::Iter;

use super::hasher::Hasher;

/// Allows the logic of the Merkle-Tree hash input aggregation to be abstracted away from the
/// Merkle-Tree, and implemented by the specific hash function.
/// Usefull for when the hash function has some sort of parallelism degree 'N', and needs to
/// collect N hash inputs at a time.
pub trait MerkleHasher<T: Sized>: Hasher {
    fn inject_and_compress_layer_in_place(
        prev_hashes: Option<&[Self::Hash]>,
        dst: &mut [Self::Hash],
        columns: &Iter<'_, &[T]>,
    );
}
