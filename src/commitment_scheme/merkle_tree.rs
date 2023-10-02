use super::hasher::Hasher;

pub struct MerkleTree<T: Hasher> {
    pub data: Vec<Vec<T::Hash>>,
    pub leaves: Vec<u32>,
    height: usize,
}

impl<T: Hasher> MerkleTree<T> {
    // Construct a new merkle tree.
    // Does not commit, hence does not hash at all.
    pub fn from_leaves(leaves: Vec<u32>) -> Self {
        let tree_height = (leaves.len() as f32).log2().ceil() as usize;
        Self {
            data: (Vec::with_capacity(tree_height)),
            leaves,
            height: (tree_height),
        }
    }

    // Hashes recursively. a 2^n layer will cause a total of (2^(n+1) - 1) hashes
    pub fn commit(&mut self) {
        // Hash leaves individually.
        let mut bottom_layer: Vec<T::Hash> = Vec::with_capacity(self.leaves.len());
        for i in 0..self.leaves.len() {
            bottom_layer.push(T::hash(&self.leaves[i].to_be_bytes()));
        }
        self.data.push(bottom_layer);

        // Build rest of the tree.
        for i in 0..self.height {
            let new_layer = Self::hash_layer(&self.data[i]);
            self.data.push(new_layer);
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

    #[inline(always)]
    pub fn hash_layer_in_place(layer: &mut [T::Hash]) {
        for i in 0..(layer.len() >> 1) {
            layer[i] = T::concat_and_hash(&layer[i * 2], &layer[i * 2 + 1]);
        }
    }
    #[inline(always)]
    pub fn root_without_caching(leaves: &[u32]) -> T::Hash {
        // Hash first layer
        let mut curr_layer: Vec<T::Hash> =
            leaves.iter().map(|n| T::hash(&n.to_be_bytes())).collect();

        // Hash rest of tree
        let mut i = leaves.len();
        while i > 1 {
            Self::hash_layer_in_place(&mut curr_layer[..i]);
            i >>= 1;
        }

        curr_layer[0]
    }
}
