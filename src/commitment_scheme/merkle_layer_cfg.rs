pub struct MerkleLayerConfig {
    pub n_sub_trees: usize,
    pub sub_tree_height: usize,
    pub sub_tree_size: usize,
}
impl MerkleLayerConfig {
    pub fn new(n_sub_trees: usize, sub_tree_height: usize) -> Self {
        assert!(n_sub_trees.is_power_of_two());
        let sub_tree_size = (1 << sub_tree_height) - 1;

        Self {
            n_sub_trees,
            sub_tree_height,
            sub_tree_size,
        }
    }
}
