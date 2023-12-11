use super::{hasher::Hasher, merkle_polylayer::MerklePolyLayer};
use super::merkle_input::MerkleTreeInput;
use crate::commitment_scheme::merkle_polylayer_cfg::MerklePolyLayerConfig;
use crate::core::fields::{Field, IntoSlice};

/// A Merkle tree
/// Hash result tree is divided into two layer-groups.
/// The first layer-group is the top most layers of the tree.
pub struct MerkleTree<'a, F: Field, H: Hasher> {
    _input: Vec<MerkleTreeInput<'a, F>>, // Remove '_' when used(decommitment).
    layers: [MerklePolyLayer<H>; 2],
}

impl<'a, F: Field + Sync, H: Hasher> MerkleTree<'a, F, H> where F: IntoSlice<H::NativeType> {
    pub fn commit(mut input: MerkleTreeInput<'a,F>, lower_layer_height:usize) -> Self {
        assert!(lower_layer_height <= input.max_injected_depth());
        let higher_layer_height = input.max_injected_depth() - lower_layer_height;
        let l1_n_sub_trees = 1 << (input.max_injected_depth() - lower_layer_height);

        let l1_input = input.split(higher_layer_height + 1);

        // TODO: make a function.
        let l1_cfg = MerklePolyLayerConfig::new(l1_n_sub_trees,lower_layer_height);
        let l0_cfg = MerklePolyLayerConfig::new(1, higher_layer_height);
        let mut layer1 = MerklePolyLayer::<H>::new(l1_cfg);
        let mut layer0 = MerklePolyLayer::<H>::new(l0_cfg);
        layer1.commit_layer_mt(&l1_input, None);
        layer0.commit_layer_mt(&input, Some(layer1.get_root_layer().as_slice()));
        
        Self{
            _input: vec![input,l1_input],
            layers: [layer0,layer1],
        }
    }

    pub fn root(&self) -> H::Hash {
        self.layers[0].get_root_layer()[0]
    }
}


#[cfg(test)]
mod tests {
    use crate::{core::fields::m31::M31, commitment_scheme::{blake3_hash::Blake3Hasher, merkle_input::MerkleTreeInput, merkle_tree}};

    fn _init_m31_test_trace(len: usize) -> Vec<M31> {
        assert!(len.is_power_of_two());
        (0..len as u32).map(M31::from_u32_unchecked).collect()
    }

    #[test]
    pub fn commit_test() {
        let trace_column = _init_m31_test_trace(64); //vec![M31::from_u32_unchecked(0);64];
        let mut input = MerkleTreeInput::new();
        input.insert_column(5, &trace_column);
        input.insert_column(5, &trace_column);
        input.insert_column(5, &trace_column);
        input.insert_column(5, &trace_column);

        let tree = super::MerkleTree::<M31,Blake3Hasher>::commit(input, 4); // BUG here: split must be lower then max_depth.
        assert_eq!(
            hex::encode(tree.root()),
            "633790d87efd0e345f5c724295a00a45d7df4380a6b95c996f186575b2c13491"
        );
    }

    #[test]
    pub fn compare_to_old_merkle_tree() {
        let trace_column = vec![M31::from_u32_unchecked(0);256]; //vec![M31::from_u32_unchecked(0);64];
        let mut input = MerkleTreeInput::new();
        input.insert_column(5, &trace_column);
  

        let tree = super::MerkleTree::<M31,Blake3Hasher>::commit(input, 4);
        let old_tree = merkle_tree::MerkleTree::<M31,Blake3Hasher>::commit(vec![trace_column.clone()]);
        assert_eq!(
            hex::encode(tree.root()),
            hex::encode(old_tree.root())
        );
    }
}