use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

#[allow(unused)]
impl MerkleOpsLifted<Poseidon252MerkleHasher> for SimdBackend {
    fn build_leaves(columns: &[&Col<Self, BaseField>]) -> Col<Self, <Poseidon252MerkleHasher as crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted>::Hash>{
        unimplemented!()
    }

    fn build_next_layer(prev_layer: &Col<Self, <Poseidon252MerkleHasher as crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted>::Hash>) -> Col<Self, <Poseidon252MerkleHasher as crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted>::Hash>{
        unimplemented!()
    }
}
