use hashbrown::HashMap;
use itertools::Itertools;
use rand::{CryptoRng, RngCore};
use std_shims::BTreeSet;
use tracing::{span, Level};

use super::ops::MerkleOpsLifted;
use crate::core::fields::m31::{BaseField, P as M31_MODULUS};
use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{
    ExtendedMerkleDecommitmentLifted, MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux,
};
use crate::core::ColumnVec;
use crate::prover::backend::{Col, Column};

/// Represents the prover side of a Merkle commitment scheme.
#[derive(Debug)]
pub struct MerkleProverLifted<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    /// Layers of the Merkle tree, sorted by increasing length.
    /// The first layer is a column of length 1, containing the root commitment.
    pub layers: Vec<Col<B, H::Hash>>,
    leaf_salts: Option<Vec<Vec<BaseField>>>,
}

impl<B: MerkleOpsLifted<H>, H: MerkleHasherLifted> MerkleProverLifted<B, H> {
    /// Commits to columns.
    /// Columns must be of power of 2 sizes, not necessarily sorted by length.
    ///
    /// # Arguments
    ///
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A new instance of `MerkleProverLifted` with the committed layers.
    pub fn commit(
        columns: Vec<&Col<B, BaseField>>,
        lifting_log_size: u32,
        log_rows_per_leaf: u32,
    ) -> Self {
        let _span = span!(Level::TRACE, "Merkle", class = "MerkleCommitment").entered();
        if columns.is_empty() {
            return Self {
                layers: vec![B::build_leaves(&[], lifting_log_size)],
                leaf_salts: None,
            };
        }

        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();
        // We enter this branch only during FRI commit phase, in which we commit 4 columns of the
        // same size. In particular, we don't need to sort the columns by size.
        if log_rows_per_leaf > 0 {
            // TODO(Leo): add support for higher log_rows_per_leaf sizes.
            assert_eq!(
                log_rows_per_leaf, 2,
                "Leaf packing is only supported for log_rows_per_leaf = 2."
            );
            let columns: [&Col<B, BaseField>; SECURE_EXTENSION_DEGREE] =
                columns.try_into().unwrap();
            let packed_columns = B::pack_leaves_input(&columns);
            let max_log_size = packed_columns[0].len().ilog2();
            assert!(lifting_log_size >= max_log_size);
            layers.push(B::build_leaves(
                &packed_columns.iter().collect_vec(),
                lifting_log_size,
            ));
        } else {
            let sorted_columns = columns.into_iter().sorted_by_key(|c| c.len()).collect_vec();
            let max_log_size = sorted_columns.last().unwrap().len().ilog2();
            assert!(lifting_log_size >= max_log_size);
            layers.push(B::build_leaves(&sorted_columns, lifting_log_size));
        }

        (0..lifting_log_size).for_each(|_| {
            layers.push(B::build_next_layer(layers.last().unwrap()));
        });
        layers.reverse();

        Self {
            layers,
            leaf_salts: None,
        }
    }

    pub fn commit_hiding<R>(
        columns: Vec<&Col<B, BaseField>>,
        lifting_log_size: u32,
        log_rows_per_leaf: u32,
        salt_felts_per_leaf: u32,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng + ?Sized,
    {
        let _span = span!(Level::TRACE, "Merkle", class = "MerkleHidingCommitment").entered();
        assert!(salt_felts_per_leaf > 0);
        if columns.is_empty() {
            return Self {
                layers: vec![B::build_leaves(&[], lifting_log_size)],
                leaf_salts: None,
            };
        }

        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();
        let leaf_salts;
        if log_rows_per_leaf > 0 {
            assert_eq!(
                log_rows_per_leaf, 2,
                "Leaf packing is only supported for log_rows_per_leaf = 2."
            );
            let columns: [&Col<B, BaseField>; SECURE_EXTENSION_DEGREE] =
                columns.try_into().unwrap();
            let packed_columns = B::pack_leaves_input(&columns);
            let max_log_size = packed_columns[0].len().ilog2();
            assert!(lifting_log_size >= max_log_size);
            let (leaves, salts) = Self::build_hiding_leaves(
                &packed_columns.iter().collect_vec(),
                lifting_log_size,
                salt_felts_per_leaf,
                rng,
            );
            layers.push(leaves);
            leaf_salts = salts;
        } else {
            let sorted_columns = columns.into_iter().sorted_by_key(|c| c.len()).collect_vec();
            let max_log_size = sorted_columns.last().unwrap().len().ilog2();
            assert!(lifting_log_size >= max_log_size);
            let (leaves, salts) = Self::build_hiding_leaves(
                &sorted_columns,
                lifting_log_size,
                salt_felts_per_leaf,
                rng,
            );
            layers.push(leaves);
            leaf_salts = salts;
        }

        (0..lifting_log_size).for_each(|_| {
            layers.push(B::build_next_layer(layers.last().unwrap()));
        });
        layers.reverse();

        Self {
            layers,
            leaf_salts: Some(leaf_salts),
        }
    }

    fn build_hiding_leaves<R>(
        columns: &[&Col<B, BaseField>],
        lifting_log_size: u32,
        salt_felts_per_leaf: u32,
        rng: &mut R,
    ) -> (Col<B, H::Hash>, Vec<Vec<BaseField>>)
    where
        R: RngCore + CryptoRng + ?Sized,
    {
        assert!(!columns.is_empty());
        assert!(columns[0].len() >= 2, "A column must be of length >= 2.");
        let hasher = H::default();
        let mut prev_layer: Vec<H> = vec![hasher; 2];
        let mut prev_layer_log_size: u32 = 1;
        for (log_size, group) in columns.iter().group_by(|c| c.len().ilog2()).into_iter() {
            let log_ratio = log_size - prev_layer_log_size;
            prev_layer = (0..1 << log_size)
                .map(|idx| prev_layer[(idx >> (log_ratio + 1) << 1) + (idx & 1)].clone())
                .collect();

            for chunk in &group.into_iter().chunks(16) {
                let cols = chunk.into_iter().copied().collect_vec();
                prev_layer.iter_mut().enumerate().for_each(|(i, hasher)| {
                    hasher.update_leaf(&cols.iter().map(|v| v.at(i)).collect_vec());
                })
            }
            prev_layer_log_size = log_size;
        }

        let log_ratio = lifting_log_size - prev_layer_log_size;
        if log_ratio > 0 {
            prev_layer = (0..1 << lifting_log_size)
                .map(|idx| prev_layer[(idx >> (log_ratio + 1) << 1) + (idx & 1)].clone())
                .collect();
        }

        let leaf_salts = (0..prev_layer.len())
            .map(|_| sample_leaf_salt(salt_felts_per_leaf, rng))
            .collect_vec();
        let leaves = prev_layer
            .into_iter()
            .zip_eq(&leaf_salts)
            .map(|(mut hasher, salt)| {
                hasher.update_leaf(salt);
                hasher.finalize()
            })
            .collect();

        (leaves, leaf_salts)
    }

    pub fn salt_felts_per_leaf(&self) -> u32 {
        self.leaf_salts
            .as_ref()
            .and_then(|salts| salts.first())
            .map(|salt| salt.len() as u32)
            .unwrap_or(0)
    }

    pub fn is_hiding(&self) -> bool {
        self.leaf_salts.is_some()
    }

    /// Decommits to columns on the given queries.
    /// Queries are given as indices to the largest column.
    ///
    /// # Arguments
    ///
    /// * `queries_position` - Vector containing the positions of the queries, in increasing order.
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A vector of queried values. For each query position, the queried values are column values
    ///   corresponding to the query position, sorted increasingly by column length.
    /// * A `MerkleDecommitment` containing the hash witness.
    pub fn decommit(
        &self,
        query_positions: &[usize],
        columns: Vec<&Col<B, BaseField>>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        ExtendedMerkleDecommitmentLifted<H>,
    ) {
        // Prepare output buffers.
        let mut queried_values: ColumnVec<Vec<BaseField>> = vec![];
        let mut decommitment = MerkleDecommitmentLifted::<H>::default();
        if let Some(leaf_salts) = &self.leaf_salts {
            decommitment.leaf_salts = BTreeSet::from_iter(query_positions.iter().copied())
                .into_iter()
                .map(|pos| leaf_salts[pos].clone())
                .collect();
        }
        let mut all_node_values: Vec<HashMap<usize, <H as MerkleHasherLifted>::Hash>> = vec![];

        // Compute the queried values.
        let max_log_size = self.layers.len() - 1;
        for col in columns.iter() {
            let log_size = col.len().ilog2() as usize;
            let shift = max_log_size - log_size;
            let res: Vec<_> = query_positions
                .iter()
                .map(|pos| col.at((pos >> (shift + 1) << 1) + (pos & 1)))
                .collect();
            queried_values.push(res);
        }

        let mut prev_layer_queries = BTreeSet::from_iter(query_positions.iter().copied())
            .into_iter()
            .collect_vec();
        // The largest log size of a layer is equal to `self.layers.len() - 1`. We start iterating
        // from the layer of log size `self.layers.len() - 2` so that we always have a previous
        // layer available for the computation.
        for layer_log_size in (0..self.layers.len() - 1).rev() {
            let mut all_node_values_for_layer =
                HashMap::<usize, <H as MerkleHasherLifted>::Hash>::new();
            // Prepare write buffer for queries to the current layer. This will propagate to the
            // next layer.
            let mut curr_layer_queries: Vec<usize> = vec![];

            // Each layer node is a hash of column values as previous layer hashes.
            // Prepare the previous layer hashes to read from.
            let prev_layer_hashes = self.layers.get(layer_log_size + 1).unwrap();
            // All chunks have either length 1 (only one child is present) or 2 (both children are
            // present).
            for queries_chunk in prev_layer_queries.as_slice().chunk_by(|a, b| a ^ 1 == *b) {
                let first = queries_chunk[0];
                // If the brother of `first` was not queried before, add its hash to the witness.
                if queries_chunk.len() == 1 {
                    decommitment
                        .hash_witness
                        .push(prev_layer_hashes.at(first ^ 1))
                }
                let curr_index = first >> 1;
                curr_layer_queries.push(curr_index);

                // Add the previous layer hashes to all_node_values.
                all_node_values_for_layer
                    .insert(2 * curr_index, prev_layer_hashes.at(2 * curr_index));
                all_node_values_for_layer
                    .insert(2 * curr_index + 1, prev_layer_hashes.at(2 * curr_index + 1));
            }
            // Propagate queries to the next layer.
            prev_layer_queries = curr_layer_queries;

            all_node_values.push(all_node_values_for_layer);
        }
        (
            queried_values,
            ExtendedMerkleDecommitmentLifted {
                decommitment,
                aux: MerkleDecommitmentLiftedAux { all_node_values },
            },
        )
    }

    pub fn root(&self) -> H::Hash {
        self.layers.first().unwrap().at(0)
    }
}

fn sample_leaf_salt<R>(salt_felts_per_leaf: u32, rng: &mut R) -> Vec<BaseField>
where
    R: RngCore + ?Sized,
{
    (0..salt_felts_per_leaf)
        .map(|_| sample_base_field(rng))
        .collect()
}

fn sample_base_field<R>(rng: &mut R) -> BaseField
where
    R: RngCore + ?Sized,
{
    loop {
        let candidate = rng.next_u32() & 0x7fff_ffff;
        if candidate < M31_MODULUS {
            return BaseField::from_u32_unchecked(candidate);
        }
    }
}

#[cfg(test)]
mod test {
    use num_traits::Zero;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;
    use crate::core::fields::m31::M31;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher as Blake2sMerkleHasherCurrent;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs_lifted::test_utils::lift_poly;
    use crate::core::vcs_lifted::verifier::{MerkleVerificationError, MerkleVerifierLifted};
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::CpuBackend;
    use crate::prover::vcs::prover::MerkleProver;

    #[test]
    fn test_empty_cols() {
        // Check Merkle commitment on empty columns.
        let mixed_degree_merkle_prover =
            MerkleProver::<CpuBackend, Blake2sMerkleHasherCurrent>::commit(vec![]);
        let lifted_merkle_prover =
            MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(vec![], 0, 0);
        assert_eq!(
            mixed_degree_merkle_prover.layers,
            lifted_merkle_prover.layers
        );
    }

    fn prepare_merkle() -> (
        Vec<Vec<BaseField>>,
        MerkleProverLifted<CpuBackend, Blake2sHasher>,
    ) {
        let max_log_size = 4;
        let columns: Vec<Vec<BaseField>> = (2..=max_log_size)
            .map(|i| (0..1 << i).map(M31::from_u32_unchecked).collect())
            .collect();
        let merkle_prover = MerkleProverLifted::<CpuBackend, Blake2sHasher>::commit(
            columns.iter().collect(),
            max_log_size,
            0,
        );
        (columns, merkle_prover)
    }

    fn prepare_hiding_merkle(
        seed: u64,
    ) -> (
        Vec<Vec<BaseField>>,
        MerkleProverLifted<CpuBackend, Blake2sHasher>,
    ) {
        let max_log_size = 4;
        let columns: Vec<Vec<BaseField>> = (2..=max_log_size)
            .map(|i| (0..1 << i).map(M31::from_u32_unchecked).collect())
            .collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let merkle_prover = MerkleProverLifted::<CpuBackend, Blake2sHasher>::commit_hiding(
            columns.iter().collect(),
            max_log_size,
            0,
            4,
            &mut rng,
        );
        (columns, merkle_prover)
    }

    #[test]
    fn test_lifted_merkle_leaves() {
        let (_, merkle_prover) = prepare_merkle();
        let leaves = &merkle_prover.layers.last().unwrap();

        // Compute the expected first leaf.
        let mut hasher = Blake2sHasher::default();
        let data = [0u8; 12];
        hasher.update(&data);
        assert_eq!(hasher.finalize(), leaves[0]);

        // Compute the expected fifth leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = Vec::new();
        data.extend(0_u32.to_le_bytes());
        data.extend(2_u32.to_le_bytes());
        data.extend(4_u32.to_le_bytes());
        hasher.update(&data);
        assert_eq!(hasher.finalize(), leaves[4]);

        // Compute the expected last leaf.
        let mut hasher = Blake2sHasher::default();
        let mut data = Vec::new();
        data.extend(3_u32.to_le_bytes());
        data.extend(7_u32.to_le_bytes());
        data.extend(15_u32.to_le_bytes());
        hasher.update(&data);

        assert_eq!(hasher.finalize(), *leaves.last().unwrap());
    }

    #[test]
    fn test_hiding_lifted_merkle_opens_salted_leaves() {
        let (cols, merkle_prover) = prepare_hiding_merkle(7);
        let queries = vec![0, 4, 15];
        let (queried_values, decommitment) =
            merkle_prover.decommit(&queries, cols.iter().collect_vec());

        assert_eq!(decommitment.decommitment.leaf_salts.len(), queries.len());
        assert!(decommitment
            .decommitment
            .leaf_salts
            .iter()
            .all(|salt| salt.len() == 4));

        let verifier =
            MerkleVerifierLifted::new_with_leaf_salts(merkle_prover.root(), vec![2, 3, 4], None, 4);
        verifier
            .verify(
                &queries,
                queried_values.clone(),
                decommitment.decommitment.clone(),
            )
            .unwrap();

        let transparent_verifier =
            MerkleVerifierLifted::new(merkle_prover.root(), vec![2, 3, 4], None);
        assert_eq!(
            transparent_verifier
                .verify(
                    &queries,
                    queried_values.clone(),
                    decommitment.decommitment.clone()
                )
                .unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );

        let mut tampered = decommitment.decommitment;
        tampered.leaf_salts[0][0] += BaseField::from(1);
        assert_eq!(
            verifier
                .verify(&queries, queried_values, tampered)
                .unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_hiding_lifted_merkle_same_columns_change_root() {
        let (_, first) = prepare_hiding_merkle(11);
        let (_, second) = prepare_hiding_merkle(12);

        assert_ne!(first.root(), second.root());
    }

    #[test]
    fn test_lifted_decommitted_values() {
        let (cols, merkle_prover) = prepare_merkle();
        // Test decommits at position 0.
        let queried_values = merkle_prover.decommit(&[0], cols.iter().collect_vec()).0;

        let expected_values = vec![vec![BaseField::zero()]; 3];
        assert_eq!(expected_values, queried_values);

        // Test decommits at position 4.
        let queried_values = merkle_prover.decommit(&[4], cols.iter().collect_vec()).0;
        let expected_values = vec![
            vec![BaseField::from_u32_unchecked(0)],
            vec![BaseField::from_u32_unchecked(2)],
            vec![BaseField::from_u32_unchecked(4)],
        ];
        assert_eq!(expected_values, queried_values);

        // Test decommits at position 15.
        let queried_values = merkle_prover.decommit(&[15], cols.iter().collect_vec()).0;
        let expected_values = vec![
            vec![BaseField::from_u32_unchecked(3)],
            vec![BaseField::from_u32_unchecked(7)],
            vec![BaseField::from_u32_unchecked(15)],
        ];
        assert_eq!(expected_values, queried_values);
    }

    /// See the docs of `[crate::prover::backend::cpu::blake2s_lifted::build_leaves]`.
    #[test]
    fn test_bit_reverse_lifted_merkle_cpu() {
        const LOG_SIZE: u32 = 3;
        const LIFTED_LOG_SIZE: u32 = 9;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let poly = CpuCirclePoly::new((0..1 << LOG_SIZE).map(BaseField::from).collect());
        let lifted_evaluation = lift_poly(&poly, LIFTED_LOG_SIZE);

        let last_column: Col<CpuBackend, BaseField> =
            (0..1 << LIFTED_LOG_SIZE).map(|_| M31::zero()).collect_vec();

        let mixed_degree_merkle_prover =
            MerkleProver::<CpuBackend, Blake2sMerkleHasherCurrent>::commit(vec![
                &lifted_evaluation.values,
                &last_column,
            ]);
        let lifted_merkle_prover_1 = MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(
            vec![&lifted_evaluation.values, &last_column],
            LIFTED_LOG_SIZE,
            0,
        );
        let lifted_merkle_prover_2 = MerkleProverLifted::<CpuBackend, Blake2sMerkleHasher>::commit(
            vec![&poly.evaluate(domain), &last_column],
            LIFTED_LOG_SIZE,
            0,
        );

        assert_eq!(lifted_merkle_prover_1.root(), lifted_merkle_prover_2.root());
        assert_eq!(
            mixed_degree_merkle_prover.root(),
            lifted_merkle_prover_1.root()
        );
    }

    #[test]
    fn test_decommitment_aux() {
        let (columns, merkle_prover) = prepare_merkle();
        let (
            _,
            ExtendedMerkleDecommitmentLifted {
                decommitment: _,
                aux,
            },
        ) = merkle_prover.decommit(&[1], columns.iter().collect_vec());

        let mut expected: Vec<HashMap<usize, Blake2sHash>> = vec![];
        merkle_prover
            .layers
            .iter()
            .skip(1)
            .rev()
            .for_each(|layer| expected.push(HashMap::from_iter([(0, layer[0]), (1, layer[1])])));
        assert_eq!(expected, aux.all_node_values);
    }
}
