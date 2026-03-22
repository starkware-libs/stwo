//! Generates [`CommitmentSchemeProofAux`] from a [`CommitmentSchemeProof`].
//!
//! Replays verification to reconstruct per-query authentication path data from the
//! multi-query Merkle decommitment (shared hash witnesses).

use core::iter::zip;

use hashbrown::HashMap;
use itertools::{zip_eq, Itertools};
use std_shims::Vec;

use super::quotients::{fri_answers, CommitmentSchemeProof, CommitmentSchemeProofAux, PointSample};
use super::utils::{prepare_preprocessed_query_positions, TreeVec};
use super::CommitmentSchemeVerifier;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, QM31, SECURE_EXTENSION_DEGREE};
use crate::core::fri::{
    fold_circle_into_line, fold_coset, FriConfig, FriLayerProofAux, FriProof, FriProofAux,
};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::poly::line::LineDomain;
use crate::core::queries::{draw_queries, Queries};
use crate::core::utils::bit_reverse_index;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{
    MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux, MerkleVerifierLifted,
    LOG_PACKED_LEAF_SIZE,
};
use crate::core::verifier::VerificationError;
use crate::core::ColumnVec;

/// Generates [`CommitmentSchemeProofAux`] from a [`CommitmentSchemeProof`].
///
/// The channel must be in the same state as when [`CommitmentSchemeVerifier::verify_values`]
/// would be called (i.e., after the commitment phase and OODS point drawing).
///
/// This replays the same channel interactions as verification, reconstructing:
/// - `unsorted_query_locations`: the raw query positions before sorting/dedup.
/// - `trace_decommitment`: per-layer Merkle node hash maps for each commitment tree.
/// - `fri`: per-layer FRI value maps and Merkle node hashes.
pub fn generate_aux<MC: MerkleChannel>(
    verifier: &CommitmentSchemeVerifier<MC>,
    sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    proof: CommitmentSchemeProof<MC::H>,
    channel: &mut MC::C,
) -> Result<CommitmentSchemeProofAux<MC::H>, VerificationError> {
    // Advance channel identically to verify_values.
    channel.mix_felts(&proof.sampled_values.clone().flatten_cols());
    let random_coeff = channel.draw_secure_felt();
    let lifting_log_size = verifier.trees.last().unwrap().height;
    let bound_log_degree = lifting_log_size - verifier.config.fri_config.log_blowup_factor;

    // Replay FRI commit phase to collect folding alphas and layer configs.
    let fri_info = replay_fri_commit::<MC>(
        channel,
        &verifier.config.fri_config,
        &proof.fri_proof,
        bound_log_degree,
    );

    // Proof of work.
    if !channel.verify_pow_nonce(verifier.config.pow_bits, proof.proof_of_work) {
        return Err(VerificationError::ProofOfWork);
    }
    channel.mix_u64(proof.proof_of_work);

    // Draw query positions.
    let unsorted_query_locations = draw_queries(
        channel,
        lifting_log_size,
        verifier.config.fri_config.n_queries,
    );
    let queries = Queries::new(&unsorted_query_locations, lifting_log_size);
    let query_positions = queries.positions.clone();

    // Build query position tree (preprocessed tree may use different positions).
    let preprocessed_query_positions = prepare_preprocessed_query_positions(
        &query_positions,
        lifting_log_size,
        verifier.trees[0].height,
    );

    // Generate trace decommitment aux for each tree.
    let trace_decommitment: Vec<MerkleDecommitmentLiftedAux<MC::H>> = verifier
        .trees
        .iter()
        .zip(proof.decommitments.iter())
        .zip(proof.queried_values.iter())
        .enumerate()
        .map(|(i, ((tree, decommitment), queried_values))| {
            let qp = if i == 0 {
                &preprocessed_query_positions
            } else {
                &query_positions
            };
            reconstruct_merkle_aux(tree, qp, queried_values, decommitment)
        })
        .collect();

    // Compute FRI answers (first layer query evals).
    let column_log_sizes = verifier
        .trees
        .as_ref()
        .map(|tree| tree.column_log_sizes.clone());
    let samples = sampled_points.zip_cols(proof.sampled_values).map_cols(
        |(sampled_points, sampled_values)| {
            zip(sampled_points, sampled_values)
                .map(|(point, value)| PointSample { point, value })
                .collect_vec()
        },
    );
    let first_layer_query_evals = fri_answers(
        column_log_sizes,
        samples,
        random_coeff,
        &query_positions,
        proof.queried_values,
        lifting_log_size,
    )?;

    // Generate FRI aux.
    let fri_aux = reconstruct_fri_aux::<MC::H>(
        &verifier.config.fri_config,
        &proof.fri_proof,
        &queries,
        &first_layer_query_evals,
        &fri_info,
    );

    Ok(CommitmentSchemeProofAux {
        unsorted_query_locations,
        trace_decommitment: TreeVec(trace_decommitment),
        fri: fri_aux,
    })
}

// =============================================================================
// Merkle aux reconstruction
// =============================================================================

/// Reconstructs [`MerkleDecommitmentLiftedAux`] by replaying Merkle verification
/// and recording all intermediate node hashes.
fn reconstruct_merkle_aux<H: MerkleHasherLifted>(
    verifier: &MerkleVerifierLifted<H>,
    query_positions: &[usize],
    queried_values: &ColumnVec<Vec<BaseField>>,
    decommitment: &MerkleDecommitmentLifted<H>,
) -> MerkleDecommitmentLiftedAux<H> {
    if verifier.height == 0 {
        return MerkleDecommitmentLiftedAux {
            all_node_values: vec![],
        };
    }

    // Sort queries by column log size and deduplicate (same as MerkleVerifierLifted::verify).
    let mut sorted_queries_iter = queried_values
        .iter()
        .zip_eq(verifier.column_log_sizes.iter())
        .sorted_by_key(|(_, col_size)| *col_size)
        .map(|(vals, _)| {
            vals.iter()
                .enumerate()
                .dedup_by(|(idx1, _), (idx2, _)| query_positions[*idx1] == query_positions[*idx2])
                .map(|(_, val)| val)
        })
        .collect_vec();

    // Build leaf hashes.
    let mut prev_layer_hashes: Vec<(usize, H::Hash)> = vec![];
    for pos in query_positions.iter().dedup() {
        let row: Vec<_> = sorted_queries_iter
            .iter_mut()
            .map(|col_iter| *col_iter.next().unwrap())
            .collect();
        let mut hasher = H::default();
        hasher.update_leaf(&row);
        prev_layer_hashes.push((*pos, hasher.finalize()));
    }

    let mut hash_witness = decommitment.hash_witness.iter();
    let mut all_node_values = vec![];

    // Process each layer bottom-to-top, recording both children for every visited parent.
    for _ in 0..verifier.height {
        let mut layer_node_values = HashMap::new();
        let mut curr_layer_hashes: Vec<(usize, H::Hash)> = vec![];

        for chunk in prev_layer_hashes.as_slice().chunk_by(|a, b| a.0 ^ 1 == b.0) {
            let (idx_0, hash_0) = chunk[0];
            let parent = idx_0 >> 1;
            let children = if chunk.len() == 1 {
                // Sibling not queried — take from witness.
                let witness = *hash_witness.next().unwrap();
                match idx_0 & 1 {
                    0 => (hash_0, witness),
                    1 => (witness, hash_0),
                    _ => unreachable!(),
                }
            } else {
                // Both siblings queried.
                let (_, hash_1) = chunk[1];
                (hash_0, hash_1)
            };
            layer_node_values.insert(2 * parent, children.0);
            layer_node_values.insert(2 * parent + 1, children.1);
            curr_layer_hashes.push((parent, H::hash_children(children)));
        }

        all_node_values.push(layer_node_values);
        prev_layer_hashes = curr_layer_hashes;
    }

    MerkleDecommitmentLiftedAux { all_node_values }
}

// =============================================================================
// FRI commit replay
// =============================================================================

struct FriCommitInfo {
    column_commitment_domain: CircleDomain,
    first_folding_alpha: SecureField,
    first_fold_step: u32,
    first_pack_leaves: bool,
    inner_layers: Vec<InnerLayerInfo>,
}

struct InnerLayerInfo {
    domain: LineDomain,
    folding_alpha: SecureField,
    fold_step: u32,
    pack_leaves: bool,
}

/// Replays the FRI commitment phase, advancing the channel identically to
/// `FriVerifier::commit` and collecting the folding alphas and layer configurations.
fn replay_fri_commit<MC: MerkleChannel>(
    channel: &mut MC::C,
    config: &FriConfig,
    fri_proof: &FriProof<MC::H>,
    column_bound_log_degree: u32,
) -> FriCommitInfo {
    // First layer.
    MC::mix_root(channel, fri_proof.first_layer.commitment);
    let column_commitment_domain =
        CanonicCoset::new(column_bound_log_degree + config.log_blowup_factor).circle_domain();
    let first_folding_alpha = channel.draw_secure_felt();
    let first_fold_step = config.fold_step;
    let first_pack_leaves =
        column_commitment_domain.log_size() >= LOG_PACKED_LEAF_SIZE && config.fold_step > 1;

    // Inner layers.
    let mut layer_bound_log = column_bound_log_degree
        .checked_sub(config.fold_step)
        .expect("fold_step exceeds column degree bound");
    let mut layer_domain =
        LineDomain::new(Coset::half_odds(layer_bound_log + config.log_blowup_factor));

    let n_inner = fri_proof.inner_layers.len();
    let mut inner_layers = Vec::with_capacity(n_inner);
    for (i, layer_proof) in fri_proof.inner_layers.iter().enumerate() {
        MC::mix_root(channel, layer_proof.commitment);
        let is_last = i == n_inner - 1;
        let fold_step = if !is_last {
            config.fold_step
        } else {
            layer_bound_log - config.log_last_layer_degree_bound
        };
        let folding_alpha = channel.draw_secure_felt();
        let pack_leaves = layer_domain.log_size() >= LOG_PACKED_LEAF_SIZE && fold_step > 1;

        inner_layers.push(InnerLayerInfo {
            domain: layer_domain,
            folding_alpha,
            fold_step,
            pack_leaves,
        });

        layer_bound_log -= fold_step;
        layer_domain = layer_domain.repeated_double(fold_step);
    }

    // Last layer poly.
    channel.mix_felts(&fri_proof.last_layer_poly);

    FriCommitInfo {
        column_commitment_domain,
        first_folding_alpha,
        first_fold_step,
        first_pack_leaves,
        inner_layers,
    }
}

// =============================================================================
// FRI aux reconstruction
// =============================================================================

/// Reconstructs [`FriProofAux`] by replaying FRI decommitment.
fn reconstruct_fri_aux<H: MerkleHasherLifted>(
    config: &FriConfig,
    fri_proof: &FriProof<H>,
    queries: &Queries,
    first_layer_query_evals: &[SecureField],
    fri_info: &FriCommitInfo,
) -> FriProofAux<H> {
    // === First layer ===
    let (first_decomm_positions, first_subset_evals, first_subset_initials) =
        decompose_layer_witness(
            queries,
            first_layer_query_evals,
            &fri_proof.first_layer.fri_witness,
            fri_info.first_fold_step,
        );

    let first_layer_aux = build_fri_layer_aux(
        &first_decomm_positions,
        &first_subset_evals,
        &fri_proof.first_layer.decommitment,
        fri_proof.first_layer.commitment,
        fri_info.column_commitment_domain.log_size(),
        fri_info.first_pack_leaves,
    );

    // Fold first layer evals for the first inner layer.
    let mut layer_queries = queries.fold(config.fold_step);
    let mut layer_query_evals = fold_circle_sparse(
        first_subset_evals,
        first_subset_initials,
        fri_info.first_folding_alpha,
        fri_info.column_commitment_domain,
        fri_info.first_fold_step,
    );

    // === Inner layers ===
    let mut inner_layers_aux = Vec::with_capacity(fri_info.inner_layers.len());
    for (layer_info, layer_proof) in zip_eq(&fri_info.inner_layers, &fri_proof.inner_layers) {
        let (decomm_positions, subset_evals, subset_initials) = decompose_layer_witness(
            &layer_queries,
            &layer_query_evals,
            &layer_proof.fri_witness,
            layer_info.fold_step,
        );

        let layer_aux = build_fri_layer_aux(
            &decomm_positions,
            &subset_evals,
            &layer_proof.decommitment,
            layer_proof.commitment,
            layer_info.domain.log_size(),
            layer_info.pack_leaves,
        );

        // Fold for next layer.
        layer_query_evals = fold_line_sparse(
            subset_evals,
            subset_initials,
            layer_info.folding_alpha,
            layer_info.domain,
            layer_info.fold_step,
        );
        layer_queries = layer_queries.fold(layer_info.fold_step);

        inner_layers_aux.push(layer_aux);
    }

    FriProofAux {
        first_layer: first_layer_aux,
        inner_layers: inner_layers_aux,
    }
}

/// Decomposes a FRI layer's witness into decommitment positions, subset evaluations,
/// and subset domain initial indexes. Mirrors the verifier's
/// `compute_decommitment_positions_and_rebuild_evals`.
fn decompose_layer_witness(
    queries: &Queries,
    query_evals: &[SecureField],
    fri_witness: &[SecureField],
    fold_step: u32,
) -> (Vec<usize>, Vec<Vec<SecureField>>, Vec<usize>) {
    let mut query_evals_iter = query_evals.iter().copied();
    let mut witness_iter = fri_witness.iter().copied();

    let mut decommitment_positions = Vec::new();
    let mut subset_evals = Vec::new();
    let mut subset_initials = Vec::new();

    for subset_queries in queries.chunk_by(|a, b| a >> fold_step == b >> fold_step) {
        let subset_start = (subset_queries[0] >> fold_step) << fold_step;
        let subset_range = subset_start..subset_start + (1 << fold_step);
        decommitment_positions.extend(subset_range.clone());

        let mut subset_query_iter = subset_queries.iter().copied().peekable();
        let eval: Vec<_> = subset_range
            .map(|pos| match subset_query_iter.next_if_eq(&pos) {
                Some(_) => query_evals_iter.next().unwrap(),
                None => witness_iter.next().unwrap(),
            })
            .collect();

        subset_evals.push(eval);
        subset_initials.push(bit_reverse_index(subset_start, queries.log_domain_size));
    }

    (decommitment_positions, subset_evals, subset_initials)
}

/// Builds [`FriLayerProofAux`] from decommitment data.
fn build_fri_layer_aux<H: MerkleHasherLifted>(
    decommitment_positions: &[usize],
    subset_evals: &[Vec<SecureField>],
    decommitment: &MerkleDecommitmentLifted<H>,
    commitment: H::Hash,
    domain_log_size: u32,
    pack_leaves: bool,
) -> FriLayerProofAux<H> {
    // Build value map: position → QM31 for all decommitment positions.
    let value_map: HashMap<usize, QM31> = decommitment_positions
        .iter()
        .zip(subset_evals.iter().flatten())
        .map(|(&pos, &val)| (pos, val))
        .collect();

    // Build Merkle verification inputs (reshape for packed leaves).
    let leaf_log_size = if pack_leaves { LOG_PACKED_LEAF_SIZE } else { 0 };
    let (merkle_positions, merkle_values) = build_merkle_verification_inputs(
        decommitment_positions,
        subset_evals.iter().flatten().copied(),
        leaf_log_size,
    );

    // Reconstruct Merkle aux.
    let merkle_verifier = MerkleVerifierLifted::new(
        commitment,
        vec![domain_log_size - leaf_log_size; SECURE_EXTENSION_DEGREE * (1 << leaf_log_size)],
        None,
    );
    let merkle_aux = reconstruct_merkle_aux(
        &merkle_verifier,
        &merkle_positions,
        &merkle_values,
        decommitment,
    );

    FriLayerProofAux {
        all_values: vec![value_map],
        decommitment: merkle_aux,
    }
}

/// Folds circle-domain sparse evaluations (first FRI layer: circle → line).
fn fold_circle_sparse(
    subset_evals: Vec<Vec<SecureField>>,
    subset_initials: Vec<usize>,
    fold_alpha: SecureField,
    source_domain: CircleDomain,
    fold_step: u32,
) -> Vec<SecureField> {
    assert!(fold_step >= 1);
    zip(subset_evals, subset_initials)
        .map(|(eval, domain_initial_index)| {
            let fold_domain_initial = source_domain.index_at(domain_initial_index);
            let circle_fold_domain =
                CircleDomain::new(Coset::new(fold_domain_initial, fold_step - 1));
            let buffer = fold_circle_into_line(&eval, circle_fold_domain, fold_alpha);
            if fold_step == 1 {
                buffer[0]
            } else {
                let line_fold_step = fold_step - 1;
                let line_fold_domain =
                    LineDomain::new(Coset::new(fold_domain_initial, line_fold_step));
                let alpha_sq = fold_alpha * fold_alpha;
                fold_coset(buffer, line_fold_domain, alpha_sq)
            }
        })
        .collect()
}

/// Folds line-domain sparse evaluations (inner FRI layers).
fn fold_line_sparse(
    subset_evals: Vec<Vec<SecureField>>,
    subset_initials: Vec<usize>,
    fold_alpha: SecureField,
    source_domain: LineDomain,
    fold_step: u32,
) -> Vec<SecureField> {
    zip(subset_evals, subset_initials)
        .map(|(eval, domain_initial_index)| {
            let fold_domain_initial = source_domain.coset().index_at(domain_initial_index);
            let fold_domain = LineDomain::new(Coset::new(fold_domain_initial, fold_step));
            fold_coset(eval, fold_domain, fold_alpha)
        })
        .collect()
}

/// Reshapes decommitment positions and values for Merkle verification.
/// Mirrors the FRI verifier's `build_merkle_verification_inputs`.
fn build_merkle_verification_inputs(
    decommitment_positions: &[usize],
    mut values: impl Iterator<Item = SecureField>,
    leaf_log_size: u32,
) -> (Vec<usize>, Vec<Vec<BaseField>>) {
    let leaf_size = 1 << leaf_log_size;
    let merkle_positions: Vec<usize> = decommitment_positions
        .iter()
        .map(|pos| pos >> leaf_log_size)
        .dedup()
        .collect();
    let mut merkle_values =
        vec![Vec::with_capacity(merkle_positions.len()); SECURE_EXTENSION_DEGREE * leaf_size];
    for _ in &merkle_positions {
        for offset in 0..leaf_size {
            let coords = values.next().unwrap().to_m31_array();
            for (coord_index, value) in coords.into_iter().enumerate() {
                merkle_values[coord_index + offset * SECURE_EXTENSION_DEGREE].push(value);
            }
        }
    }
    (merkle_positions, merkle_values)
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::generate_aux;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
    use crate::core::pcs::utils::TreeVec;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use crate::core::test_utils::test_channel;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::verifier::COMPOSITION_LOG_SPLIT;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::CpuBackend;
    use crate::prover::poly::circle::PolyOps;
    use crate::prover::CommitmentSchemeProver;

    /// Proves and then reconstructs aux, comparing against the prover's aux.
    #[test]
    fn test_generate_aux_matches_prover() {
        const LOG_N_ROWS: u32 = 8;
        const LOG_BLOWUP: u32 = 2;
        let config = PcsConfig {
            pow_bits: 0,
            fri_config: crate::core::fri::FriConfig::new(0, LOG_BLOWUP, 3, 1),
            lifting_log_size: Some(LOG_N_ROWS + LOG_BLOWUP),
        };

        // Set up a simple proof with one polynomial.
        let mut prover_channel = test_channel();
        let twiddles = CpuBackend::precompute_twiddles(
            crate::core::poly::circle::CanonicCoset::new(LOG_N_ROWS + LOG_BLOWUP)
                .circle_domain()
                .half_coset,
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sMerkleChannel>::new(config, &twiddles);

        // Commit a column.
        let poly = CpuCirclePoly::new(
            (0..1u32 << LOG_N_ROWS)
                .map(crate::core::fields::m31::BaseField::from)
                .collect(),
        );
        let polys = vec![poly];
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(polys);
        tree_builder.commit(&mut prover_channel);

        // Commit composition polynomial.
        let composition_log_degree = LOG_N_ROWS;
        let split_log_degree = composition_log_degree - COMPOSITION_LOG_SPLIT;
        let comp_polys: Vec<_> = (0..2 * SECURE_EXTENSION_DEGREE)
            .map(|_| {
                CpuCirclePoly::new(
                    (0..1u32 << split_log_degree)
                        .map(crate::core::fields::m31::BaseField::from)
                        .collect(),
                )
            })
            .collect();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(comp_polys);
        tree_builder.commit(&mut prover_channel);

        // Build sample points (one per column, at a random point).
        let oods_point = CirclePoint::<SecureField>::get_random_point(&mut prover_channel);
        let lifting_log_size = commitment_scheme
            .trees
            .last()
            .unwrap()
            .commitment
            .layers
            .len() as u32
            - 1;
        let sample_points = TreeVec::new(vec![
            vec![vec![oods_point.repeated_double(
                lifting_log_size - (LOG_N_ROWS + LOG_BLOWUP),
            )]],
            vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE],
        ]);

        // Prove.
        let extended_proof =
            commitment_scheme.prove_values(sample_points.clone(), &mut prover_channel);
        let prover_aux = extended_proof.aux;
        let proof = extended_proof.proof;

        // Set up verifier.
        let mut verifier_channel = test_channel();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(proof.commitments[0], &[LOG_N_ROWS], &mut verifier_channel);
        verifier.commit(
            proof.commitments[1],
            &[split_log_degree; 2 * SECURE_EXTENSION_DEGREE],
            &mut verifier_channel,
        );
        let _oods_point = CirclePoint::<SecureField>::get_random_point(&mut verifier_channel);

        // Generate aux from proof.
        let reconstructed_aux =
            generate_aux(&verifier, sample_points, proof, &mut verifier_channel).unwrap();

        // Compare.
        assert_eq!(
            prover_aux.unsorted_query_locations, reconstructed_aux.unsorted_query_locations,
            "unsorted_query_locations mismatch"
        );

        // Compare trace decommitment aux.
        for (tree_idx, (prover_tree_aux, recon_tree_aux)) in prover_aux
            .trace_decommitment
            .iter()
            .zip(reconstructed_aux.trace_decommitment.iter())
            .enumerate()
        {
            assert_eq!(
                prover_tree_aux.all_node_values.len(),
                recon_tree_aux.all_node_values.len(),
                "tree {tree_idx}: layer count mismatch"
            );
            for (layer, (prover_layer, recon_layer)) in prover_tree_aux
                .all_node_values
                .iter()
                .zip(recon_tree_aux.all_node_values.iter())
                .enumerate()
            {
                assert_eq!(
                    prover_layer, recon_layer,
                    "tree {tree_idx}, layer {layer}: node values mismatch"
                );
            }
        }

        // Compare FRI aux.
        assert_eq!(
            prover_aux.fri.first_layer.all_values, reconstructed_aux.fri.first_layer.all_values,
            "FRI first layer all_values mismatch"
        );
        assert_eq!(
            prover_aux.fri.first_layer.decommitment.all_node_values,
            reconstructed_aux
                .fri
                .first_layer
                .decommitment
                .all_node_values,
            "FRI first layer merkle aux mismatch"
        );
        for (i, (prover_inner, recon_inner)) in prover_aux
            .fri
            .inner_layers
            .iter()
            .zip(reconstructed_aux.fri.inner_layers.iter())
            .enumerate()
        {
            assert_eq!(
                prover_inner.all_values, recon_inner.all_values,
                "FRI inner layer {i} all_values mismatch"
            );
            assert_eq!(
                prover_inner.decommitment.all_node_values, recon_inner.decommitment.all_node_values,
                "FRI inner layer {i} merkle aux mismatch"
            );
        }
    }

    /// Tests with fold_step=2 which exercises packed leaf logic.
    #[test]
    fn test_generate_aux_with_packed_leaves() {
        const LOG_N_ROWS: u32 = 10;
        const LOG_BLOWUP: u32 = 2;
        let config = PcsConfig {
            pow_bits: 0,
            fri_config: crate::core::fri::FriConfig::new(0, LOG_BLOWUP, 5, 2),
            lifting_log_size: Some(LOG_N_ROWS + LOG_BLOWUP),
        };

        let mut prover_channel = test_channel();
        let twiddles = CpuBackend::precompute_twiddles(
            crate::core::poly::circle::CanonicCoset::new(LOG_N_ROWS + LOG_BLOWUP)
                .circle_domain()
                .half_coset,
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sMerkleChannel>::new(config, &twiddles);

        let poly = CpuCirclePoly::new(
            (0..1u32 << LOG_N_ROWS)
                .map(crate::core::fields::m31::BaseField::from)
                .collect(),
        );
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(vec![poly]);
        tree_builder.commit(&mut prover_channel);

        let composition_log_degree = LOG_N_ROWS;
        let split_log_degree = composition_log_degree - COMPOSITION_LOG_SPLIT;
        let comp_polys: Vec<_> = (0..2 * SECURE_EXTENSION_DEGREE)
            .map(|_| {
                CpuCirclePoly::new(
                    (0..1u32 << split_log_degree)
                        .map(crate::core::fields::m31::BaseField::from)
                        .collect(),
                )
            })
            .collect();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(comp_polys);
        tree_builder.commit(&mut prover_channel);

        let oods_point = CirclePoint::<SecureField>::get_random_point(&mut prover_channel);
        let lifting_log_size = commitment_scheme
            .trees
            .last()
            .unwrap()
            .commitment
            .layers
            .len() as u32
            - 1;
        let sample_points = TreeVec::new(vec![
            vec![vec![oods_point.repeated_double(
                lifting_log_size - (LOG_N_ROWS + LOG_BLOWUP),
            )]],
            vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE],
        ]);

        let extended_proof =
            commitment_scheme.prove_values(sample_points.clone(), &mut prover_channel);
        let prover_aux = extended_proof.aux;
        let proof = extended_proof.proof;

        let mut verifier_channel = test_channel();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(proof.commitments[0], &[LOG_N_ROWS], &mut verifier_channel);
        verifier.commit(
            proof.commitments[1],
            &[split_log_degree; 2 * SECURE_EXTENSION_DEGREE],
            &mut verifier_channel,
        );
        let _oods_point = CirclePoint::<SecureField>::get_random_point(&mut verifier_channel);

        let reconstructed_aux =
            generate_aux(&verifier, sample_points, proof, &mut verifier_channel).unwrap();

        assert_eq!(
            prover_aux.unsorted_query_locations,
            reconstructed_aux.unsorted_query_locations,
        );
        for (tree_idx, (p, r)) in prover_aux
            .trace_decommitment
            .iter()
            .zip(reconstructed_aux.trace_decommitment.iter())
            .enumerate()
        {
            assert_eq!(
                p.all_node_values, r.all_node_values,
                "tree {tree_idx} mismatch"
            );
        }
        assert_eq!(
            prover_aux.fri.first_layer.all_values,
            reconstructed_aux.fri.first_layer.all_values,
        );
        assert_eq!(
            prover_aux.fri.first_layer.decommitment.all_node_values,
            reconstructed_aux
                .fri
                .first_layer
                .decommitment
                .all_node_values,
        );
        for (i, (p, r)) in prover_aux
            .fri
            .inner_layers
            .iter()
            .zip(reconstructed_aux.fri.inner_layers.iter())
            .enumerate()
        {
            assert_eq!(p.all_values, r.all_values, "FRI inner layer {i}");
            assert_eq!(
                p.decommitment.all_node_values, r.decommitment.all_node_values,
                "FRI inner layer {i} merkle"
            );
        }
    }
}
