//! Implements a FRI polynomial commitment scheme.
//! This is a protocol where the prover can commit on a set of polynomials and then prove their
//! opening on a set of points.
//! Note: This implementation is not really a polynomial commitment scheme, because we are not in
//! the unique decoding regime. This is enough for a STARK proof though, where we only want to imply
//! the existence of such polynomials, and are ok with having a small decoding list.
//! Note: Opened points cannot come from the commitment domain.
pub mod utils;
use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

pub use self::utils::TreeVec;
use super::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use super::backend::CPUBackend;
use super::channel::Blake2sChannel;
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fri::{
    CirclePolyDegreeBound, FriConfig, FriProof, FriProver, FriVerifier, SparseCircleEvaluation,
};
use super::oods::get_pair_oods_quotient;
use super::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation};
use super::poly::BitReversedOrder;
use super::proof_of_work::{ProofOfWork, ProofOfWorkProof};
use super::prover::{
    VerificationError, LOG_BLOWUP_FACTOR, LOG_LAST_LAYER_DEGREE_BOUND, N_QUERIES,
    PROOF_OF_WORK_BITS,
};
use super::queries::SparseSubCircleDomain;
use super::ColumnVec;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_input::{MerkleTreeColumnLayout, MerkleTreeInput};
use crate::commitment_scheme::mixed_degree_decommitment::MixedDecommitment;
use crate::commitment_scheme::mixed_degree_merkle_tree::MixedDegreeMerkleTree;
use crate::core::channel::Channel;

type MerkleHasher = Blake2sHasher;
type ProofChannel = Blake2sChannel;

/// The prover side of a FRI polynomial commitment scheme. See [self].
pub struct CommitmentSchemeProver {
    pub trees: TreeVec<CommitmentTreeProver>,
    pub log_blowup_factor: u32,
}

impl CommitmentSchemeProver {
    pub fn new(log_blowup_factor: u32) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::<CommitmentTreeProver>::default(),
            log_blowup_factor,
        }
    }

    pub fn commit(&mut self, polynomials: ColumnVec<CPUCirclePoly>, channel: &mut ProofChannel) {
        let tree = CommitmentTreeProver::new(polynomials, self.log_blowup_factor, channel);
        self.trees.push(tree);
    }

    pub fn roots(&self) -> TreeVec<Blake2sHash> {
        self.trees.as_ref().map(|tree| tree.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&CPUCirclePoly>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    fn evaluations(&self) -> TreeVec<ColumnVec<&CPUCircleEvaluation<BaseField, BitReversedOrder>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.evaluations.iter().collect())
    }

    pub fn prove_values(
        &self,
        prove_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut ProofChannel,
    ) -> CommitmentSchemeProof {
        // Evaluate polynomials on open points.
        let proved_values =
            self.polynomials()
                .zip_cols(&prove_points)
                .map_cols(|(poly, points)| {
                    points
                        .iter()
                        .map(|point| poly.eval_at_point(*point))
                        .collect_vec()
                });
        channel.mix_felts(&proved_values.clone().flatten_cols());

        // Compute oods quotients for boundary constraints on prove_points.
        let quotients = self
            .evaluations()
            .zip_cols(&proved_values)
            .zip_cols(&prove_points)
            .map_cols(|((evaluation, values), points)| {
                zip(points, values)
                    .map(|(&point, &value)| {
                        get_pair_oods_quotient(point, value, evaluation).bit_reverse()
                    })
                    .collect_vec()
            });

        // Run FRI commitment phase on the oods quotients.
        let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
        // TODO(spapini): Remove rev() when we start accumulating by size.
        //   This is only done because fri demands descending sizes.
        let fri_prover = FriProver::<CPUBackend, MerkleHasher>::commit(
            channel,
            fri_config,
            &quotients.flatten_cols_rev(),
        );

        // Proof of work.
        let proof_of_work = ProofOfWork::new(PROOF_OF_WORK_BITS).prove(channel);

        // FRI decommitment phase.
        let (fri_proof, fri_query_domains) = fri_prover.decommit(channel);

        // Decommit the FRI queries on the merkle trees.
        let decommitment_results = self.trees.as_ref().map(|tree| {
            tree.decommit(
                fri_query_domains[&(tree.polynomials[0].log_size() + self.log_blowup_factor)]
                    .flatten(),
            )
        });
        let queried_values = decommitment_results.as_ref().map(|(v, _)| v.clone());
        let decommitments = decommitment_results.map(|(_, d)| d);

        CommitmentSchemeProof {
            proved_values,
            decommitments,
            queried_values,
            proof_of_work,
            fri_proof,
        }
    }
}

#[derive(Debug)]
pub struct CommitmentSchemeProof {
    pub proved_values: TreeVec<ColumnVec<Vec<SecureField>>>,
    pub decommitments: TreeVec<MixedDecommitment<BaseField, MerkleHasher>>,
    pub queried_values: TreeVec<ColumnVec<Vec<BaseField>>>,
    pub proof_of_work: ProofOfWorkProof,
    pub fri_proof: FriProof<MerkleHasher>,
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver {
    pub polynomials: ColumnVec<CPUCirclePoly>,
    pub evaluations: ColumnVec<CPUCircleEvaluation<BaseField, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    pub commitment: MixedDegreeMerkleTree<BaseField, Blake2sHasher>,
    column_layout: MerkleTreeColumnLayout,
}

impl CommitmentTreeProver {
    fn new(
        polynomials: ColumnVec<CPUCirclePoly>,
        log_blowup_factor: u32,
        channel: &mut ProofChannel,
    ) -> Self {
        let evaluations = polynomials
            .iter()
            .map(|poly| {
                poly.evaluate(
                    CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain(),
                )
            })
            .collect_vec();

        let mut merkle_input = MerkleTreeInput::new();
        const LOG_N_BASEFIELD_ELEMENTS_IN_SACK: u32 = 4;

        // The desired depth for column of log_length n is such that Blake2s hashes are filled(64B).
        // Explicitly: THere are 2^(d-1) hash 'sacks' at depth d, hence,  with elements of 4 bytes,
        //  2^(d-1) = 2^n / 16, => d = n-3.
        // Assuming rectangle trace, all columns go to the same depth.
        // TOOD(AlonH): remove this assumption.
        let inject_depth = std::cmp::max::<i32>(
            evaluations[0].len().ilog2() as i32 - (LOG_N_BASEFIELD_ELEMENTS_IN_SACK as i32 - 1),
            1,
        );
        for column in evaluations.iter().map(|eval| &eval.values) {
            merkle_input.insert_column(inject_depth as usize, column);
        }
        let (tree, root) = MixedDegreeMerkleTree::<BaseField, Blake2sHasher>::commit(&merkle_input);
        channel.mix_digest(root);

        let column_layout = merkle_input.column_layout();

        CommitmentTreeProver {
            polynomials,
            evaluations,
            commitment: tree,
            column_layout,
        }
    }

    // TODO(AlonH): change interface after integrating mixed degree merkle.
    /// Decommits the merkle tree on the given query positions.
    fn decommit(
        &self,
        queries: Vec<usize>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        MixedDecommitment<BaseField, Blake2sHasher>,
    ) {
        let values = self
            .evaluations
            .iter()
            .map(|c| queries.iter().map(|p| c[*p]).collect())
            .collect();
        // Assuming rectangle trace, queries should be similar for all columns.
        // TOOD(AlonH): remove this assumption.
        let queries = std::iter::repeat(queries.to_vec())
            .take(self.evaluations.len())
            .collect_vec();

        // Rebuild the merkle input for now.
        // TODO(Ohad): change after tree refactor. Consider removing the input struct and have the
        // decommitment take queries and columns only.
        let eval_vec = self
            .evaluations
            .iter()
            .map(|eval| &eval.values[..])
            .collect_vec();
        let input = self.column_layout.build_input(&eval_vec);
        let decommitment = self.commitment.decommit(&input, &queries);
        (values, decommitment)
    }
}

impl Deref for CommitmentTreeProver {
    type Target = MixedDegreeMerkleTree<BaseField, Blake2sHasher>;

    fn deref(&self) -> &Self::Target {
        &self.commitment
    }
}

/// The verifier side of a FRI polynomial commitment scheme. See [self].
#[derive(Default)]
pub struct CommitmentSchemeVerifier {
    pub trees: TreeVec<CommitmentTreeVerifier>,
}

impl CommitmentSchemeVerifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// A [TreeVec<ColumnVec>] of the log sizes of each column in each commitment tree.
    fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        self.trees.as_ref().map(|tree| tree.log_sizes.to_vec())
    }

    /// Reads a commitment from the prover.
    pub fn commit(
        &mut self,
        commitment: Blake2sHash,
        log_sizes: Vec<u32>,
        channel: &mut ProofChannel,
    ) {
        let verifier = CommitmentTreeVerifier::new(commitment, log_sizes, channel);
        self.trees.push(verifier);
    }

    pub fn verify_values(
        &self,
        prove_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof,
        channel: &mut ProofChannel,
    ) -> Result<(), VerificationError> {
        channel.mix_felts(&proof.proved_values.clone().flatten_cols());

        // Compute degree bounds for OODS quotients without looking at the proof.
        let bounds = self
            .column_log_sizes()
            .zip_cols(&prove_points)
            .map_cols(|(log_size, prove_points)| {
                vec![CirclePolyDegreeBound::new(log_size); prove_points.len()]
            })
            .flatten_cols_rev();

        // FRI commitment phase on OODS quotients.
        let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
        let mut fri_verifier =
            FriVerifier::commit(channel, fri_config, proof.fri_proof, bounds).unwrap();

        // Verify proof of work.
        ProofOfWork::new(PROOF_OF_WORK_BITS).verify(channel, &proof.proof_of_work)?;

        // Get FRI query domains.
        let fri_query_domains = fri_verifier.column_opening_positions(channel);

        // Verify merkle decommitments.
        if !self
            .trees
            .as_ref()
            .zip(&proof.decommitments)
            .map(|(tree, decommitment)| {
                // TODO(spapini): Also verify proved_values here.
                // Assuming columns are of equal lengths, replicate queries for all columns.
                // TOOD(AlonH): remove this assumption.
                tree.verify(
                    decommitment,
                    &std::iter::repeat(
                        fri_query_domains[&(tree.log_sizes[0] + LOG_BLOWUP_FACTOR)]
                            .flatten()
                            .clone(),
                    )
                    .take(tree.log_sizes.len())
                    .collect_vec(),
                )
            })
            .iter()
            .all(|x| *x)
        {
            return Err(VerificationError::MerkleVerificationFailed);
        }

        // Answer FRI queries.
        let mut fri_answers = self
            .column_log_sizes()
            .zip_cols(proof.proved_values)
            .zip_cols(prove_points)
            .zip_cols(proof.queried_values)
            .map_cols(
                // For each column.
                |(((log_size, proved_values), opened_points), queried_values)| {
                    zip(opened_points, proved_values)
                        .map(|(point, value)| {
                            // For each opening point of that column.
                            eval_quotients_on_sparse_domain(
                                queried_values.clone(),
                                &fri_query_domains[&(log_size + LOG_BLOWUP_FACTOR)],
                                CanonicCoset::new(log_size + LOG_BLOWUP_FACTOR).circle_domain(),
                                point,
                                value,
                            )
                        })
                        .collect_vec()
                },
            )
            .flatten_cols()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        // TODO(spapini): Remove reverse.
        fri_answers.reverse();
        fri_verifier.decommit(fri_answers)?;
        Ok(())
    }
}

/// Evaluates the oods quotients on the sparse domain.
fn eval_quotients_on_sparse_domain(
    queried_values: Vec<BaseField>,
    query_domains: &SparseSubCircleDomain,
    commitment_domain: CircleDomain,
    point: CirclePoint<SecureField>,
    value: SecureField,
) -> Result<SparseCircleEvaluation<SecureField>, VerificationError> {
    let queried_values = &mut queried_values.into_iter();
    let res = SparseCircleEvaluation::new(
        query_domains
            .iter()
            .map(|subdomain| {
                let values = queried_values.take(1 << subdomain.log_size).collect_vec();
                if values.len() != 1 << subdomain.log_size {
                    return Err(VerificationError::InvalidStructure);
                }
                let subeval =
                    CircleEvaluation::new(subdomain.to_circle_domain(&commitment_domain), values);
                Ok(get_pair_oods_quotient(point, value, &subeval).bit_reverse())
            })
            .collect::<Result<_, _>>()?,
    );
    assert!(
        queried_values.is_empty(),
        "Not all queried values were used"
    );
    Ok(res)
}

/// Verifier data for a single commitment tree in a commitment scheme.
pub struct CommitmentTreeVerifier {
    pub commitment: Blake2sHash,
    pub log_sizes: Vec<u32>,
}

impl CommitmentTreeVerifier {
    pub fn new(commitment: Blake2sHash, log_sizes: Vec<u32>, channel: &mut ProofChannel) -> Self {
        channel.mix_digest(commitment);
        CommitmentTreeVerifier {
            commitment,
            log_sizes,
        }
    }

    pub fn verify(
        &self,
        decommitment: &MixedDecommitment<BaseField, Blake2sHasher>,
        queries: &[Vec<usize>],
    ) -> bool {
        decommitment.verify(
            self.commitment,
            queries,
            decommitment.queried_values.iter().copied(),
        )
    }
}
