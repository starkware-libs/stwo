//! Implements a FRI polynomial commitment scheme.
//! This is a protocol where the prover can commit on a set of polynomials and then prove their
//! opening on a set of points.
//! Note: This implementation is not really a polynomial commitment scheme, because we are not in
//! the unique decoding regime. This is enough for a STARK proof though, where we onyl want to imply
//! the existence of such polynomials, and re ok with having a small decoding list.

mod utils;

use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

pub use self::utils::{TreeColumns, TreeVec};
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
    LOG_BLOWUP_FACTOR, LOG_LAST_LAYER_DEGREE_BOUND, N_QUERIES, PROOF_OF_WORK_BITS,
};
use super::queries::SparseSubCircleDomain;
use super::ColumnVec;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

type MerkleHasher = Blake2sHasher;
type ProofChannel = Blake2sChannel;

/// The prover size of a FRI polynomial commitment scheme. See [self].
pub struct CommitmentSchemeProver {
    pub trees: TreeVec<CommitmentTreeProver>,
    pub log_blowup_factor: u32,
}

impl CommitmentSchemeProver {
    pub fn new(log_blowup_factor: u32) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::<CommitmentTreeProver>::new(),
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

    pub fn polys(&self) -> TreeColumns<&CPUCirclePoly> {
        self.trees.to_cols(|tree| tree.polynomials.iter().collect())
    }

    fn evaluations(&self) -> TreeColumns<&CPUCircleEvaluation<BaseField, BitReversedOrder>> {
        self.trees.to_cols(|tree| tree.evaluations.iter().collect())
    }

    pub fn open_values(
        &self,
        open_points: TreeColumns<Vec<CirclePoint<SecureField>>>,
        channel: &mut ProofChannel,
    ) -> CommitmentSchemeProof {
        // Evaluate polynomials on open points.
        let opened_values = self.polys().zip_cols(&open_points).map(|(poly, points)| {
            points
                .iter()
                .map(|point| poly.eval_at_point(*point))
                .collect_vec()
        });

        // Compute oods quotients for boundary constraints on open_points.
        let quotients = self
            .evaluations()
            .zip_cols(&opened_values)
            .zip_cols(&open_points)
            .map(|((evaluation, values), points)| {
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
            &quotients.flatten_all_rev(),
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
        let queried_values = decommitment_results.to_cols(|(v, _)| v.clone());
        let decommitments = decommitment_results.map(|(_, d)| d);

        CommitmentSchemeProof {
            opened_values,
            decommitments,
            queried_values,
            proof_of_work,
            fri_proof,
        }
    }
}

pub struct CommitmentSchemeProof {
    pub opened_values: TreeColumns<Vec<SecureField>>,
    pub decommitments: TreeVec<MerkleDecommitment<BaseField, MerkleHasher>>,
    pub queried_values: TreeColumns<Vec<BaseField>>,
    pub proof_of_work: ProofOfWorkProof,
    pub fri_proof: FriProof<MerkleHasher>,
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver {
    pub polynomials: ColumnVec<CPUCirclePoly>,
    pub evaluations: ColumnVec<CPUCircleEvaluation<BaseField, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    commitment: MerkleTree<BaseField, MerkleHasher>,
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
        let commitment = MerkleTree::<BaseField, MerkleHasher>::commit(
            evaluations
                .iter()
                .map(|eval| eval.values.clone())
                .collect_vec(),
        );
        channel.mix_digest(commitment.root());

        CommitmentTreeProver {
            polynomials,
            evaluations,
            commitment,
        }
    }

    // TODO(AlonH): change interface after integrating mixed degree merkle.
    /// Decommits the merkle tree on the given query positions.
    fn decommit(
        &self,
        queries: Vec<usize>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        MerkleDecommitment<BaseField, MerkleHasher>,
    ) {
        let values = self
            .evaluations
            .iter()
            .map(|c| queries.iter().map(|p| c[*p]).collect())
            .collect();
        let decommitment = self.commitment.generate_decommitment(queries);
        (values, decommitment)
    }
}

impl Deref for CommitmentTreeProver {
    type Target = MerkleTree<BaseField, MerkleHasher>;

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

    /// A [TreeColumns] of the log sizes of each column in each commitment tree.
    fn column_log_sizes(&self) -> TreeColumns<u32> {
        self.trees.to_cols(|tree| tree.log_sizes.to_vec())
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

    pub fn verify_opening(
        &self,
        open_points: TreeColumns<Vec<CirclePoint<SecureField>>>,
        proof: CommitmentSchemeProof,
        channel: &mut ProofChannel,
    ) -> bool {
        // Compute degree bounds for oods quotients without looking at the proof.
        let bounds = self
            .column_log_sizes()
            .zip_cols(&open_points)
            .map(|(log_size, open_points)| {
                vec![CirclePolyDegreeBound::new(log_size); open_points.len()]
            })
            .flatten_all_rev();

        // FRI commitment phase on oods quotients.
        let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
        let mut fri_verifier =
            FriVerifier::commit(channel, fri_config, proof.fri_proof, bounds).unwrap();

        // Verify proof of work.
        ProofOfWork::new(PROOF_OF_WORK_BITS).verify(channel, &proof.proof_of_work);

        // Get FRI query domains.
        let fri_query_domains = fri_verifier.column_opening_positions(channel);

        // Verify merkle decommitments.
        if !self
            .trees
            .as_ref()
            .zip(&proof.decommitments)
            .map(|(tree, decommitment)| {
                // TODO(spapini): Also very opened_value here.
                tree.verify(
                    decommitment,
                    &fri_query_domains[&(tree.log_sizes[0] + 1)].flatten(),
                )
            })
            .0
            .iter()
            .all(|x| *x)
        {
            return false;
        }

        // Answer FRI queries.
        let mut fri_answers = self
            .column_log_sizes()
            .zip_cols(proof.opened_values)
            .zip_cols(open_points)
            .zip_cols(proof.queried_values)
            .map(
                // For each column.
                |(((log_size, opened_values), opened_points), queried_values)| {
                    zip(opened_points, opened_values)
                        .map(|(point, value)| {
                            // For each opening point of that column.
                            eval_quotients_on_sparse_domain(
                                queried_values.clone(),
                                &fri_query_domains[&(log_size + 1)],
                                CanonicCoset::new(log_size + 1).circle_domain(),
                                point,
                                value,
                            )
                        })
                        .collect_vec()
                },
            )
            .flatten_all();

        // TODO(spapini): Remove reverse.
        fri_answers.reverse();
        fri_verifier.decommit(fri_answers).is_ok()
    }
}

fn eval_quotients_on_sparse_domain(
    queried_values: Vec<BaseField>,
    query_domains: &SparseSubCircleDomain,
    commitment_domain: CircleDomain,
    point: CirclePoint<SecureField>,
    value: SecureField,
) -> SparseCircleEvaluation<SecureField> {
    let queried_values = &mut queried_values.clone().into_iter();
    let res = SparseCircleEvaluation::new(
        query_domains
            .iter()
            .map(|subdomain| {
                let subeval = CircleEvaluation::new(
                    subdomain.to_circle_domain(&commitment_domain),
                    queried_values.take(1 << subdomain.log_size).collect(),
                );
                get_pair_oods_quotient(point, value, &subeval).bit_reverse()
            })
            .collect(),
    );
    assert!(queried_values.is_empty());
    res
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
        decommitment: &MerkleDecommitment<BaseField, MerkleHasher>,
        positions: &[usize],
    ) -> bool {
        decommitment.verify(self.commitment, positions)
    }
}
