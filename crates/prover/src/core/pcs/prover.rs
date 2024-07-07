use std::collections::BTreeMap;

use itertools::Itertools;
use tracing::{span, Level};

use super::super::channel::Blake2sChannel;
use super::super::circle::CirclePoint;
use super::super::fields::m31::BaseField;
use super::super::fields::qm31::SecureField;
use super::super::fri::{FriConfig, FriProof, FriProver};
use super::super::poly::circle::CanonicCoset;
use super::super::poly::BitReversedOrder;
use super::super::proof_of_work::{ProofOfWork, ProofOfWorkProof};
use super::super::prover::{
    LOG_BLOWUP_FACTOR, LOG_LAST_LAYER_DEGREE_BOUND, N_QUERIES, PROOF_OF_WORK_BITS,
};
use super::super::ColumnVec;
use super::quotients::{compute_fri_quotients, PointSample};
use super::utils::TreeVec;
use crate::core::backend::Backend;
use crate::core::channel::Channel;
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::ops::MerkleOps;
use crate::core::vcs::prover::{MerkleDecommitment, MerkleProver};

type MerkleHasher = Blake2sMerkleHasher;
type ProofChannel = Blake2sChannel;

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<B: Backend + MerkleOps<MerkleHasher>> {
    pub trees: TreeVec<CommitmentTreeProver<B>>,
    pub log_blowup_factor: u32,
}

impl<B: Backend + MerkleOps<MerkleHasher>> CommitmentSchemeProver<B> {
    pub fn new(log_blowup_factor: u32) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            log_blowup_factor,
        }
    }

    pub fn commit(
        &mut self,
        polynomials: ColumnVec<CirclePoly<B>>,
        channel: &mut ProofChannel,
        twiddles: &TwiddleTree<B>,
    ) {
        let _span = span!(Level::INFO, "Commitment").entered();
        let tree =
            CommitmentTreeProver::new(polynomials, self.log_blowup_factor, channel, twiddles);
        self.trees.push(tree);
    }

    pub fn commit_on_evals(
        &mut self,
        evals: ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
        channel: &mut Blake2sChannel,
        twiddles: &TwiddleTree<B>,
    ) {
        let span = span!(Level::INFO, "Interpolation").entered();
        let polys = evals
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect();
        span.exit();
        self.commit(polys, channel, twiddles);
    }

    pub fn roots(&self) -> TreeVec<Blake2sHash> {
        self.trees.as_ref().map(|tree| tree.commitment.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&CirclePoly<B>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    fn evaluations(&self) -> TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.evaluations.iter().collect())
    }

    pub fn prove_values(
        &self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut ProofChannel,
        twiddles: &TwiddleTree<B>,
    ) -> CommitmentSchemeProof {
        // Evaluate polynomials on open points.
        let span = span!(Level::INFO, "Evaluate columns out of domain").entered();
        let samples = self
            .polynomials()
            .zip_cols(&sampled_points)
            .map_cols(|(poly, points)| {
                points
                    .iter()
                    .map(|&point| PointSample {
                        point,
                        value: poly.eval_at_point(point),
                    })
                    .collect_vec()
            });
        span.exit();
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        // Compute oods quotients for boundary constraints on the sampled points.
        let columns = self.evaluations().flatten();
        let quotients = compute_fri_quotients(&columns, &samples.flatten(), channel.draw_felt());

        // Run FRI commitment phase on the oods quotients.
        let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
        let fri_prover =
            FriProver::<B, MerkleHasher>::commit(channel, fri_config, &quotients, twiddles);

        // Proof of work.
        let proof_of_work = ProofOfWork::new(PROOF_OF_WORK_BITS).prove(channel);

        // FRI decommitment phase.
        let (fri_proof, fri_query_domains) = fri_prover.decommit(channel);

        // Decommit the FRI queries on the merkle trees.
        let decommitment_results = self.trees.as_ref().map(|tree| {
            let queries = fri_query_domains
                .iter()
                .map(|(&log_size, domain)| (log_size, domain.flatten()))
                .collect();
            tree.decommit(queries)
        });

        let queried_values = decommitment_results.as_ref().map(|(v, _)| v.clone());
        let decommitments = decommitment_results.map(|(_, d)| d);

        CommitmentSchemeProof {
            sampled_values,
            decommitments,
            queried_values,
            proof_of_work,
            fri_proof,
        }
    }
}

#[derive(Debug)]
pub struct CommitmentSchemeProof {
    pub sampled_values: TreeVec<ColumnVec<Vec<SecureField>>>,
    pub decommitments: TreeVec<MerkleDecommitment<MerkleHasher>>,
    pub queried_values: TreeVec<ColumnVec<Vec<BaseField>>>,
    pub proof_of_work: ProofOfWorkProof,
    pub fri_proof: FriProof<MerkleHasher>,
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver<B: Backend + MerkleOps<MerkleHasher>> {
    pub polynomials: ColumnVec<CirclePoly<B>>,
    pub evaluations: ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    pub commitment: MerkleProver<B, MerkleHasher>,
}

impl<B: Backend + MerkleOps<MerkleHasher>> CommitmentTreeProver<B> {
    fn new(
        polynomials: ColumnVec<CirclePoly<B>>,
        log_blowup_factor: u32,
        channel: &mut ProofChannel,
        twiddles: &TwiddleTree<B>,
    ) -> Self {
        let span = span!(Level::INFO, "Extension").entered();
        let evaluations = polynomials
            .iter()
            .map(|poly| {
                poly.evaluate_with_twiddles(
                    CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain(),
                    twiddles,
                )
            })
            .collect_vec();

        span.exit();

        let _span = span!(Level::INFO, "Merkle").entered();
        let tree = MerkleProver::commit(evaluations.iter().map(|eval| &eval.values).collect());
        channel.mix_digest(tree.root());

        CommitmentTreeProver {
            polynomials,
            evaluations,
            commitment: tree,
        }
    }

    /// Decommits the merkle tree on the given query positions.
    /// Returns the values at the queried positions and the decommitment.
    /// The queries are given as a mapping from the log size of the layer size to the queried
    /// positions on each column of that size.
    fn decommit(
        &self,
        queries: BTreeMap<u32, Vec<usize>>,
    ) -> (ColumnVec<Vec<BaseField>>, MerkleDecommitment<MerkleHasher>) {
        let eval_vec = self
            .evaluations
            .iter()
            .map(|eval| &eval.values)
            .collect_vec();
        self.commitment.decommit(queries, eval_vec)
    }
}
