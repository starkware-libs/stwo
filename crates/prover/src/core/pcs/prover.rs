use std::collections::{BTreeMap, HashMap};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tracing::{span, Level};

use super::super::circle::CirclePoint;
use super::super::fields::m31::BaseField;
use super::super::fields::qm31::SecureField;
use super::super::fri::{FriProof, FriProver};
use super::super::poly::BitReversedOrder;
use super::super::ColumnVec;
use super::quotients::{compute_fri_quotients, PointSample};
use super::utils::TreeVec;
use super::{PcsConfig, TreeSubspan};
use crate::core::air::{Poly, Trace};
use crate::core::backend::{BackendForChannel, Col};
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::prover::{MerkleDecommitment, MerkleProver};

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<'a, B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub trees: TreeVec<CommitmentTreeProver<B, MC>>,
    pub config: PcsConfig,
    twiddles: &'a TwiddleTree<B>,
    pub store_polynomials_coefficients: bool,
}

impl<'a, B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentSchemeProver<'a, B, MC> {
    pub fn new(
        config: PcsConfig,
        twiddles: &'a TwiddleTree<B>,
        store_polynomials_coefficients: bool,
    ) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients,
        }
    }

    fn commit(&mut self, polynomials: ColumnVec<CirclePoly<B>>, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        let tree = CommitmentTreeProver::new(
            polynomials,
            self.config.fri_config.log_blowup_factor,
            channel,
            self.twiddles,
            self.store_polynomials_coefficients,
        );
        self.trees.push(tree);
    }

    pub fn tree_builder(&mut self) -> TreeBuilder<'_, 'a, B, MC> {
        TreeBuilder {
            tree_index: self.trees.len(),
            commitment_scheme: self,
            polys: Vec::default(),
        }
    }

    pub fn roots(&self) -> TreeVec<<MC::H as MerkleHasher>::Hash> {
        self.trees.as_ref().map(|tree| tree.commitment.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&Poly<B>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    pub fn trace(&self) -> Trace<'_, B> {
        Trace {
            polys: self.polynomials(),
        }
    }

    pub fn build_weights_hash_map(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    ) -> HashMap<(u32, CirclePoint<SecureField>), Col<B, SecureField>> {
        let mut weights_hash_map = HashMap::new();
        self.polynomials()
            .zip_cols(sampled_points)
            .map_cols(|(poly, points)| {
                points.iter().for_each(|&point| {
                    let log_size = poly.eval.domain.log_size();
                    weights_hash_map
                        .entry((log_size, point))
                        .or_insert_with(|| {
                            CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(
                                CanonicCoset::new(log_size),
                                point,
                            )
                        });
                })
            });
        weights_hash_map
    }

    pub fn prove_values(
        self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> CommitmentSchemeProof<MC::H> {
        // Evaluate polynomials on open points.
        let span = span!(Level::INFO, "Evaluate columns out of domain").entered();
        let weights_hash_map = self.build_weights_hash_map(&sampled_points);
        let samples = self
            .polynomials()
            .zip_cols(&sampled_points)
            .map_cols(|(poly, points)| {
                points
                    .iter()
                    .map(|&point| PointSample {
                        point,
                        value: poly.eval_at_point(
                            point,
                            &weights_hash_map[&(poly.eval.domain.log_size(), point)],
                        ),
                    })
                    .collect_vec()
            });
        span.exit();
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        // Compute oods quotients for boundary constraints on the sampled points.
        let columns: ColumnVec<_> = self
            .polynomials()
            .flatten()
            .into_iter()
            .map(|poly| &poly.eval)
            .collect();

        let quotients = compute_fri_quotients(
            &columns,
            &samples.flatten(),
            channel.draw_felt(),
            self.config.fri_config.log_blowup_factor,
        );

        // Run FRI commitment phase on the oods quotients.
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let (fri_proof, query_positions_per_log_size) = fri_prover.decommit(channel);

        // Decommit the FRI queries on the merkle trees.
        let decommitment_results = self
            .trees
            .as_ref()
            .map(|tree| tree.decommit(&query_positions_per_log_size));

        let queried_values = decommitment_results.as_ref().map(|(v, _)| v.clone());
        let decommitments = decommitment_results.map(|(_, d)| d);

        CommitmentSchemeProof {
            commitments: self.roots(),
            sampled_values,
            decommitments,
            queried_values,
            proof_of_work,
            fri_proof,
            config: self.config,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentSchemeProof<H: MerkleHasher> {
    pub config: PcsConfig,
    pub commitments: TreeVec<H::Hash>,
    pub sampled_values: TreeVec<ColumnVec<Vec<SecureField>>>,
    pub decommitments: TreeVec<MerkleDecommitment<H>>,
    pub queried_values: TreeVec<Vec<BaseField>>,
    pub proof_of_work: u64,
    pub fri_proof: FriProof<H>,
}

pub struct TreeBuilder<'a, 'b, B: BackendForChannel<MC>, MC: MerkleChannel> {
    tree_index: usize,
    commitment_scheme: &'a mut CommitmentSchemeProver<'b, B, MC>,
    polys: ColumnVec<CirclePoly<B>>,
}
impl<B: BackendForChannel<MC>, MC: MerkleChannel> TreeBuilder<'_, '_, B, MC> {
    pub fn extend_evals(
        &mut self,
        columns: impl IntoIterator<Item = CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> TreeSubspan {
        let span = span!(Level::INFO, "Interpolation for commitment").entered();
        let polys = B::interpolate_columns(columns, self.commitment_scheme.twiddles);
        span.exit();

        self.extend_polys(polys)
    }

    pub fn extend_polys(
        &mut self,
        columns: impl IntoIterator<Item = CirclePoly<B>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys.extend(columns);
        let col_end = self.polys.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }

    pub fn commit(self, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        self.commitment_scheme.commit(self.polys, channel);
    }
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver<B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub polynomials: ColumnVec<Poly<B>>,
    pub commitment: MerkleProver<B, MC::H>,
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeProver<B, MC> {
    pub fn new(
        polynomials: ColumnVec<CirclePoly<B>>,
        log_blowup_factor: u32,
        channel: &mut MC::C,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
    ) -> Self {
        let span = span!(Level::INFO, "Extension").entered();
        let polynomials = B::evaluate_polynomials(
            polynomials,
            log_blowup_factor,
            store_polynomials_coefficients,
            twiddles,
        );
        span.exit();

        let _span = span!(Level::INFO, "Merkle").entered();
        let tree = MerkleProver::commit(
            polynomials
                .iter()
                .map(|poly: &Poly<B>| &poly.eval.values)
                .collect(),
        );
        MC::mix_root(channel, tree.root());

        CommitmentTreeProver {
            polynomials,
            commitment: tree,
        }
    }

    /// Decommits the merkle tree on the given query positions.
    /// Returns the values at the queried positions and the decommitment.
    /// The queries are given as a mapping from the log size of the layer size to the queried
    /// positions on each column of that size.
    fn decommit(
        &self,
        queries: &BTreeMap<u32, Vec<usize>>,
    ) -> (Vec<BaseField>, MerkleDecommitment<MC::H>) {
        let eval_vec = self
            .polynomials
            .iter()
            .map(|poly| &poly.eval.values)
            .collect_vec();
        self.commitment.decommit(queries, eval_vec)
    }
}
