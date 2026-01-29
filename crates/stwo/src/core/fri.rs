use core::fmt::Debug;
use core::iter::zip;
use core::ops::RangeInclusive;

use hashbrown::HashMap;
use itertools::{zip_eq, Itertools};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std_shims::{vec, Vec};
use thiserror::Error;

use super::channel::{Channel, MerkleChannel};
use super::fields::qm31::{SecureField, QM31, SECURE_EXTENSION_DEGREE};
use super::poly::circle::CircleDomain;
use super::queries::{draw_queries, Queries};
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CanonicCoset;
use crate::core::poly::line::{LineDomain, LinePoly};
use crate::core::utils::bit_reverse_index;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{
    MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux, MerkleVerificationError,
    MerkleVerifierLifted,
};

/// FRI proof config
// TODO(andrew): Support different step sizes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FriConfig {
    pub log_blowup_factor: u32,
    pub log_last_layer_degree_bound: u32,
    pub n_queries: usize,
    // TODO(andrew): fold_steps.
}

impl FriConfig {
    const LOG_MIN_LAST_LAYER_DEGREE_BOUND: u32 = 0;
    const LOG_MAX_LAST_LAYER_DEGREE_BOUND: u32 = 10;
    const LOG_LAST_LAYER_DEGREE_BOUND_RANGE: RangeInclusive<u32> =
        Self::LOG_MIN_LAST_LAYER_DEGREE_BOUND..=Self::LOG_MAX_LAST_LAYER_DEGREE_BOUND;

    const LOG_MIN_BLOWUP_FACTOR: u32 = 1;
    const LOG_MAX_BLOWUP_FACTOR: u32 = 16;
    const LOG_BLOWUP_FACTOR_RANGE: RangeInclusive<u32> =
        Self::LOG_MIN_BLOWUP_FACTOR..=Self::LOG_MAX_BLOWUP_FACTOR;

    /// Creates a new FRI configuration.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `log_last_layer_degree_bound` is greater than 10.
    /// * `log_blowup_factor` is equal to zero or greater than 16.
    pub fn new(log_last_layer_degree_bound: u32, log_blowup_factor: u32, n_queries: usize) -> Self {
        assert!(Self::LOG_LAST_LAYER_DEGREE_BOUND_RANGE.contains(&log_last_layer_degree_bound));
        assert!(Self::LOG_BLOWUP_FACTOR_RANGE.contains(&log_blowup_factor));
        Self {
            log_blowup_factor,
            log_last_layer_degree_bound,
            n_queries,
        }
    }

    pub const fn last_layer_domain_size(&self) -> usize {
        1 << (self.log_last_layer_degree_bound + self.log_blowup_factor)
    }

    pub const fn security_bits(&self) -> u32 {
        self.log_blowup_factor * self.n_queries as u32
    }
}

pub struct FriVerifier<MC: MerkleChannel> {
    config: FriConfig,
    // TODO(andrew): The first layer currently commits to all input polynomials. Consider allowing
    // flexibility to only commit to input polynomials on a per-log-size basis. This allows
    // flexibility for cases where committing to the first layer, for a specific log size, isn't
    // necessary. FRI would simply return more query positions for the "uncommitted" log sizes.
    first_layer: FriFirstLayerVerifier<MC::H>,
    inner_layers: Vec<FriInnerLayerVerifier<MC::H>>,
    last_layer_domain: LineDomain,
    last_layer_poly: LinePoly,
    /// The queries used for decommitment. Initialized when calling
    /// [`FriVerifier::sample_query_positions()`].
    queries: Option<Queries>,
}

impl<MC: MerkleChannel> FriVerifier<MC> {
    /// Verifies the commitment stage of FRI.
    ///
    /// `column_bound` should be the committed circle polynomial degree bound.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof contains an invalid number of FRI layers.
    /// * The degree of the last layer polynomial is too high.
    pub fn commit(
        channel: &mut MC::C,
        config: FriConfig,
        proof: FriProof<MC::H>,
        column_bound: CirclePolyDegreeBound,
    ) -> Result<Self, FriVerificationError> {
        MC::mix_root(channel, proof.first_layer.commitment);

        let column_commitment_domain =
            CanonicCoset::new(column_bound.log_degree_bound + config.log_blowup_factor)
                .circle_domain();

        let first_layer = FriFirstLayerVerifier {
            column_commitment_domain,
            proof: proof.first_layer,
            folding_alpha: channel.draw_secure_felt(),
        };

        let mut inner_layers = Vec::new();
        let mut layer_bound = column_bound.fold_to_line();
        let mut layer_domain = LineDomain::new(Coset::half_odds(
            layer_bound.log_degree_bound + config.log_blowup_factor,
        ));

        for (layer_index, proof) in proof.inner_layers.into_iter().enumerate() {
            MC::mix_root(channel, proof.commitment);

            inner_layers.push(FriInnerLayerVerifier {
                domain: layer_domain,
                folding_alpha: channel.draw_secure_felt(),
                layer_index,
                proof,
            });

            layer_bound = layer_bound
                .fold(FOLD_STEP)
                .ok_or(FriVerificationError::InvalidNumFriLayers)?;
            layer_domain = layer_domain.double();
        }

        if layer_bound.log_degree_bound != config.log_last_layer_degree_bound {
            return Err(FriVerificationError::InvalidNumFriLayers);
        }

        let last_layer_domain = layer_domain;
        let last_layer_poly = proof.last_layer_poly;

        if last_layer_poly.len() > (1 << config.log_last_layer_degree_bound) {
            return Err(FriVerificationError::LastLayerDegreeInvalid);
        }

        channel.mix_felts(&last_layer_poly);

        Ok(Self {
            config,
            first_layer,
            inner_layers,
            last_layer_domain,
            last_layer_poly,
            queries: None,
        })
    }

    /// Verifies the decommitment stage of FRI.
    ///
    /// The query evals need to be provided in the same order as their commitment.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The queries were not yet sampled.
    /// * The queries were sampled on the wrong domain size.
    /// * There aren't the same number of decommitted values as degree bounds.
    // TODO(andrew): Finish docs.
    pub fn decommit(
        mut self,
        first_layer_query_evals: Vec<SecureField>,
    ) -> Result<(), FriVerificationError> {
        let queries = self.queries.take().expect("queries not sampled");
        self.decommit_on_queries(&queries, first_layer_query_evals)
    }

    fn decommit_on_queries(
        self,
        queries: &Queries,
        first_layer_query_evals: Vec<SecureField>,
    ) -> Result<(), FriVerificationError> {
        let first_layer_sparse_eval =
            self.decommit_first_layer(queries, first_layer_query_evals)?;
        let inner_layer_queries = queries.fold(CIRCLE_TO_LINE_FOLD_STEP);
        let (last_layer_queries, last_layer_query_evals) =
            self.decommit_inner_layers(&inner_layer_queries, first_layer_sparse_eval)?;
        self.decommit_last_layer(last_layer_queries, last_layer_query_evals)
    }

    /// Verifies the first layer decommitment.
    ///
    /// Returns the queries and first layer folded column evaluations needed for
    /// verifying the remaining layers.
    fn decommit_first_layer(
        &self,
        queries: &Queries,
        first_layer_query_evals: Vec<SecureField>,
    ) -> Result<SparseEvaluation, FriVerificationError> {
        self.first_layer.verify(queries, first_layer_query_evals)
    }

    /// Verifies all inner layer decommitments.
    ///
    /// Returns the queries and query evaluations needed for verifying the last FRI layer.
    fn decommit_inner_layers(
        &self,
        queries: &Queries,
        first_layer_sparse_evals: SparseEvaluation,
    ) -> Result<(Queries, Vec<SecureField>), FriVerificationError> {
        let mut layer_queries = queries.clone();
        let mut layer_query_evals = vec![SecureField::zero(); layer_queries.len()];
        let first_layer_column_domain = self.first_layer.column_commitment_domain;

        // Fold the first layer.
        let folded_column_evals = first_layer_sparse_evals
            .fold_circle(self.first_layer.folding_alpha, first_layer_column_domain);

        accumulate_line(
            &mut layer_query_evals,
            &folded_column_evals,
            self.first_layer.folding_alpha,
        );
        for layer in self.inner_layers.iter() {
            // Verify the layer and fold it using the current layer's folding alpha.
            (layer_queries, layer_query_evals) =
                layer.verify_and_fold(layer_queries, layer_query_evals)?;
        }

        Ok((layer_queries, layer_query_evals))
    }

    /// Verifies the last layer.
    fn decommit_last_layer(
        self,
        queries: Queries,
        query_evals: Vec<SecureField>,
    ) -> Result<(), FriVerificationError> {
        let Self {
            last_layer_domain: domain,
            last_layer_poly,
            ..
        } = self;

        for (&query, query_eval) in zip(&*queries, query_evals) {
            let x = domain.at(bit_reverse_index(query, domain.log_size()));

            if query_eval != last_layer_poly.eval_at_point(x.into()) {
                return Err(FriVerificationError::LastLayerEvaluationsInvalid);
            }
        }

        Ok(())
    }

    /// Samples and returns query positions mapped by column log size.
    pub fn sample_query_positions(&mut self, channel: &mut MC::C) -> Vec<usize> {
        let first_layer_log_size = self.first_layer.column_commitment_domain.log_size();
        let unsorted_query_locations =
            draw_queries(channel, first_layer_log_size, self.config.n_queries);
        let queries = Queries::new(&unsorted_query_locations, first_layer_log_size);
        self.queries = Some(queries.clone());
        queries.positions
    }
}

fn accumulate_line(
    layer_query_evals: &mut [SecureField],
    column_query_evals: &[SecureField],
    folding_alpha: SecureField,
) {
    let folding_alpha_squared = folding_alpha.square();
    for (curr_layer_eval, folded_column_eval) in zip_eq(layer_query_evals, column_query_evals) {
        *curr_layer_eval *= folding_alpha_squared;
        *curr_layer_eval += *folded_column_eval;
    }
}

#[derive(Clone, Copy, Debug, Error)]
pub enum FriVerificationError {
    #[error("proof contains an invalid number of FRI layers")]
    InvalidNumFriLayers,
    #[error("evaluations are invalid in the first layer")]
    FirstLayerEvaluationsInvalid,
    #[error("queries do not resolve to their commitment in the first layer")]
    FirstLayerCommitmentInvalid { error: MerkleVerificationError },
    #[error("queries do not resolve to their commitment in inner layer {inner_layer}")]
    InnerLayerCommitmentInvalid {
        inner_layer: usize,
        error: MerkleVerificationError,
    },
    #[error("evaluations are invalid in inner layer {inner_layer}")]
    InnerLayerEvaluationsInvalid { inner_layer: usize },
    #[error("degree of last layer is invalid")]
    LastLayerDegreeInvalid,
    #[error("evaluations in the last layer are invalid")]
    LastLayerEvaluationsInvalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CirclePolyDegreeBound {
    log_degree_bound: u32,
}

impl CirclePolyDegreeBound {
    pub const fn new(log_degree_bound: u32) -> Self {
        Self { log_degree_bound }
    }

    /// Maps a circle polynomial's degree bound to the degree bound of the univariate (line)
    /// polynomial it gets folded into.
    const fn fold_to_line(&self) -> LinePolyDegreeBound {
        LinePolyDegreeBound {
            log_degree_bound: self.log_degree_bound - CIRCLE_TO_LINE_FOLD_STEP,
        }
    }
}

impl PartialOrd<LinePolyDegreeBound> for CirclePolyDegreeBound {
    fn partial_cmp(&self, other: &LinePolyDegreeBound) -> Option<core::cmp::Ordering> {
        Some(self.log_degree_bound.cmp(&other.log_degree_bound))
    }
}

impl PartialEq<LinePolyDegreeBound> for CirclePolyDegreeBound {
    fn eq(&self, other: &LinePolyDegreeBound) -> bool {
        self.log_degree_bound == other.log_degree_bound
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LinePolyDegreeBound {
    log_degree_bound: u32,
}

impl LinePolyDegreeBound {
    /// Returns [None] if the unfolded degree bound is smaller than the folding factor.
    const fn fold(self, n_folds: u32) -> Option<Self> {
        if self.log_degree_bound < n_folds {
            return None;
        }

        let log_degree_bound = self.log_degree_bound - n_folds;
        Some(Self { log_degree_bound })
    }
}

/// A FRI proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriProof<H: MerkleHasherLifted> {
    pub first_layer: FriLayerProof<H>,
    pub inner_layers: Vec<FriLayerProof<H>>,
    pub last_layer_poly: LinePoly,
}

/// Auxiliary data produced by the prover.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriProofAux<H: MerkleHasherLifted> {
    pub first_layer: FriLayerProofAux<H>,
    pub inner_layers: Vec<FriLayerProofAux<H>>,
}

#[derive(Clone, Debug)]
pub struct ExtendedFriProof<H: MerkleHasherLifted> {
    pub proof: FriProof<H>,
    pub aux: FriProofAux<H>,
}

/// Number of folds for univariate polynomials.
// TODO(andrew): Support different step sizes.
pub const FOLD_STEP: u32 = 1;

/// Number of folds when folding a circle polynomial to univariate polynomial.
pub const CIRCLE_TO_LINE_FOLD_STEP: u32 = 1;

/// Proof of an individual FRI layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriLayerProof<H: MerkleHasherLifted> {
    /// Values that the verifier needs but cannot deduce from previous computations, in the
    /// order they are needed. This complements the values that were queried. These must be
    /// supplied directly to the verifier.
    pub fri_witness: Vec<SecureField>,
    pub decommitment: MerkleDecommitmentLifted<H>,
    pub commitment: H::Hash,
}

/// Auxiliary data for a single FRI layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriLayerProofAux<H: MerkleHasherLifted> {
    /// For each column (of different size), the values of all nodes that participate in the
    /// decommitment.
    // TODO(lior): Remove the `Vec<>` once mixed-degree Merkle is removed.
    pub all_values: Vec<HashMap<usize, QM31>>,
    /// The auxiliary data for the merkle decommitment.
    pub decommitment: MerkleDecommitmentLiftedAux<H>,
}

#[derive(Clone, Debug)]
pub struct ExtendedFriLayerProof<H: MerkleHasherLifted> {
    pub proof: FriLayerProof<H>,
    pub aux: FriLayerProofAux<H>,
}

struct FriFirstLayerVerifier<H: MerkleHasherLifted> {
    /// The commitment domain all the circle polynomials in the first layer.
    column_commitment_domain: CircleDomain,
    folding_alpha: SecureField,
    proof: FriLayerProof<H>,
}

impl<H: MerkleHasherLifted> FriFirstLayerVerifier<H> {
    /// Verifies the first layer's merkle decommitment, and returns the evaluations needed to fold
    /// the committed column.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof doesn't store enough evaluations.
    /// * The merkle decommitment is invalid.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The queries are sampled on the wrong domain.
    /// * There are an invalid number of provided column evals.
    fn verify(
        &self,
        queries: &Queries,
        column_query_evals: Vec<SecureField>,
    ) -> Result<SparseEvaluation, FriVerificationError> {
        let column_log_size = self.column_commitment_domain.log_size();
        assert_eq!(queries.log_domain_size, column_log_size);

        let mut fri_witness = self.proof.fri_witness.iter().copied();

        let (decommitment_positions, sparse_evaluation) =
            compute_decommitment_positions_and_rebuild_evals(
                queries,
                &column_query_evals,
                &mut fri_witness,
                CIRCLE_TO_LINE_FOLD_STEP,
            )
            .map_err(|InsufficientWitnessError| {
                FriVerificationError::FirstLayerEvaluationsInvalid
            })?;

        // A QM31 column is committed as 4 M31 columns.
        let mut decommitmented_values: [Vec<BaseField>; SECURE_EXTENSION_DEGREE] =
            core::array::from_fn(|_| Vec::new());
        sparse_evaluation
            .subset_evals
            .iter()
            .flatten()
            .for_each(|x| {
                let arr = x.to_m31_array();
                for (i, val) in arr.into_iter().enumerate() {
                    decommitmented_values[i].push(val);
                }
            });

        // Check all proof evals have been consumed.
        if fri_witness.next().is_some() {
            return Err(FriVerificationError::FirstLayerEvaluationsInvalid);
        }

        let merkle_verifier = MerkleVerifierLifted::new(
            self.proof.commitment,
            vec![self.column_commitment_domain.log_size(); SECURE_EXTENSION_DEGREE],
        );

        merkle_verifier
            .verify(
                &decommitment_positions,
                decommitmented_values.to_vec(),
                self.proof.decommitment.clone(),
            )
            .map_err(|error| FriVerificationError::FirstLayerCommitmentInvalid { error })?;

        Ok(sparse_evaluation)
    }
}

struct FriInnerLayerVerifier<H: MerkleHasherLifted> {
    domain: LineDomain,
    folding_alpha: SecureField,
    layer_index: usize,
    proof: FriLayerProof<H>,
}

impl<H: MerkleHasherLifted> FriInnerLayerVerifier<H> {
    /// Verifies the layer's merkle decommitment and returns the the folded queries and query evals.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof doesn't store the correct number of evaluations.
    /// * The merkle decommitment is invalid.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The number of queries doesn't match the number of evals.
    /// * The queries are sampled on the wrong domain.
    fn verify_and_fold(
        &self,
        queries: Queries,
        evals_at_queries: Vec<SecureField>,
    ) -> Result<(Queries, Vec<SecureField>), FriVerificationError> {
        assert_eq!(queries.log_domain_size, self.domain.log_size());

        let mut fri_witness = self.proof.fri_witness.iter().copied();

        let (decommitment_positions, sparse_evaluation) =
            compute_decommitment_positions_and_rebuild_evals(
                &queries,
                &evals_at_queries,
                &mut fri_witness,
                FOLD_STEP,
            )
            .map_err(|InsufficientWitnessError| {
                FriVerificationError::InnerLayerEvaluationsInvalid {
                    inner_layer: self.layer_index,
                }
            })?;

        // Check all proof evals have been consumed.
        if fri_witness.next().is_some() {
            return Err(FriVerificationError::InnerLayerEvaluationsInvalid {
                inner_layer: self.layer_index,
            });
        }

        // A QM31 column is committed as 4 M31 columns.
        let mut decommitmented_values: [Vec<BaseField>; SECURE_EXTENSION_DEGREE] =
            core::array::from_fn(|_| Vec::new());
        sparse_evaluation
            .subset_evals
            .iter()
            .flatten()
            .for_each(|x| {
                let arr = x.to_m31_array();
                for (i, val) in arr.into_iter().enumerate() {
                    decommitmented_values[i].push(val);
                }
            });

        let merkle_verifier = MerkleVerifierLifted::new(
            self.proof.commitment,
            vec![self.domain.log_size(); SECURE_EXTENSION_DEGREE],
        );

        merkle_verifier
            .verify(
                &decommitment_positions,
                decommitmented_values.to_vec(),
                self.proof.decommitment.clone(),
            )
            .map_err(|e| FriVerificationError::InnerLayerCommitmentInvalid {
                inner_layer: self.layer_index,
                error: e,
            })?;

        let folded_queries = queries.fold(FOLD_STEP);
        let folded_evals = sparse_evaluation.fold_line(self.folding_alpha, self.domain);

        Ok((folded_queries, folded_evals))
    }
}

/// Returns a column's merkle tree decommitment positions and re-builds the evaluations needed by
/// the verifier for folding and decommitment.
///
/// # Panics
///
/// Panics if the number of queries doesn't match the number of query evals.
fn compute_decommitment_positions_and_rebuild_evals(
    queries: &Queries,
    query_evals: &[QM31],
    mut witness_evals: impl Iterator<Item = QM31>,
    fold_step: u32,
) -> Result<(Vec<usize>, SparseEvaluation), InsufficientWitnessError> {
    let mut query_evals = query_evals.iter().copied();

    let mut decommitment_positions = Vec::new();
    let mut subset_evals = Vec::new();
    let mut subset_domain_index_initials = Vec::new();

    // Group queries by the subset they reside in.
    for subset_queries in queries.chunk_by(|a, b| a >> fold_step == b >> fold_step) {
        let subset_start = (subset_queries[0] >> fold_step) << fold_step;
        let subset_decommitment_positions = subset_start..subset_start + (1 << fold_step);
        decommitment_positions.extend(subset_decommitment_positions.clone());

        let mut subset_queries_iter = subset_queries.iter().copied().peekable();

        let subset_eval = subset_decommitment_positions
            .map(|position| match subset_queries_iter.next_if_eq(&position) {
                Some(_) => Ok(query_evals.next().unwrap()),
                None => witness_evals.next().ok_or(InsufficientWitnessError),
            })
            .collect::<Result<_, _>>()?;

        subset_evals.push(subset_eval);
        subset_domain_index_initials.push(bit_reverse_index(subset_start, queries.log_domain_size));
    }

    let sparse_evaluation = SparseEvaluation::new(subset_evals, subset_domain_index_initials);

    Ok((decommitment_positions, sparse_evaluation))
}

#[derive(Debug)]
struct InsufficientWitnessError;

/// Foldable subsets of evaluations on a [`CircleCoefficients`] or [`LinePoly`].
///
/// [`CircleCoefficients`]: crate::core::poly::circle::CircleCoefficients
struct SparseEvaluation {
    // TODO(andrew): Perhaps subset isn't the right word. Coset, Subgroup?
    subset_evals: Vec<Vec<SecureField>>,
    subset_domain_initial_indexes: Vec<usize>,
}

impl SparseEvaluation {
    /// # Panics
    ///
    /// Panics if a subset size doesn't equal `2^FOLD_STEP` or there aren't the same number of
    /// domain indexes as subsets.
    fn new(subset_evals: Vec<Vec<SecureField>>, subset_domain_initial_indexes: Vec<usize>) -> Self {
        let fold_factor = 1 << FOLD_STEP;
        assert!(subset_evals.iter().all(|e| e.len() == fold_factor));
        assert_eq!(subset_evals.len(), subset_domain_initial_indexes.len());
        Self {
            subset_evals,
            subset_domain_initial_indexes,
        }
    }

    fn fold_line(self, fold_alpha: SecureField, source_domain: LineDomain) -> Vec<SecureField> {
        zip(self.subset_evals, self.subset_domain_initial_indexes)
            .map(|(eval, domain_initial_index)| {
                let fold_domain_initial = source_domain.coset().index_at(domain_initial_index);
                let fold_domain = LineDomain::new(Coset::new(fold_domain_initial, FOLD_STEP));
                let (_, folded_values) = fold_line(&eval, fold_domain, fold_alpha);
                folded_values[0]
            })
            .collect()
    }

    fn fold_circle(self, fold_alpha: SecureField, source_domain: CircleDomain) -> Vec<SecureField> {
        zip(self.subset_evals, self.subset_domain_initial_indexes)
            .map(|(eval, domain_initial_index)| {
                let fold_domain_initial = source_domain.index_at(domain_initial_index);
                let fold_domain = CircleDomain::new(Coset::new(
                    fold_domain_initial,
                    CIRCLE_TO_LINE_FOLD_STEP - 1,
                ));
                let eval = eval.into_iter().collect_vec();
                let mut buffer = vec![SecureField::zero(); fold_domain.half_coset.size()];
                fold_circle_into_line(&mut buffer, &eval, fold_domain, fold_alpha);
                buffer[0]
            })
            .collect()
    }
}

/// Folds a degree `d` polynomial into a degree `d/2` polynomial.
/// See [`crate::prover::fri::FriOps::fold_line`].
pub fn fold_line(
    eval: &[SecureField],
    domain: LineDomain,
    alpha: SecureField,
) -> (LineDomain, Vec<SecureField>) {
    let n = eval.len();
    assert!(n >= 2, "Evaluation too small");

    let folded_values = eval
        .iter()
        .tuples()
        .enumerate()
        .map(|(i, (&f_x, &f_neg_x))| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let x = domain.at(bit_reverse_index(i << FOLD_STEP, domain.log_size()));

            let (mut f0, mut f1) = (f_x, f_neg_x);
            ibutterfly(&mut f0, &mut f1, x.inverse());
            f0 + alpha * f1
        })
        .collect();

    (domain.double(), folded_values)
}

/// Folds and accumulates a degree `d` circle polynomial into a degree `d/2` univariate
/// polynomial.
/// See [`crate::prover::fri::FriOps::fold_circle_into_line`].
pub fn fold_circle_into_line(
    dst: &mut [SecureField],
    src: &[SecureField],
    src_domain: CircleDomain,
    alpha: SecureField,
) {
    assert_eq!(src.len() >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

    let alpha_sq = alpha * alpha;

    src.iter()
        .tuples()
        .enumerate()
        .for_each(|(i, (&f_p, &f_neg_p))| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let p = src_domain.at(bit_reverse_index(
                i << CIRCLE_TO_LINE_FOLD_STEP,
                src_domain.log_size(),
            ));

            // Calculate `f0(px)` and `f1(px)` such that `2f(p) = f0(px) + py * f1(px)`.
            let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
            ibutterfly(&mut f0_px, &mut f1_px, p.y.inverse());
            let f_prime = alpha * f1_px + f0_px;

            dst[i] = dst[i] * alpha_sq + f_prime;
        });
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;
    use num_traits::{One, Zero};

    use super::FriVerificationError;
    use crate::core::circle::Coset;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::Field;
    use crate::core::fri::{
        fold_circle_into_line, fold_line, CirclePolyDegreeBound, FriConfig,
        CIRCLE_TO_LINE_FOLD_STEP,
    };
    use crate::core::poly::circle::CircleDomain;
    use crate::core::poly::line::{LineDomain, LinePoly};
    use crate::core::queries::Queries;
    use crate::core::test_utils::test_channel;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::{ColumnOps, CpuBackend};
    use crate::prover::line::LineEvaluation;
    use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
    use crate::prover::poly::BitReversedOrder;

    /// Default blowup factor used for tests.
    const LOG_BLOWUP_FACTOR: u32 = 2;

    type FriProver<'a> = crate::prover::fri::FriProver<'a, CpuBackend, Blake2sMerkleChannel>;
    type FriVerifier = super::FriVerifier<Blake2sMerkleChannel>;

    #[test]
    fn fold_line_works() {
        const DEGREE: usize = 8;
        // Coefficients are bit-reversed.
        let even_coeffs: [SecureField; DEGREE / 2] = [1, 2, 1, 3].map(SecureField::from);
        let odd_coeffs: [SecureField; DEGREE / 2] = [3, 5, 4, 1].map(SecureField::from);
        let poly = LinePoly::new([even_coeffs, odd_coeffs].concat());
        let even_poly = LinePoly::new(even_coeffs.to_vec());
        let odd_poly = LinePoly::new(odd_coeffs.to_vec());
        let alpha = BaseField::from_u32_unchecked(19283).into();
        let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2()));
        let mut values = domain
            .iter()
            .map(|p| poly.eval_at_point(p.into()))
            .collect();
        CpuBackend::bit_reverse_column(&mut values);

        let (drp_domain, drp_evals) = fold_line(&values, domain, alpha);
        let mut drp_evals = drp_evals.into_iter().collect_vec();
        CpuBackend::bit_reverse_column(&mut drp_evals);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        for (i, (&drp_eval, x)) in zip(&drp_evals, drp_domain).enumerate() {
            let f_e: SecureField = even_poly.eval_at_point(x.into());
            let f_o: SecureField = odd_poly.eval_at_point(x.into());
            assert_eq!(drp_eval, (f_e + alpha * f_o).double(), "mismatch at {i}");
        }
    }

    #[test]
    fn fold_circle_to_line_works() {
        const LOG_DEGREE: u32 = 4;
        let circle_evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let alpha = SecureField::one();
        let folded_domain = LineDomain::new(circle_evaluation.domain.half_coset);

        let mut folded_evaluation = vec![SecureField::zero(); folded_domain.size()];
        fold_circle_into_line(
            &mut folded_evaluation,
            &circle_evaluation.values.into_iter().collect_vec(),
            circle_evaluation.domain,
            alpha,
        );
        let folded_evaluation =
            LineEvaluation::new(folded_domain, folded_evaluation.into_iter().collect());

        assert_eq!(
            log_degree_bound(folded_evaluation),
            LOG_DEGREE - CIRCLE_TO_LINE_FOLD_STEP
        );
    }

    #[test]
    fn valid_proof_passes_verification() -> Result<(), FriVerificationError> {
        const LOG_DEGREE: u32 = 4;
        let column = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);
        let queries = Queries::from_positions(vec![5], column.domain.log_size());
        let config = FriConfig::new(1, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&column, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &column, &twiddles);
        let proof = prover.decommit_on_queries(&queries).proof;
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        verifier.decommit_on_queries(&queries, decommitment_value)
    }

    #[test]
    fn valid_proof_with_constant_last_layer_passes_verification() -> Result<(), FriVerificationError>
    {
        const LOG_DEGREE: u32 = 3;
        const LAST_LAYER_LOG_BOUND: u32 = 0;
        let column = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);
        let queries = Queries::from_positions(vec![5], column.domain.log_size());
        let config = FriConfig::new(LAST_LAYER_LOG_BOUND, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&column, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &column, &twiddles);
        let proof = prover.decommit_on_queries(&queries).proof;
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        verifier.decommit_on_queries(&queries, decommitment_value)
    }

    #[test]
    fn proof_with_removed_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let evaluation = polynomial_evaluation(6, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(evaluation.domain.half_coset);
        let log_domain_size = evaluation.domain.log_size();
        let queries = Queries::from_positions(vec![1], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let prover = FriProver::commit(&mut test_channel(), config, &evaluation, &twiddles);
        let proof = prover.decommit_on_queries(&queries).proof;
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        // Set verifier's config to expect one extra layer than prover config.
        let mut invalid_config = config;
        invalid_config.log_last_layer_degree_bound -= 1;

        let verifier = FriVerifier::commit(&mut test_channel(), invalid_config, proof, bound);

        assert!(matches!(
            verifier,
            Err(FriVerificationError::InvalidNumFriLayers)
        ));
    }

    #[test]
    fn proof_with_added_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(evaluation.domain.half_coset);
        let log_domain_size = evaluation.domain.log_size();
        let queries = Queries::from_positions(vec![1], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let prover = FriProver::commit(&mut test_channel(), config, &evaluation, &twiddles);
        let proof = prover.decommit_on_queries(&queries).proof;
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        // Set verifier's config to expect one less layer than prover config.
        let mut invalid_config = config;
        invalid_config.log_last_layer_degree_bound += 1;

        let verifier = FriVerifier::commit(&mut test_channel(), invalid_config, proof, bound);

        assert!(matches!(
            verifier,
            Err(FriVerificationError::InvalidNumFriLayers)
        ));
    }

    #[test]
    fn proof_with_invalid_inner_layer_evaluation_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(evaluation.domain.half_coset);
        let log_domain_size = evaluation.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&evaluation, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &evaluation, &twiddles);
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let mut proof = prover.decommit_on_queries(&queries).proof;
        // Remove an evaluation from the second layer's proof.
        proof.inner_layers[1].fri_witness.pop();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, decommitment_value);

        assert!(matches!(
            verification_result,
            Err(FriVerificationError::InnerLayerEvaluationsInvalid { inner_layer: 1 })
        ));
    }

    #[test]
    fn proof_with_invalid_inner_layer_decommitment_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(evaluation.domain.half_coset);
        let log_domain_size = evaluation.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&evaluation, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &evaluation, &twiddles);
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let mut proof = prover.decommit_on_queries(&queries).proof;
        // Modify the committed values in the second layer.
        proof.inner_layers[1].fri_witness[0] += BaseField::one();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, decommitment_value);

        assert!(matches!(
            verification_result,
            Err(FriVerificationError::InnerLayerCommitmentInvalid { inner_layer: 1, .. })
        ));
    }

    #[test]
    fn proof_with_invalid_last_layer_degree_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        const LOG_MAX_LAST_LAYER_DEGREE: u32 = 2;
        let evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(evaluation.domain.half_coset);
        let log_domain_size = evaluation.domain.log_size();
        let queries = Queries::from_positions(vec![1, 7, 8], log_domain_size);
        let config = FriConfig::new(LOG_MAX_LAST_LAYER_DEGREE, LOG_BLOWUP_FACTOR, queries.len());
        let prover = FriProver::commit(&mut test_channel(), config, &evaluation, &twiddles);
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let mut proof = prover.decommit_on_queries(&queries).proof;
        let bad_last_layer_coeffs = vec![One::one(); 1 << (LOG_MAX_LAST_LAYER_DEGREE + 1)];
        proof.last_layer_poly = LinePoly::new(bad_last_layer_coeffs);

        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound);

        assert!(matches!(
            verifier,
            Err(FriVerificationError::LastLayerDegreeInvalid)
        ));
    }

    #[test]
    fn proof_with_invalid_last_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(evaluation.domain.half_coset);
        let log_domain_size = evaluation.domain.log_size();
        let queries = Queries::from_positions(vec![1, 7, 8], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&evaluation, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &evaluation, &twiddles);
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let mut proof = prover.decommit_on_queries(&queries).proof;
        // Compromise the last layer polynomial's first coefficient.
        proof.last_layer_poly[0] += BaseField::one();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, decommitment_value);

        assert!(matches!(
            verification_result,
            Err(FriVerificationError::LastLayerEvaluationsInvalid)
        ));
    }

    #[test]
    #[should_panic]
    fn decommit_queries_on_invalid_domain_fails_verification() {
        const LOG_DEGREE: u32 = 3;
        let evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(evaluation.domain.half_coset);
        let log_domain_size = evaluation.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let config = FriConfig::new(1, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&evaluation, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &evaluation, &twiddles);
        let proof = prover.decommit_on_queries(&queries).proof;
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();
        // Simulate the verifier sampling queries on a smaller domain.
        let mut invalid_queries = queries.clone();
        invalid_queries.log_domain_size -= 1;

        let _ = verifier.decommit_on_queries(&invalid_queries, decommitment_value);
    }

    /// Returns an evaluation of a random polynomial with degree `2^log_degree`.
    ///
    /// The evaluation domain size is `2^(log_degree + log_blowup_factor)`.
    fn polynomial_evaluation(
        log_degree: u32,
        log_blowup_factor: u32,
    ) -> SecureEvaluation<CpuBackend, BitReversedOrder> {
        let poly = CpuCirclePoly::new(vec![BaseField::one(); 1 << log_degree]);
        let coset = Coset::half_odds(log_degree + log_blowup_factor - 1);
        let domain = CircleDomain::new(coset);
        let values = poly.evaluate(domain);
        SecureEvaluation::new(domain, values.into_iter().map(SecureField::from).collect())
    }

    /// Returns the log degree bound of a polynomial.
    fn log_degree_bound(polynomial: LineEvaluation<CpuBackend>) -> u32 {
        let coeffs = polynomial.interpolate().into_ordered_coefficients();
        let degree = coeffs.into_iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        (degree + 1).ilog2()
    }

    fn query_polynomial(
        polynomial: &SecureEvaluation<CpuBackend, BitReversedOrder>,
        queries: &Queries,
    ) -> Vec<SecureField> {
        let queries = queries.fold(queries.log_domain_size - polynomial.domain.log_size());
        query_polynomial_at_positions(polynomial, &queries.positions)
    }

    fn query_polynomial_at_positions(
        polynomial: &SecureEvaluation<CpuBackend, BitReversedOrder>,
        query_positions: &[usize],
    ) -> Vec<SecureField> {
        query_positions.iter().map(|p| polynomial.at(*p)).collect()
    }
}
