use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::iter::zip;
use std::ops::RangeInclusive;

use itertools::{zip_eq, Itertools};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use super::backend::{Col, ColumnOps, CpuBackend};
use super::channel::{Channel, MerkleChannel};
use super::fields::m31::BaseField;
use super::fields::qm31::{SecureField, QM31};
use super::fields::secure_column::{SecureColumnByCoords, SECURE_EXTENSION_DEGREE};
use super::poly::circle::{CircleDomain, PolyOps, SecureEvaluation};
use super::poly::line::{LineEvaluation, LinePoly};
use super::poly::twiddles::TwiddleTree;
use super::poly::BitReversedOrder;
use super::queries::Queries;
use super::ColumnVec;
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CanonicCoset;
use crate::core::poly::line::LineDomain;
use crate::core::utils::bit_reverse_index;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::vcs::prover::{MerkleDecommitment, MerkleProver};
use crate::core::vcs::verifier::{MerkleVerificationError, MerkleVerifier};

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

    const fn last_layer_domain_size(&self) -> usize {
        1 << (self.log_last_layer_degree_bound + self.log_blowup_factor)
    }

    pub const fn security_bits(&self) -> u32 {
        self.log_blowup_factor * self.n_queries as u32
    }
}

pub trait FriOps: ColumnOps<BaseField> + PolyOps + Sized + ColumnOps<SecureField> {
    /// Folds a degree `d` polynomial into a degree `d/2` polynomial.
    ///
    /// Let `eval` be a polynomial evaluated on a [LineDomain] `E`, `alpha` be a random field
    /// element and `pi(x) = 2x^2 - 1` be the circle's x-coordinate doubling map. This function
    /// returns `f' = f0 + alpha * f1` evaluated on `pi(E)` such that `2f(x) = f0(pi(x)) + x *
    /// f1(pi(x))`.
    ///
    /// # Panics
    ///
    /// Panics if there are less than two evaluations.
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self>;

    /// Folds and accumulates a degree `d` circle polynomial into a degree `d/2` univariate
    /// polynomial.
    ///
    /// Let `src` be the evaluation of a circle polynomial `f` on a
    /// [`CircleDomain`] `E`. This function computes evaluations of `f' = f0
    /// + alpha * f1` on the x-coordinates of `E` such that `2f(p) = f0(px) + py * f1(px)`. The
    /// evaluations of `f'` are accumulated into `dst` by the formula `dst = dst * alpha^2 + f'`.
    ///
    /// # Panics
    ///
    /// Panics if `src` is not double the length of `dst`.
    ///
    /// [`CircleDomain`]: super::poly::circle::CircleDomain
    // TODO(andrew): Make folding factor generic.
    // TODO(andrew): Fold directly into FRI layer to prevent allocation.
    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    );

    /// Decomposes a FRI-space polynomial into a polynomial inside the fft-space and the
    /// remainder term.
    /// FRI-space: polynomials of total degree n/2.
    /// Based on lemma #12 from the CircleStark paper: f(P) = g(P)+ lambda * alternating(P),
    /// where lambda is the cosset diff of eval, and g is a polynomial in the fft-space.
    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField);
}

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
pub struct FriProver<'a, B: FriOps + MerkleOps<MC::H>, MC: MerkleChannel> {
    config: FriConfig,
    first_layer: FriFirstLayerProver<'a, B, MC::H>,
    inner_layers: Vec<FriInnerLayerProver<B, MC::H>>,
    last_layer_poly: LinePoly,
}

impl<'a, B: FriOps + MerkleOps<MC::H>, MC: MerkleChannel> FriProver<'a, B, MC> {
    /// Commits to multiple circle polynomials.
    ///
    /// `columns` must be provided in descending order by size with at most one column per size.
    ///
    /// This is a batched commitment that handles multiple mixed-degree polynomials, each
    /// evaluated over domains of varying sizes. Instead of combining these evaluations into
    /// a single polynomial on a unified domain for commitment, this function commits to each
    /// polynomial on its respective domain. The evaluations are then efficiently merged in the
    /// FRI layer corresponding to the size of a polynomial's domain.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `columns` is empty or not sorted in descending order by domain size.
    /// * An evaluation is not from a sufficiently low degree circle polynomial.
    /// * An evaluation's domain is smaller than the last layer.
    /// * An evaluation's domain is not a canonic circle domain.
    #[instrument(skip_all)]
    pub fn commit(
        channel: &mut MC::C,
        config: FriConfig,
        columns: &'a [SecureEvaluation<B, BitReversedOrder>],
        twiddles: &TwiddleTree<B>,
    ) -> Self {
        assert!(!columns.is_empty(), "no columns");
        assert!(columns.iter().all(|e| e.domain.is_canonic()), "not canonic");
        assert!(
            columns.array_windows().all(|[a, b]| a.len() > b.len()),
            "column sizes not decreasing"
        );

        let first_layer = Self::commit_first_layer(channel, columns);
        let (inner_layers, last_layer_evaluation) =
            Self::commit_inner_layers(channel, config, columns, twiddles);
        let last_layer_poly = Self::commit_last_layer(channel, config, last_layer_evaluation);

        Self {
            config,
            first_layer,
            inner_layers,
            last_layer_poly,
        }
    }

    /// Commits to the first FRI layer.
    ///
    /// The first layer commits to all input circle polynomial columns (possibly of mixed degree)
    /// involved in FRI.
    ///
    /// All `columns` must be provided in descending order by size.
    fn commit_first_layer(
        channel: &mut MC::C,
        columns: &'a [SecureEvaluation<B, BitReversedOrder>],
    ) -> FriFirstLayerProver<'a, B, MC::H> {
        let layer = FriFirstLayerProver::new(columns);
        MC::mix_root(channel, layer.merkle_tree.root());
        layer
    }

    /// Builds and commits to the inner FRI layers (all layers except the first and last).
    ///
    /// All `columns` must be provided in descending order by size. Note there is at most one column
    /// of each size.
    ///
    /// Returns all inner layers and the evaluation of the last layer.
    fn commit_inner_layers(
        channel: &mut MC::C,
        config: FriConfig,
        columns: &[SecureEvaluation<B, BitReversedOrder>],
        twiddles: &TwiddleTree<B>,
    ) -> (Vec<FriInnerLayerProver<B, MC::H>>, LineEvaluation<B>) {
        /// Returns the size of the line evaluation a circle evaluation gets folded into.
        fn folded_size(v: &SecureEvaluation<impl PolyOps, BitReversedOrder>) -> usize {
            v.len() >> CIRCLE_TO_LINE_FOLD_STEP
        }

        let first_inner_layer_log_size = folded_size(&columns[0]).ilog2();
        let first_inner_layer_domain =
            LineDomain::new(Coset::half_odds(first_inner_layer_log_size));

        let mut layer_evaluation = LineEvaluation::new_zero(first_inner_layer_domain);
        let mut columns = columns.iter().peekable();
        let mut layers = Vec::new();
        let folding_alpha = channel.draw_felt();

        // Folding the max size column.
        B::fold_circle_into_line(
            &mut layer_evaluation,
            columns.next().unwrap(),
            folding_alpha,
            twiddles,
        );

        while layer_evaluation.len() > config.last_layer_domain_size() {
            let layer = FriInnerLayerProver::new(layer_evaluation);
            MC::mix_root(channel, layer.merkle_tree.root());
            let folding_alpha = channel.draw_felt();
            layer_evaluation = B::fold_line(&layer.evaluation, folding_alpha, twiddles);

            // Check for circle polys in the first layer that should be combined in this layer.
            if let Some(column) = columns.next_if(|c| folded_size(c) == layer_evaluation.len()) {
                B::fold_circle_into_line(&mut layer_evaluation, column, folding_alpha, twiddles);
            }
            layers.push(layer);
        }

        // Check all columns have been consumed.
        assert!(columns.is_empty());

        (layers, layer_evaluation)
    }

    /// Builds and commits to the last layer.
    ///
    /// The layer is committed to by sending the verifier all the coefficients of the remaining
    /// polynomial.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation domain size exceeds the maximum last layer domain size.
    /// * The evaluation is not of sufficiently low degree.
    fn commit_last_layer(
        channel: &mut MC::C,
        config: FriConfig,
        evaluation: LineEvaluation<B>,
    ) -> LinePoly {
        assert_eq!(evaluation.len(), config.last_layer_domain_size());

        let evaluation = evaluation.to_cpu();
        let mut coeffs = evaluation.interpolate().into_ordered_coefficients();

        let last_layer_degree_bound = 1 << config.log_last_layer_degree_bound;
        let zeros = coeffs.split_off(last_layer_degree_bound);
        assert!(zeros.iter().all(SecureField::is_zero), "invalid degree");

        let last_layer_poly = LinePoly::from_ordered_coefficients(coeffs);
        channel.mix_felts(&last_layer_poly);

        last_layer_poly
    }

    /// Returns a FRI proof and the query positions.
    ///
    /// Returned query positions are mapped by column commitment domain log size.
    pub fn decommit(self, channel: &mut MC::C) -> (FriProof<MC::H>, BTreeMap<u32, Vec<usize>>) {
        let max_column_log_size = self.first_layer.max_column_log_size();
        let queries = Queries::generate(channel, max_column_log_size, self.config.n_queries);
        let column_log_sizes = self.first_layer.column_log_sizes();
        let query_positions_by_log_size =
            get_query_positions_by_log_size(&queries, column_log_sizes);
        let proof = self.decommit_on_queries(&queries);
        (proof, query_positions_by_log_size)
    }

    /// # Panics
    ///
    /// Panics if the queries were sampled on the wrong domain size.
    fn decommit_on_queries(self, queries: &Queries) -> FriProof<MC::H> {
        let Self {
            config: _,
            first_layer,
            inner_layers,
            last_layer_poly,
        } = self;

        let first_layer_proof = first_layer.decommit(queries);

        let inner_layer_proofs = inner_layers
            .into_iter()
            .scan(
                queries.fold(CIRCLE_TO_LINE_FOLD_STEP),
                |layer_queries, layer| {
                    let layer_proof = layer.decommit(layer_queries);
                    *layer_queries = layer_queries.fold(FOLD_STEP);
                    Some(layer_proof)
                },
            )
            .collect();

        FriProof {
            first_layer: first_layer_proof,
            inner_layers: inner_layer_proofs,
            last_layer_poly,
        }
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
    /// `column_bounds` should be the committed circle polynomial degree bounds in descending order.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof contains an invalid number of FRI layers.
    /// * The degree of the last layer polynomial is too high.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * There are no degree bounds.
    /// * The degree bounds are not sorted in descending order.
    /// * A degree bound is less than or equal to the last layer's degree bound.
    pub fn commit(
        channel: &mut MC::C,
        config: FriConfig,
        proof: FriProof<MC::H>,
        column_bounds: Vec<CirclePolyDegreeBound>,
    ) -> Result<Self, FriVerificationError> {
        assert!(column_bounds.is_sorted_by_key(|b| Reverse(*b)));

        MC::mix_root(channel, proof.first_layer.commitment);

        let max_column_bound = column_bounds[0];
        let column_commitment_domains = column_bounds
            .iter()
            .map(|bound| {
                let commitment_domain_log_size = bound.log_degree_bound + config.log_blowup_factor;
                CanonicCoset::new(commitment_domain_log_size).circle_domain()
            })
            .collect();

        let first_layer = FriFirstLayerVerifier {
            column_bounds,
            column_commitment_domains,
            proof: proof.first_layer,
            folding_alpha: channel.draw_felt(),
        };

        let mut inner_layers = Vec::new();
        let mut layer_bound = max_column_bound.fold_to_line();
        let mut layer_domain = LineDomain::new(Coset::half_odds(
            layer_bound.log_degree_bound + config.log_blowup_factor,
        ));

        for (layer_index, proof) in proof.inner_layers.into_iter().enumerate() {
            MC::mix_root(channel, proof.commitment);

            inner_layers.push(FriInnerLayerVerifier {
                degree_bound: layer_bound,
                domain: layer_domain,
                folding_alpha: channel.draw_felt(),
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
        first_layer_query_evals: ColumnVec<Vec<SecureField>>,
    ) -> Result<(), FriVerificationError> {
        let queries = self.queries.take().expect("queries not sampled");
        self.decommit_on_queries(&queries, first_layer_query_evals)
    }

    fn decommit_on_queries(
        self,
        queries: &Queries,
        first_layer_query_evals: ColumnVec<Vec<SecureField>>,
    ) -> Result<(), FriVerificationError> {
        let first_layer_sparse_evals =
            self.decommit_first_layer(queries, first_layer_query_evals)?;
        let inner_layer_queries = queries.fold(CIRCLE_TO_LINE_FOLD_STEP);
        let (last_layer_queries, last_layer_query_evals) =
            self.decommit_inner_layers(&inner_layer_queries, first_layer_sparse_evals)?;
        self.decommit_last_layer(last_layer_queries, last_layer_query_evals)
    }

    /// Verifies the first layer decommitment.
    ///
    /// Returns the queries and first layer folded column evaluations needed for
    /// verifying the remaining layers.
    fn decommit_first_layer(
        &self,
        queries: &Queries,
        first_layer_query_evals: ColumnVec<Vec<SecureField>>,
    ) -> Result<ColumnVec<SparseEvaluation>, FriVerificationError> {
        self.first_layer.verify(queries, first_layer_query_evals)
    }

    /// Verifies all inner layer decommitments.
    ///
    /// Returns the queries and query evaluations needed for verifying the last FRI layer.
    fn decommit_inner_layers(
        &self,
        queries: &Queries,
        first_layer_sparse_evals: ColumnVec<SparseEvaluation>,
    ) -> Result<(Queries, Vec<SecureField>), FriVerificationError> {
        let mut layer_queries = queries.clone();
        let mut layer_query_evals = vec![SecureField::zero(); layer_queries.len()];
        let mut first_layer_sparse_evals = first_layer_sparse_evals.into_iter();
        let first_layer_column_bounds = self.first_layer.column_bounds.iter();
        let first_layer_column_domains = self.first_layer.column_commitment_domains.iter();
        let mut first_layer_columns = first_layer_column_bounds
            .zip_eq(first_layer_column_domains)
            .peekable();
        let mut previous_folding_alpha = self.first_layer.folding_alpha;

        for layer in self.inner_layers.iter() {
            // Check for evals committed in the first layer that need to be folded into this layer.
            while let Some((_, column_domain)) =
                first_layer_columns.next_if(|(b, _)| b.fold_to_line() == layer.degree_bound)
            {
                // Use the previous layer's folding alpha to fold the circle's sparse evals into
                // the current layer.
                let folded_column_evals = first_layer_sparse_evals
                    .next()
                    .unwrap()
                    .fold_circle(previous_folding_alpha, *column_domain);

                accumulate_line(
                    &mut layer_query_evals,
                    &folded_column_evals,
                    previous_folding_alpha,
                );
            }

            // Verify the layer and fold it using the current layer's folding alpha.
            (layer_queries, layer_query_evals) =
                layer.verify_and_fold(layer_queries, layer_query_evals)?;
            previous_folding_alpha = layer.folding_alpha;
        }

        // Check all values have been consumed.
        assert!(first_layer_columns.is_empty());
        assert!(first_layer_sparse_evals.is_empty());

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
    pub fn sample_query_positions(&mut self, channel: &mut MC::C) -> BTreeMap<u32, Vec<usize>> {
        let column_log_sizes = self
            .first_layer
            .column_commitment_domains
            .iter()
            .map(|domain| domain.log_size())
            .collect::<BTreeSet<u32>>();
        let max_column_log_size = *column_log_sizes.iter().max().unwrap();
        let queries = Queries::generate(channel, max_column_log_size, self.config.n_queries);
        let query_positions_by_log_size =
            get_query_positions_by_log_size(&queries, column_log_sizes);
        self.queries = Some(queries);
        query_positions_by_log_size
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

/// Returns the column query positions mapped by sample domain log size.
///
/// The column log sizes must be unique and in descending order.
/// Returned column query positions are mapped by their log size.
fn get_query_positions_by_log_size(
    queries: &Queries,
    column_log_sizes: BTreeSet<u32>,
) -> BTreeMap<u32, Vec<usize>> {
    column_log_sizes
        .into_iter()
        .map(|column_log_size| {
            let column_queries = queries.fold(queries.log_domain_size - column_log_size);
            (column_log_size, column_queries.positions)
        })
        .collect()
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
    fn partial_cmp(&self, other: &LinePolyDegreeBound) -> Option<std::cmp::Ordering> {
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
pub struct FriProof<H: MerkleHasher> {
    pub first_layer: FriLayerProof<H>,
    pub inner_layers: Vec<FriLayerProof<H>>,
    pub last_layer_poly: LinePoly,
}

/// Number of folds for univariate polynomials.
// TODO(andrew): Support different step sizes.
pub const FOLD_STEP: u32 = 1;

/// Number of folds when folding a circle polynomial to univariate polynomial.
pub const CIRCLE_TO_LINE_FOLD_STEP: u32 = 1;

/// Proof of an individual FRI layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriLayerProof<H: MerkleHasher> {
    /// Values that the verifier needs but cannot deduce from previous computations, in the
    /// order they are needed. This complements the values that were queried. These must be
    /// supplied directly to the verifier.
    pub fri_witness: Vec<SecureField>,
    pub decommitment: MerkleDecommitment<H>,
    pub commitment: H::Hash,
}

struct FriFirstLayerVerifier<H: MerkleHasher> {
    /// The list of degree bounds of all circle polynomials commited in the first layer.
    column_bounds: Vec<CirclePolyDegreeBound>,
    /// The commitment domain all the circle polynomials in the first layer.
    column_commitment_domains: Vec<CircleDomain>,
    folding_alpha: SecureField,
    proof: FriLayerProof<H>,
}

impl<H: MerkleHasher> FriFirstLayerVerifier<H> {
    /// Verifies the first layer's merkle decommitment, and returns the evaluations needed for
    /// folding the columns to their corresponding layer.
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
        query_evals_by_column: ColumnVec<Vec<SecureField>>,
    ) -> Result<ColumnVec<SparseEvaluation>, FriVerificationError> {
        // Columns are provided in descending order by size.
        let max_column_log_size = self.column_commitment_domains[0].log_size();
        assert_eq!(queries.log_domain_size, max_column_log_size);

        let mut fri_witness = self.proof.fri_witness.iter().copied();
        let mut decommitment_positions_by_log_size = BTreeMap::new();
        let mut sparse_evals_by_column = Vec::new();

        let mut decommitmented_values = vec![];
        for (&column_domain, column_query_evals) in
            zip_eq(&self.column_commitment_domains, query_evals_by_column)
        {
            let column_queries = queries.fold(queries.log_domain_size - column_domain.log_size());

            let (column_decommitment_positions, sparse_evaluation) =
                compute_decommitment_positions_and_rebuild_evals(
                    &column_queries,
                    &column_query_evals,
                    &mut fri_witness,
                    CIRCLE_TO_LINE_FOLD_STEP,
                )
                .map_err(|InsufficientWitnessError| {
                    FriVerificationError::FirstLayerEvaluationsInvalid
                })?;

            // Columns of the same size have the same decommitment positions.
            decommitment_positions_by_log_size
                .insert(column_domain.log_size(), column_decommitment_positions);

            decommitmented_values.extend(
                sparse_evaluation
                    .subset_evals
                    .iter()
                    .flatten()
                    .flat_map(|qm31| qm31.to_m31_array()),
            );
            sparse_evals_by_column.push(sparse_evaluation);
        }

        // Check all proof evals have been consumed.
        if !fri_witness.is_empty() {
            return Err(FriVerificationError::FirstLayerEvaluationsInvalid);
        }

        let merkle_verifier = MerkleVerifier::new(
            self.proof.commitment,
            self.column_commitment_domains
                .iter()
                .flat_map(|column_domain| [column_domain.log_size(); SECURE_EXTENSION_DEGREE])
                .collect(),
        );

        merkle_verifier
            .verify(
                &decommitment_positions_by_log_size,
                decommitmented_values,
                self.proof.decommitment.clone(),
            )
            .map_err(|error| FriVerificationError::FirstLayerCommitmentInvalid { error })?;

        Ok(sparse_evals_by_column)
    }
}

struct FriInnerLayerVerifier<H: MerkleHasher> {
    degree_bound: LinePolyDegreeBound,
    domain: LineDomain,
    folding_alpha: SecureField,
    layer_index: usize,
    proof: FriLayerProof<H>,
}

impl<H: MerkleHasher> FriInnerLayerVerifier<H> {
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
        if !fri_witness.is_empty() {
            return Err(FriVerificationError::InnerLayerEvaluationsInvalid {
                inner_layer: self.layer_index,
            });
        }

        let decommitmented_values = sparse_evaluation
            .subset_evals
            .iter()
            .flatten()
            .flat_map(|qm31| qm31.to_m31_array())
            .collect_vec();

        let merkle_verifier = MerkleVerifier::new(
            self.proof.commitment,
            vec![self.domain.log_size(); SECURE_EXTENSION_DEGREE],
        );

        merkle_verifier
            .verify(
                &BTreeMap::from_iter([(self.domain.log_size(), decommitment_positions)]),
                decommitmented_values,
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

/// Commitment to the first FRI layer.
///
/// The first layer commits to all circle polynomials (possibly of mixed degree) involved in FRI.
struct FriFirstLayerProver<'a, B: FriOps + MerkleOps<H>, H: MerkleHasher> {
    columns: &'a [SecureEvaluation<B, BitReversedOrder>],
    merkle_tree: MerkleProver<B, H>,
}

impl<'a, B: FriOps + MerkleOps<H>, H: MerkleHasher> FriFirstLayerProver<'a, B, H> {
    fn new(columns: &'a [SecureEvaluation<B, BitReversedOrder>]) -> Self {
        let coordinate_columns = extract_coordinate_columns(columns);
        let merkle_tree = MerkleProver::commit(coordinate_columns);

        FriFirstLayerProver {
            columns,
            merkle_tree,
        }
    }

    /// Returns the sizes of all circle polynomial commitment domains.
    fn column_log_sizes(&self) -> BTreeSet<u32> {
        self.columns.iter().map(|e| e.domain.log_size()).collect()
    }

    fn max_column_log_size(&self) -> u32 {
        *self.column_log_sizes().iter().max().unwrap()
    }

    fn decommit(self, queries: &Queries) -> FriLayerProof<H> {
        let max_column_log_size = *self.column_log_sizes().iter().max().unwrap();
        assert_eq!(queries.log_domain_size, max_column_log_size);

        let mut fri_witness = Vec::new();
        let mut decommitment_positions_by_log_size = BTreeMap::new();

        for column in self.columns {
            let column_log_size = column.domain.log_size();
            let column_queries = queries.fold(queries.log_domain_size - column_log_size);

            let (column_decommitment_positions, column_witness) =
                compute_decommitment_positions_and_witness_evals(
                    column,
                    &column_queries.positions,
                    CIRCLE_TO_LINE_FOLD_STEP,
                );

            decommitment_positions_by_log_size
                .insert(column_log_size, column_decommitment_positions);
            fri_witness.extend(column_witness);
        }

        let (_evals, decommitment) = self.merkle_tree.decommit(
            &decommitment_positions_by_log_size,
            extract_coordinate_columns(self.columns),
        );

        let commitment = self.merkle_tree.root();

        FriLayerProof {
            fri_witness,
            decommitment,
            commitment,
        }
    }
}

/// Extracts all base field coordinate columns from each secure column.
fn extract_coordinate_columns<B: PolyOps>(
    columns: &[SecureEvaluation<B, BitReversedOrder>],
) -> Vec<&Col<B, BaseField>> {
    let mut coordinate_columns = Vec::new();

    for secure_column in columns {
        for coordinate_column in secure_column.columns.iter() {
            coordinate_columns.push(coordinate_column);
        }
    }

    coordinate_columns
}

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): Support different step sizes and update docs.
// TODO(andrew): The docs are wrong. Each leaf of the merkle tree commits to a single
// QM31 value. This is inefficient and should be changed.
struct FriInnerLayerProver<B: FriOps + MerkleOps<H>, H: MerkleHasher> {
    evaluation: LineEvaluation<B>,
    merkle_tree: MerkleProver<B, H>,
}

impl<B: FriOps + MerkleOps<H>, H: MerkleHasher> FriInnerLayerProver<B, H> {
    fn new(evaluation: LineEvaluation<B>) -> Self {
        let merkle_tree = MerkleProver::commit(evaluation.values.columns.iter().collect_vec());
        FriInnerLayerProver {
            evaluation,
            merkle_tree,
        }
    }

    fn decommit(self, queries: &Queries) -> FriLayerProof<H> {
        let (decommitment_positions, fri_witness) =
            compute_decommitment_positions_and_witness_evals(
                &self.evaluation.values,
                queries,
                FOLD_STEP,
            );

        let layer_log_size = self.evaluation.domain().log_size();
        let (_evals, decommitment) = self.merkle_tree.decommit(
            &BTreeMap::from_iter([(layer_log_size, decommitment_positions)]),
            self.evaluation.values.columns.iter().collect_vec(),
        );

        let commitment = self.merkle_tree.root();

        FriLayerProof {
            fri_witness,
            decommitment,
            commitment,
        }
    }
}

/// Returns a column's merkle tree decommitment positions and the evals the verifier can't
/// deduce from previous computations but requires for decommitment and folding.
fn compute_decommitment_positions_and_witness_evals(
    column: &SecureColumnByCoords<impl PolyOps>,
    query_positions: &[usize],
    fold_step: u32,
) -> (Vec<usize>, Vec<QM31>) {
    let mut decommitment_positions = Vec::new();
    let mut witness_evals = Vec::new();

    // Group queries by the folding coset they reside in.
    for subset_queries in query_positions.chunk_by(|a, b| a >> fold_step == b >> fold_step) {
        let subset_start = (subset_queries[0] >> fold_step) << fold_step;
        let subset_decommitment_positions = subset_start..subset_start + (1 << fold_step);
        let mut subset_queries_iter = subset_queries.iter().peekable();

        for position in subset_decommitment_positions {
            // Add decommitment position.
            decommitment_positions.push(position);

            // Skip evals the verifier can calculate.
            if subset_queries_iter.next_if_eq(&&position).is_some() {
                continue;
            }

            let eval = column.at(position);
            witness_evals.push(eval);
        }
    }

    (decommitment_positions, witness_evals)
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

/// Foldable subsets of evaluations on a [`CirclePoly`] or [`LinePoly`].
///
/// [`CirclePoly`]: crate::core::poly::circle::CirclePoly
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
                let eval = LineEvaluation::new(fold_domain, eval.into_iter().collect());
                fold_line(&eval, fold_alpha).values.at(0)
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
                let eval = SecureEvaluation::new(fold_domain, eval.into_iter().collect());
                let mut buffer = LineEvaluation::new_zero(LineDomain::new(fold_domain.half_coset));
                fold_circle_into_line(&mut buffer, &eval, fold_alpha);
                buffer.values.at(0)
            })
            .collect()
    }
}

/// Folds a degree `d` polynomial into a degree `d/2` polynomial.
/// See [`FriOps::fold_line`].
pub fn fold_line(
    eval: &LineEvaluation<CpuBackend>,
    alpha: SecureField,
) -> LineEvaluation<CpuBackend> {
    let n = eval.len();
    assert!(n >= 2, "Evaluation too small");

    let domain = eval.domain();

    let folded_values = eval
        .values
        .into_iter()
        .array_chunks()
        .enumerate()
        .map(|(i, [f_x, f_neg_x])| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let x = domain.at(bit_reverse_index(i << FOLD_STEP, domain.log_size()));

            let (mut f0, mut f1) = (f_x, f_neg_x);
            ibutterfly(&mut f0, &mut f1, x.inverse());
            f0 + alpha * f1
        })
        .collect();

    LineEvaluation::new(domain.double(), folded_values)
}

/// Folds and accumulates a degree `d` circle polynomial into a degree `d/2` univariate
/// polynomial.
/// See [`FriOps::fold_circle_into_line`].
pub fn fold_circle_into_line(
    dst: &mut LineEvaluation<CpuBackend>,
    src: &SecureEvaluation<CpuBackend, BitReversedOrder>,
    alpha: SecureField,
) {
    assert_eq!(src.len() >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

    let domain = src.domain;
    let alpha_sq = alpha * alpha;

    src.into_iter()
        .array_chunks()
        .enumerate()
        .for_each(|(i, [f_p, f_neg_p])| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let p = domain.at(bit_reverse_index(
                i << CIRCLE_TO_LINE_FOLD_STEP,
                domain.log_size(),
            ));

            // Calculate `f0(px)` and `f1(px)` such that `2f(p) = f0(px) + py * f1(px)`.
            let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
            ibutterfly(&mut f0_px, &mut f1_px, p.y.inverse());
            let f_prime = alpha * f1_px + f0_px;

            dst.values.set(i, dst.values.at(i) * alpha_sq + f_prime);
        });
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;
    use std::iter::zip;

    use itertools::Itertools;
    use num_traits::{One, Zero};

    use super::FriVerificationError;
    use crate::core::backend::cpu::CpuCirclePoly;
    use crate::core::backend::{ColumnOps, CpuBackend};
    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::Field;
    use crate::core::fri::{
        fold_circle_into_line, fold_line, CirclePolyDegreeBound, FriConfig,
        CIRCLE_TO_LINE_FOLD_STEP,
    };
    use crate::core::poly::circle::{CircleDomain, PolyOps, SecureEvaluation};
    use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};
    use crate::core::poly::BitReversedOrder;
    use crate::core::queries::Queries;
    use crate::core::test_utils::test_channel;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;

    /// Default blowup factor used for tests.
    const LOG_BLOWUP_FACTOR: u32 = 2;

    type FriProver<'a> = super::FriProver<'a, CpuBackend, Blake2sMerkleChannel>;
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
        let drp_domain = domain.double();
        let mut values = domain
            .iter()
            .map(|p| poly.eval_at_point(p.into()))
            .collect();
        CpuBackend::bit_reverse_column(&mut values);
        let evals = LineEvaluation::new(domain, values.into_iter().collect());

        let drp_evals = fold_line(&evals, alpha);
        let mut drp_evals = drp_evals.values.into_iter().collect_vec();
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

        let mut folded_evaluation = LineEvaluation::new_zero(folded_domain);
        fold_circle_into_line(&mut folded_evaluation, &circle_evaluation, alpha);

        assert_eq!(
            log_degree_bound(folded_evaluation),
            LOG_DEGREE - CIRCLE_TO_LINE_FOLD_STEP
        );
    }

    #[test]
    #[should_panic = "invalid degree"]
    fn committing_high_degree_polynomial_fails() {
        const LOG_EXPECTED_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR;
        const LOG_INVALID_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR - 1;
        let config = FriConfig::new(2, LOG_EXPECTED_BLOWUP_FACTOR, 3);
        let column = &[polynomial_evaluation(6, LOG_INVALID_BLOWUP_FACTOR)];
        let twiddles = CpuBackend::precompute_twiddles(column[0].domain.half_coset);

        FriProver::commit(&mut test_channel(), config, column, &twiddles);
    }

    #[test]
    #[should_panic = "not canonic"]
    fn committing_column_from_invalid_domain_fails() {
        let invalid_domain = CircleDomain::new(Coset::new(CirclePointIndex::generator(), 3));
        assert!(!invalid_domain.is_canonic(), "must be an invalid domain");
        let config = FriConfig::new(2, 2, 3);
        let column = SecureEvaluation::new(
            invalid_domain,
            [SecureField::one(); 1 << 4].into_iter().collect(),
        );
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);
        let columns = &[column];

        FriProver::commit(&mut test_channel(), config, columns, &twiddles);
    }

    #[test]
    fn valid_proof_passes_verification() -> Result<(), FriVerificationError> {
        const LOG_DEGREE: u32 = 4;
        let column = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);
        let queries = Queries::from_positions(vec![5], column.domain.log_size());
        let config = FriConfig::new(1, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&column, &queries);
        let columns = &[column];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let proof = prover.decommit_on_queries(&queries);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        verifier.decommit_on_queries(&queries, vec![decommitment_value])
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
        let columns = &[column];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let proof = prover.decommit_on_queries(&queries);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        verifier.decommit_on_queries(&queries, vec![decommitment_value])
    }

    #[test]
    fn valid_mixed_degree_proof_passes_verification() -> Result<(), FriVerificationError> {
        const LOG_DEGREES: [u32; 3] = [6, 5, 4];
        let columns = LOG_DEGREES.map(|log_d| polynomial_evaluation(log_d, LOG_BLOWUP_FACTOR));
        let twiddles = CpuBackend::precompute_twiddles(columns[0].domain.half_coset);
        let log_domain_size = columns[0].domain.log_size();
        let queries = Queries::from_positions(vec![7, 70], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let prover = FriProver::commit(&mut test_channel(), config, &columns, &twiddles);
        let proof = prover.decommit_on_queries(&queries);
        let query_evals = columns.map(|p| query_polynomial(&p, &queries)).to_vec();
        let bounds = LOG_DEGREES.map(CirclePolyDegreeBound::new).to_vec();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bounds).unwrap();

        verifier.decommit_on_queries(&queries, query_evals)
    }

    #[test]
    fn mixed_degree_proof_with_queries_sampled_from_channel_passes_verification(
    ) -> Result<(), FriVerificationError> {
        const LOG_DEGREES: [u32; 3] = [6, 5, 4];
        let columns = LOG_DEGREES.map(|log_d| polynomial_evaluation(log_d, LOG_BLOWUP_FACTOR));
        let twiddles = CpuBackend::precompute_twiddles(columns[0].domain.half_coset);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3);
        let prover = FriProver::commit(&mut test_channel(), config, &columns, &twiddles);
        let (proof, prover_query_positions_by_log_size) = prover.decommit(&mut test_channel());
        let query_evals_by_column = columns.map(|eval| {
            let query_positions = &prover_query_positions_by_log_size[&eval.domain.log_size()];
            query_polynomial_at_positions(&eval, query_positions)
        });
        let bounds = LOG_DEGREES.map(CirclePolyDegreeBound::new).to_vec();

        let mut verifier = FriVerifier::commit(&mut test_channel(), config, proof, bounds).unwrap();
        let verifier_query_positions_by_log_size =
            verifier.sample_query_positions(&mut test_channel());

        assert_eq!(
            prover_query_positions_by_log_size,
            verifier_query_positions_by_log_size
        );
        verifier.decommit(query_evals_by_column.to_vec())
    }

    #[test]
    fn proof_with_removed_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let evaluation = polynomial_evaluation(6, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(evaluation.domain.half_coset);
        let log_domain_size = evaluation.domain.log_size();
        let queries = Queries::from_positions(vec![1], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let columns = &[evaluation];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let proof = prover.decommit_on_queries(&queries);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
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
        let columns = &[evaluation];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let proof = prover.decommit_on_queries(&queries);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
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
        let columns = &[evaluation];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let mut proof = prover.decommit_on_queries(&queries);
        // Remove an evaluation from the second layer's proof.
        proof.inner_layers[1].fri_witness.pop();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, vec![decommitment_value]);

        assert_matches!(
            verification_result,
            Err(FriVerificationError::InnerLayerEvaluationsInvalid { inner_layer: 1 })
        );
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
        let columns = &[evaluation];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let mut proof = prover.decommit_on_queries(&queries);
        // Modify the committed values in the second layer.
        proof.inner_layers[1].fri_witness[0] += BaseField::one();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, vec![decommitment_value]);

        assert_matches!(
            verification_result,
            Err(FriVerificationError::InnerLayerCommitmentInvalid { inner_layer: 1, .. })
        );
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
        let columns = &[evaluation];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let mut proof = prover.decommit_on_queries(&queries);
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
        let columns = &[evaluation];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let mut proof = prover.decommit_on_queries(&queries);
        // Compromise the last layer polynomial's first coefficient.
        proof.last_layer_poly[0] += BaseField::one();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, vec![decommitment_value]);

        assert_matches!(
            verification_result,
            Err(FriVerificationError::LastLayerEvaluationsInvalid)
        );
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
        let columns = &[evaluation];
        let prover = FriProver::commit(&mut test_channel(), config, columns, &twiddles);
        let proof = prover.decommit_on_queries(&queries);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();
        // Simulate the verifier sampling queries on a smaller domain.
        let mut invalid_queries = queries.clone();
        invalid_queries.log_domain_size -= 1;

        let _ = verifier.decommit_on_queries(&invalid_queries, vec![decommitment_value]);
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
