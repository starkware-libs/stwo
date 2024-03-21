use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::iter::zip;
use std::ops::RangeInclusive;

use itertools::Itertools;
use num_traits::Zero;
use thiserror::Error;

use super::backend::cpu::CPULineEvaluation;
use super::backend::CPUBackend;
use super::channel::Channel;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::{ExtensionOf, Field, FieldOps};
use super::poly::circle::CircleEvaluation;
use super::poly::line::{LineEvaluation, LinePoly};
use super::poly::BitReversedOrder;
// TODO(andrew): Create fri/ directory, move queries.rs there and split this file up.
use super::queries::{Queries, SparseSubCircleDomain};
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::backend::Column;
use crate::core::circle::Coset;
use crate::core::poly::line::LineDomain;
use crate::core::utils::bit_reverse_index;

/// FRI proof config
// TODO(andrew): Support different step sizes.
#[derive(Debug, Clone, Copy)]
pub struct FriConfig {
    log_blowup_factor: u32,
    log_last_layer_degree_bound: u32,
    n_queries: usize,
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

    fn last_layer_domain_size(&self) -> usize {
        1 << (self.log_last_layer_degree_bound + self.log_blowup_factor)
    }
}

pub trait FriOps: FieldOps<SecureField> + Sized {
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
        eval: &LineEvaluation<Self, SecureField, BitReversedOrder>,
        alpha: SecureField,
    ) -> LineEvaluation<Self, SecureField, BitReversedOrder>;

    /// Folds and accumulates a degree `d` circle polynomial into a degree `d/2` univariate
    /// polynomial.
    ///
    /// Let `src` be the evaluation of a circle polynomial `f` on a [CircleDomain] `E`. This
    /// function computes evaluations of `f' = f0 + alpha * f1` on the x-coordinates of `E` such
    /// that `2f(p) = f0(px) + py * f1(px)`. The evaluations of `f'` are accumulated into `dst`
    /// by the formula `dst = dst * alpha^2 + f'`.
    ///
    /// # Panics
    ///
    /// Panics if `src` is not double the length of `dst`.
    // TODO(andrew): Make folding factor generic.
    // TODO(andrew): Fold directly into FRI layer to prevent allocation.
    fn fold_circle_into_line<F: Field>(
        dst: &mut LineEvaluation<Self, SecureField, BitReversedOrder>,
        src: &CircleEvaluation<Self, F, BitReversedOrder>,
        alpha: SecureField,
    ) where
        F: ExtensionOf<BaseField>,
        SecureField: ExtensionOf<F> + Field,
        Self: FieldOps<F>;
}

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
pub struct FriProver<B: FriOps, H: Hasher> {
    config: FriConfig,
    inner_layers: Vec<FriLayerProver<B, H>>,
    last_layer_poly: LinePoly<SecureField>,
    /// Unique sizes of committed columns sorted in descending order.
    column_log_sizes: Vec<u32>,
}

impl<B: FriOps, H: Hasher<NativeType = u8>> FriProver<B, H> {
    /// Commits to multiple [CircleEvaluation]s.
    ///
    /// `columns` must be provided in descending order by size.
    ///
    /// Mixed degree STARKs involve polynomials evaluated on multiple domains of different size.
    /// Combining evaluations on different sized domains into an evaluation of a single polynomial
    /// on a single domain for the purpose of commitment is inefficient. Instead, commit to multiple
    /// polynomials so combining of evaluations can be taken care of efficiently at the appropriate
    /// FRI layer. All evaluations must be taken over canonic [CircleDomain]s.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `columns` is empty or not sorted in ascending order by domain size.
    /// * An evaluation is not from a sufficiently low degree circle polynomial.
    /// * An evaluation's domain is smaller than the last layer.
    /// * An evaluation's domain is not a canonic circle domain.
    // TODO(andrew): Add docs for all evaluations needing to be from canonic domains.
    pub fn commit<F>(
        channel: &mut impl Channel<Digest = H::Hash>,
        config: FriConfig,
        columns: &[CircleEvaluation<B, F, BitReversedOrder>],
    ) -> Self
    where
        F: ExtensionOf<BaseField>,
        SecureField: ExtensionOf<F>,
        B: FieldOps<F>,
    {
        assert!(!columns.is_empty(), "no columns");
        assert!(columns.is_sorted_by_key(|e| Reverse(e.len())), "not sorted");
        assert!(columns.iter().all(|e| e.domain.is_canonic()), "not canonic");
        let (inner_layers, last_layer_evaluation) =
            Self::commit_inner_layers(channel, config, columns);
        let last_layer_poly = Self::commit_last_layer(channel, config, last_layer_evaluation);

        let column_log_sizes = columns
            .iter()
            .map(|e| e.domain.log_size())
            .dedup()
            .collect();
        Self {
            config,
            inner_layers,
            last_layer_poly,
            column_log_sizes,
        }
    }

    /// Builds and commits to the inner FRI layers (all layers except the last layer).
    ///
    /// All `columns` must be provided in descending order by size.
    ///
    /// Returns all inner layers and the evaluation of the last layer.
    fn commit_inner_layers<F>(
        channel: &mut impl Channel<Digest = H::Hash>,
        config: FriConfig,
        columns: &[CircleEvaluation<B, F, BitReversedOrder>],
    ) -> (
        Vec<FriLayerProver<B, H>>,
        LineEvaluation<B, SecureField, BitReversedOrder>,
    )
    where
        F: ExtensionOf<BaseField>,
        SecureField: ExtensionOf<F>,
        B: FriOps + FieldOps<F>,
    {
        // Returns the length of the [LineEvaluation] a [CircleEvaluation] gets folded into.
        let folded_len = |e: &CircleEvaluation<B, F, _>| e.len() >> CIRCLE_TO_LINE_FOLD_STEP;

        let first_layer_size = folded_len(&columns[0]);
        let first_layer_domain = LineDomain::new(Coset::half_odds(first_layer_size.ilog2()));
        let mut layer_evaluation = LineEvaluation::new_zero(first_layer_domain);

        let mut columns = columns.iter().peekable();

        let mut layers = Vec::new();

        // Circle polynomials can all be folded with the same alpha.
        let circle_poly_alpha = channel.draw_felt();

        while layer_evaluation.len() > config.last_layer_domain_size() {
            // Check for any columns (circle poly evaluations) that should be combined.
            while let Some(column) = columns.next_if(|c| folded_len(c) == layer_evaluation.len()) {
                B::fold_circle_into_line(&mut layer_evaluation, column, circle_poly_alpha);
            }

            let layer = FriLayerProver::new(layer_evaluation);
            channel.mix_digest(layer.merkle_tree.root());
            let folding_alpha = channel.draw_felt();
            let folded_layer_evaluation = B::fold_line(&layer.evaluation, folding_alpha);

            layer_evaluation = folded_layer_evaluation;
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
        channel: &mut impl Channel<Digest = H::Hash>,
        config: FriConfig,
        evaluation: LineEvaluation<B, SecureField, BitReversedOrder>,
    ) -> LinePoly<SecureField> {
        assert_eq!(evaluation.len(), config.last_layer_domain_size());

        let evaluation = evaluation.bit_reverse().to_cpu();
        let mut coeffs = evaluation.interpolate().into_ordered_coefficients();

        let last_layer_degree_bound = 1 << config.log_last_layer_degree_bound;
        let zeros = coeffs.split_off(last_layer_degree_bound);
        assert!(zeros.iter().all(SecureField::is_zero), "invalid degree");

        let last_layer_poly = LinePoly::from_ordered_coefficients(coeffs);
        channel.mix_felts(&last_layer_poly);

        last_layer_poly
    }

    /// Generates a FRI proof and returns it with the opening positions for the committed columns.
    ///
    /// Returned column opening positions are mapped by their log size.
    pub fn decommit(
        self,
        channel: &mut impl Channel<Digest = H::Hash>,
    ) -> (FriProof<H>, BTreeMap<u32, SparseSubCircleDomain>) {
        let max_column_log_size = self.column_log_sizes[0];
        let queries = Queries::generate(channel, max_column_log_size, self.config.n_queries);
        let positions = get_opening_positions(&queries, &self.column_log_sizes);
        let proof = self.decommit_on_queries(&queries);
        (proof, positions)
    }

    /// # Panics
    ///
    /// Panics if the queries were sampled on the wrong domain size.
    fn decommit_on_queries(self, queries: &Queries) -> FriProof<H> {
        let max_column_log_size = self.column_log_sizes[0];
        assert_eq!(queries.log_domain_size, max_column_log_size);
        let first_layer_queries = queries.fold(CIRCLE_TO_LINE_FOLD_STEP);
        let inner_layers = self
            .inner_layers
            .into_iter()
            .scan(first_layer_queries, |layer_queries, layer| {
                let layer_proof = layer.decommit(layer_queries);
                *layer_queries = layer_queries.fold(FOLD_STEP);
                Some(layer_proof)
            })
            .collect();

        let last_layer_poly = self.last_layer_poly;

        FriProof {
            inner_layers,
            last_layer_poly,
        }
    }
}

pub struct FriVerifier<H: Hasher> {
    config: FriConfig,
    /// Alpha used to fold all circle polynomials to univariate polynomials.
    circle_poly_alpha: SecureField,
    /// Domain size queries should be sampled from.
    expected_query_log_domain_size: u32,
    /// The list of degree bounds of all committed circle polynomials.
    column_bounds: Vec<CirclePolyDegreeBound>,
    inner_layers: Vec<FriLayerVerifier<H>>,
    last_layer_domain: LineDomain,
    last_layer_poly: LinePoly<SecureField>,
    /// The queries used for decommitment. Initialized when calling
    /// [`FriVerifier::column_opening_positions`].
    queries: Option<Queries>,
}

impl<H: Hasher<NativeType = u8>> FriVerifier<H> {
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
        channel: &mut impl Channel<Digest = H::Hash>,
        config: FriConfig,
        proof: FriProof<H>,
        column_bounds: Vec<CirclePolyDegreeBound>,
    ) -> Result<Self, FriVerificationError> {
        assert!(column_bounds.is_sorted_by_key(|b| Reverse(*b)));

        let max_column_bound = column_bounds[0];
        let expected_query_log_domain_size =
            max_column_bound.log_degree_bound + config.log_blowup_factor;

        // Circle polynomials can all be folded with the same alpha.
        let circle_poly_alpha = channel.draw_felt();

        let mut inner_layers = Vec::new();
        let mut layer_bound = max_column_bound.fold_to_line();
        let mut layer_domain = LineDomain::new(Coset::half_odds(
            layer_bound.log_degree_bound + config.log_blowup_factor,
        ));

        for (layer_index, proof) in proof.inner_layers.into_iter().enumerate() {
            channel.mix_digest(proof.commitment);

            let folding_alpha = channel.draw_felt();

            inner_layers.push(FriLayerVerifier {
                degree_bound: layer_bound,
                domain: layer_domain,
                folding_alpha,
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
            circle_poly_alpha,
            column_bounds,
            expected_query_log_domain_size,
            inner_layers,
            last_layer_domain,
            last_layer_poly,
            queries: None,
        })
    }

    /// Verifies the decommitment stage of FRI.
    ///
    /// The decommitment values need to be provided in the same order as their commitment.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The queries were not yet sampled.
    /// * The queries were sampled on the wrong domain size.
    /// * There aren't the same number of decommitted values as degree bounds.
    // TODO(andrew): Finish docs.
    pub fn decommit<F>(
        mut self,
        decommitted_values: Vec<SparseCircleEvaluation<F>>,
    ) -> Result<(), FriVerificationError>
    where
        F: ExtensionOf<BaseField>,
        SecureField: ExtensionOf<F>,
    {
        let queries = self.queries.take().expect("queries not sampled");
        self.decommit_on_queries(&queries, decommitted_values)
    }

    fn decommit_on_queries<F>(
        self,
        queries: &Queries,
        decommitted_values: Vec<SparseCircleEvaluation<F>>,
    ) -> Result<(), FriVerificationError>
    where
        F: ExtensionOf<BaseField>,
        SecureField: ExtensionOf<F>,
    {
        assert_eq!(queries.log_domain_size, self.expected_query_log_domain_size);
        assert_eq!(decommitted_values.len(), self.column_bounds.len());

        let (last_layer_queries, last_layer_query_evals) =
            self.decommit_inner_layers(queries, decommitted_values)?;

        self.decommit_last_layer(last_layer_queries, last_layer_query_evals)
    }

    /// Verifies all inner layer decommitments.
    ///
    /// Returns the queries and query evaluations needed for verifying the last FRI layer.
    fn decommit_inner_layers<F>(
        &self,
        queries: &Queries,
        decommitted_values: Vec<SparseCircleEvaluation<F>>,
    ) -> Result<(Queries, Vec<SecureField>), FriVerificationError>
    where
        F: ExtensionOf<BaseField>,
        SecureField: ExtensionOf<F> + Field,
    {
        let circle_poly_alpha = self.circle_poly_alpha;
        let circle_poly_alpha_sq = circle_poly_alpha * circle_poly_alpha;

        let mut decommitted_values = decommitted_values.into_iter();
        let mut column_bounds = self.column_bounds.iter().copied().peekable();
        let mut layer_queries = queries.fold(CIRCLE_TO_LINE_FOLD_STEP);
        let mut layer_query_evals = vec![SecureField::zero(); layer_queries.len()];

        for layer in self.inner_layers.iter() {
            // Check for column evals that need to folded into this layer.
            while column_bounds
                .next_if(|b| b.fold_to_line() == layer.degree_bound)
                .is_some()
            {
                let sparse_evaluation = decommitted_values.next().unwrap();
                let folded_evals = sparse_evaluation.fold(circle_poly_alpha);
                assert_eq!(folded_evals.len(), layer_query_evals.len());

                for (layer_eval, folded_eval) in zip(&mut layer_query_evals, folded_evals) {
                    *layer_eval = *layer_eval * circle_poly_alpha_sq + folded_eval;
                }
            }

            (layer_queries, layer_query_evals) =
                layer.verify_and_fold(layer_queries, layer_query_evals)?;
        }

        // Check all values have been consumed.
        assert!(column_bounds.is_empty());
        assert!(decommitted_values.is_empty());

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

    /// Samples queries and returns the opening positions for each unique column size.
    ///
    /// The order of the opening positions corresponds to the order of the column commitment.
    pub fn column_opening_positions(
        &mut self,
        channel: &mut impl Channel<Digest = H::Hash>,
    ) -> BTreeMap<u32, SparseSubCircleDomain> {
        let column_log_sizes = self
            .column_bounds
            .iter()
            .dedup()
            .map(|b| b.log_degree_bound + self.config.log_blowup_factor)
            .collect_vec();
        let queries = Queries::generate(channel, column_log_sizes[0], self.config.n_queries);
        let positions = get_opening_positions(&queries, &column_log_sizes);
        self.queries = Some(queries);
        positions
    }
}

/// Returns the column opening positions needed for verification.
///
/// The column log sizes must be unique and in descending order. Returned
/// column opening positions are mapped by their log size.
fn get_opening_positions(
    queries: &Queries,
    column_log_sizes: &[u32],
) -> BTreeMap<u32, SparseSubCircleDomain> {
    let mut prev_log_size = column_log_sizes[0];
    assert!(prev_log_size == queries.log_domain_size);
    let mut prev_queries = queries.clone();
    let mut positions = BTreeMap::new();
    positions.insert(prev_log_size, prev_queries.opening_positions(FOLD_STEP));
    for log_size in column_log_sizes.iter().skip(1) {
        let n_folds = prev_log_size - log_size;
        let queries = prev_queries.fold(n_folds);
        positions.insert(*log_size, queries.opening_positions(FOLD_STEP));
        prev_log_size = *log_size;
        prev_queries = queries;
    }
    positions
}

pub trait FriChannel {
    type Digest;

    type Field;

    /// Reseeds the channel with a commitment to an inner FRI layer.
    fn reseed_with_inner_layer(&mut self, commitment: &Self::Digest);

    /// Reseeds the channel with the FRI last layer polynomial.
    fn reseed_with_last_layer(&mut self, last_layer: &LinePoly<Self::Field>);

    /// Draws a random field element.
    fn draw(&mut self) -> Self::Field;
}

#[derive(Clone, Copy, Debug, Error)]
pub enum FriVerificationError {
    #[error("proof contains an invalid number of FRI layers")]
    InvalidNumFriLayers,
    #[error("queries do not resolve to their commitment in layer {layer}")]
    InnerLayerCommitmentInvalid { layer: usize },
    #[error("evaluations are invalid in layer {layer}")]
    InnerLayerEvaluationsInvalid { layer: usize },
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
    pub fn new(log_degree_bound: u32) -> Self {
        Self { log_degree_bound }
    }

    /// Maps a circle polynomial's degree bound to the degree bound of the univariate (line)
    /// polynomial it gets folded into.
    fn fold_to_line(&self) -> LinePolyDegreeBound {
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
    fn fold(self, n_folds: u32) -> Option<Self> {
        if self.log_degree_bound < n_folds {
            return None;
        }

        let log_degree_bound = self.log_degree_bound - n_folds;
        Some(Self { log_degree_bound })
    }
}

/// A FRI proof.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        deserialize = "<H as Hasher>::Hash: serde::Deserialize<'de>",
        serialize = "<H as Hasher>::Hash: serde::Serialize"
    ))
)]
pub struct FriProof<H: Hasher> {
    pub inner_layers: Vec<FriLayerProof<H>>,
    pub last_layer_poly: LinePoly<SecureField>,
}

/// Number of folds for univariate polynomials.
// TODO(andrew): Support different step sizes.
pub const FOLD_STEP: u32 = 1;

/// Number of folds when folding a circle polynomial to univariate polynomial.
pub const CIRCLE_TO_LINE_FOLD_STEP: u32 = 1;

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
///
/// The subset corresponds to the set of evaluations needed by a FRI verifier.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        deserialize = "<H as Hasher>::Hash: serde::Deserialize<'de>",
        serialize = "<H as Hasher>::Hash: serde::Serialize"
    ))
)]
pub struct FriLayerProof<H: Hasher> {
    /// The subset stored corresponds to the set of evaluations the verifier doesn't have but needs
    /// to fold and verify the merkle decommitment.
    pub evals_subset: Vec<SecureField>,
    pub decommitment: MerkleDecommitment<SecureField, H>,
    pub commitment: H::Hash,
}

struct FriLayerVerifier<H: Hasher> {
    degree_bound: LinePolyDegreeBound,
    domain: LineDomain,
    folding_alpha: SecureField,
    layer_index: usize,
    proof: FriLayerProof<H>,
}

impl<H: Hasher<NativeType = u8>> FriLayerVerifier<H> {
    /// Verifies the layer's merkle decommitment and returns the the folded queries and query evals.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof doesn't store enough evaluations.
    /// * The merkle decommitment is invalid.
    ///
    /// # Panics
    ///
    /// Panics if the number of queries doesn't match the number of evals.
    fn verify_and_fold(
        &self,
        queries: Queries,
        evals_at_queries: Vec<SecureField>,
    ) -> Result<(Queries, Vec<SecureField>), FriVerificationError> {
        let decommitment = &self.proof.decommitment;
        let commitment = self.proof.commitment;

        // Extract the evals needed for decommitment and folding.
        let sparse_evaluation = self.extract_evaluation(&queries, &evals_at_queries)?;

        // TODO: When leaf values are removed from the decommitment, also remove this block.
        {
            let mut expected_decommitment_evals = Vec::new();

            for leaf in decommitment.values() {
                // Ensure each leaf is a single value.
                if let [eval] = *leaf {
                    expected_decommitment_evals.push(eval);
                } else {
                    return Err(FriVerificationError::InnerLayerCommitmentInvalid {
                        layer: self.layer_index,
                    });
                }
            }

            let actual_decommitment_evals = sparse_evaluation
                .subline_evals
                .iter()
                .flat_map(|e| &e.values);

            if !actual_decommitment_evals.eq(&expected_decommitment_evals) {
                return Err(FriVerificationError::InnerLayerCommitmentInvalid {
                    layer: self.layer_index,
                });
            }
        }

        let folded_queries = queries.fold(FOLD_STEP);

        // Positions of all the decommitment evals.
        let decommitment_positions = folded_queries
            .iter()
            .flat_map(|folded_query| {
                let start = folded_query << FOLD_STEP;
                let end = start + (1 << FOLD_STEP);
                start..end
            })
            .collect::<Vec<usize>>();

        if !decommitment.verify(commitment, &decommitment_positions) {
            return Err(FriVerificationError::InnerLayerCommitmentInvalid {
                layer: self.layer_index,
            });
        }

        let evals_at_folded_queries = sparse_evaluation.fold(self.folding_alpha);

        Ok((folded_queries, evals_at_folded_queries))
    }

    /// Returns the evaluations needed for decommitment.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the proof doesn't store enough evaluations.
    ///
    /// # Panics
    ///
    /// Panics if the number of queries doesn't match the number of evals.
    fn extract_evaluation(
        &self,
        queries: &Queries,
        evals_at_queries: &[SecureField],
    ) -> Result<SparseLineEvaluation, FriVerificationError> {
        // Evals provided by the verifier.
        let mut evals_at_queries = evals_at_queries.iter().copied();

        // Evals stored in the proof.
        let mut proof_evals = self.proof.evals_subset.iter().copied();

        let mut all_subline_evals = Vec::new();

        // Group queries by the subline they reside in.
        for subline_queries in queries.group_by(|a, b| a >> FOLD_STEP == b >> FOLD_STEP) {
            let subline_start = (subline_queries[0] >> FOLD_STEP) << FOLD_STEP;
            let subline_end = subline_start + (1 << FOLD_STEP);

            let mut subline_evals = Vec::new();
            let mut subline_queries = subline_queries.iter().peekable();

            // Insert the evals.
            for eval_position in subline_start..subline_end {
                let eval = match subline_queries.next_if_eq(&&eval_position) {
                    Some(_) => evals_at_queries.next().unwrap(),
                    None => proof_evals.next().ok_or(
                        FriVerificationError::InnerLayerEvaluationsInvalid {
                            layer: self.layer_index,
                        },
                    )?,
                };

                subline_evals.push(eval);
            }

            // Construct the domain.
            // TODO(andrew): Create a constructor for LineDomain.
            let subline_initial_index = bit_reverse_index(subline_start, self.domain.log_size());
            let subline_initial = self.domain.coset().index_at(subline_initial_index);
            let subline_domain = LineDomain::new(Coset::new(subline_initial, FOLD_STEP));

            all_subline_evals.push(LineEvaluation::new(subline_domain, subline_evals));
        }

        // Check all proof evals have been consumed.
        if !proof_evals.is_empty() {
            return Err(FriVerificationError::InnerLayerEvaluationsInvalid {
                layer: self.layer_index,
            });
        }

        Ok(SparseLineEvaluation::new(all_subline_evals))
    }
}

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): Support different step sizes.
struct FriLayerProver<B: FriOps, H: Hasher> {
    evaluation: LineEvaluation<B, SecureField, BitReversedOrder>,
    merkle_tree: MerkleTree<SecureField, H>,
}

impl<B: FriOps, H: Hasher<NativeType = u8>> FriLayerProver<B, H> {
    fn new(evaluation: LineEvaluation<B, SecureField, BitReversedOrder>) -> Self {
        // TODO(spapini): Commit on slice.
        // TODO(spapini): Merkle tree in backend.
        let merkle_tree = MerkleTree::commit(vec![evaluation.values.to_vec()]);
        #[allow(unreachable_code)]
        FriLayerProver {
            evaluation,
            merkle_tree,
        }
    }

    /// Generates a decommitment of the subline evaluations at the specified positions.
    fn decommit(self, queries: &Queries) -> FriLayerProof<H> {
        let mut decommit_positions = Vec::new();
        let mut evals_subset = Vec::new();

        // Group queries by the subline they reside in.
        // TODO(andrew): Explain what a "subline" is at the top of the module.
        for query_group in queries.group_by(|a, b| a >> FOLD_STEP == b >> FOLD_STEP) {
            let subline_start = (query_group[0] >> FOLD_STEP) << FOLD_STEP;
            let subline_end = subline_start + (1 << FOLD_STEP);

            let mut subline_queries = query_group.iter().peekable();

            for eval_position in subline_start..subline_end {
                // Add decommitment position.
                decommit_positions.push(eval_position);

                // Skip evals the verifier can calculate.
                if subline_queries.next_if_eq(&&eval_position).is_some() {
                    continue;
                }

                let eval = self.evaluation.values.at(eval_position);
                evals_subset.push(eval);
            }
        }

        let commitment = self.merkle_tree.root();
        let decommitment = self.merkle_tree.generate_decommitment(decommit_positions);

        FriLayerProof {
            evals_subset,
            decommitment,
            commitment,
        }
    }
}

/// Holds a foldable subset of circle polynomial evaluations.
#[derive(Debug, Clone)]
pub struct SparseCircleEvaluation<F: ExtensionOf<BaseField>> {
    subcircle_evals: Vec<CircleEvaluation<CPUBackend, F, BitReversedOrder>>,
}

impl<F: ExtensionOf<BaseField>> SparseCircleEvaluation<F> {
    /// # Panics
    ///
    /// Panics if the evaluation domain sizes don't equal the folding factor.
    pub fn new(subcircle_evals: Vec<CircleEvaluation<CPUBackend, F, BitReversedOrder>>) -> Self {
        let folding_factor = 1 << CIRCLE_TO_LINE_FOLD_STEP;
        assert!(subcircle_evals.iter().all(|e| e.len() == folding_factor));
        Self { subcircle_evals }
    }

    fn fold(self, alpha: SecureField) -> Vec<SecureField>
    where
        SecureField: ExtensionOf<F>,
    {
        self.subcircle_evals
            .into_iter()
            .map(|e| {
                let buffer_domain = LineDomain::new(e.domain.half_coset);
                let mut buffer = LineEvaluation::new(buffer_domain, vec![SecureField::zero()]);
                CPUBackend::fold_circle_into_line(&mut buffer, &e, alpha);
                buffer.values[0]
            })
            .collect()
    }
}

/// Holds a small foldable subset of univariate SecureField polynomial evaluations.
/// Evaluation is held at the CPU backend.
#[derive(Debug, Clone)]
struct SparseLineEvaluation {
    subline_evals: Vec<CPULineEvaluation<SecureField, BitReversedOrder>>,
}

impl SparseLineEvaluation {
    /// # Panics
    ///
    /// Panics if the evaluation domain sizes don't equal the folding factor.
    fn new(subline_evals: Vec<CPULineEvaluation<SecureField, BitReversedOrder>>) -> Self {
        let folding_factor = 1 << FOLD_STEP;
        assert!(subline_evals.iter().all(|e| e.len() == folding_factor));
        Self { subline_evals }
    }

    fn fold(self, alpha: SecureField) -> Vec<SecureField> {
        self.subline_evals
            .into_iter()
            .map(|e| CPUBackend::fold_line(&e, alpha).values[0])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::{One, Zero};

    use super::{get_opening_positions, FriVerificationError, SparseCircleEvaluation};
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly, CPULineEvaluation};
    use crate::core::backend::CPUBackend;
    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::{ExtensionOf, Field};
    use crate::core::fri::{
        CirclePolyDegreeBound, FriConfig, FriOps, FriVerifier, CIRCLE_TO_LINE_FOLD_STEP,
    };
    use crate::core::poly::circle::{CircleDomain, CircleEvaluation};
    use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};
    use crate::core::poly::{BitReversedOrder, NaturalOrder};
    use crate::core::queries::{Queries, SparseSubCircleDomain};
    use crate::core::test_utils::test_channel;

    /// Default blowup factor used for tests.
    const LOG_BLOWUP_FACTOR: u32 = 2;

    type FriProver = super::FriProver<CPUBackend, Blake2sHasher>;

    #[test]
    fn fold_line_works() {
        const DEGREE: usize = 8;
        // Coefficients are bit-reversed.
        let even_coeffs: [SecureField; DEGREE / 2] = [1, 2, 1, 3]
            .map(BaseField::from_u32_unchecked)
            .map(SecureField::from);
        let odd_coeffs: [SecureField; DEGREE / 2] = [3, 5, 4, 1]
            .map(BaseField::from_u32_unchecked)
            .map(SecureField::from);
        let poly = LinePoly::new([even_coeffs, odd_coeffs].concat());
        let even_poly = LinePoly::new(even_coeffs.to_vec());
        let odd_poly = LinePoly::new(odd_coeffs.to_vec());
        let alpha = BaseField::from_u32_unchecked(19283).into();
        let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2()));
        let drp_domain = domain.double();
        let evals = poly.evaluate(domain).bit_reverse();

        let drp_evals = CPUBackend::fold_line(&evals, alpha);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        let drp_evals = drp_evals.bit_reverse();
        for (i, (&drp_eval, x)) in zip(&drp_evals.values, drp_domain).enumerate() {
            let f_e: SecureField = even_poly.eval_at_point(x.into());
            let f_o: SecureField = odd_poly.eval_at_point(x.into());
            assert_eq!(drp_eval, (f_e + alpha * f_o).double(), "mismatch at {i}");
        }
    }

    #[test]
    fn fold_circle_to_line_works() {
        const LOG_DEGREE: u32 = 4;
        let circle_evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let num_folded_evals = circle_evaluation.domain.size() >> CIRCLE_TO_LINE_FOLD_STEP;
        let alpha = SecureField::one();
        let folded_domain = LineDomain::new(circle_evaluation.domain.half_coset);

        let mut folded_evaluation =
            LineEvaluation::new(folded_domain, vec![Zero::zero(); num_folded_evals]);
        CPUBackend::fold_circle_into_line(&mut folded_evaluation, &circle_evaluation, alpha);

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
        let evaluation = polynomial_evaluation(6, LOG_INVALID_BLOWUP_FACTOR);

        FriProver::commit(&mut test_channel(), config, &[evaluation]);
    }

    #[test]
    #[should_panic = "not canonic"]
    fn committing_evaluation_from_invalid_domain_fails() {
        let invalid_domain = CircleDomain::new(Coset::new(CirclePointIndex::generator(), 3));
        assert!(!invalid_domain.is_canonic(), "must be an invalid domain");
        let evaluation = CircleEvaluation::new(invalid_domain, vec![BaseField::one(); 1 << 4]);

        FriProver::commit(&mut test_channel(), FriConfig::new(2, 2, 3), &[evaluation]);
    }

    #[test]
    fn valid_proof_passes_verification() -> Result<(), FriVerificationError> {
        const LOG_DEGREE: u32 = 3;
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let config = FriConfig::new(1, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
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
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let config = FriConfig::new(LAST_LAYER_LOG_BOUND, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
        let proof = prover.decommit_on_queries(&queries);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        verifier.decommit_on_queries(&queries, vec![decommitment_value])
    }

    #[test]
    fn valid_mixed_degree_proof_passes_verification() -> Result<(), FriVerificationError> {
        const LOG_DEGREES: [u32; 3] = [6, 5, 4];
        let polynomials = LOG_DEGREES.map(|log_d| polynomial_evaluation(log_d, LOG_BLOWUP_FACTOR));
        let log_domain_size = polynomials[0].domain.log_size();
        let queries = Queries::from_positions(vec![7, 70], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let prover = FriProver::commit(&mut test_channel(), config, &polynomials);
        let decommitment_values = polynomials.map(|p| query_polynomial(&p, &queries)).to_vec();
        let proof = prover.decommit_on_queries(&queries);
        let bounds = LOG_DEGREES.map(CirclePolyDegreeBound::new).to_vec();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bounds).unwrap();

        verifier.decommit_on_queries(&queries, decommitment_values)
    }

    #[test]
    fn valid_mixed_degree_end_to_end_proof_passes_verification() -> Result<(), FriVerificationError>
    {
        const LOG_DEGREES: [u32; 3] = [6, 5, 4];
        let polynomials = LOG_DEGREES.map(|log_d| polynomial_evaluation(log_d, LOG_BLOWUP_FACTOR));
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3);
        let prover = FriProver::commit(&mut test_channel(), config, &polynomials);
        let (proof, prover_opening_positions) = prover.decommit(&mut test_channel());
        let decommitment_values = zip(&polynomials, prover_opening_positions.values().rev())
            .map(|(poly, positions)| open_polynomial(poly, positions))
            .collect();
        let bounds = LOG_DEGREES.map(CirclePolyDegreeBound::new).to_vec();

        let mut verifier = FriVerifier::commit(&mut test_channel(), config, proof, bounds).unwrap();
        let verifier_opening_positions = verifier.column_opening_positions(&mut test_channel());

        assert_eq!(prover_opening_positions, verifier_opening_positions);
        verifier.decommit(decommitment_values)
    }

    #[test]
    fn proof_with_removed_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let polynomial = polynomial_evaluation(6, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
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
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
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
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let mut proof = prover.decommit_on_queries(&queries);
        // Remove an evaluation from the second layer's proof.
        proof.inner_layers[1].evals_subset.pop();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, vec![decommitment_value]);

        assert!(matches!(
            verification_result,
            Err(FriVerificationError::InnerLayerEvaluationsInvalid { layer: 1 })
        ));
    }

    #[test]
    fn proof_with_invalid_inner_layer_decommitment_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let mut proof = prover.decommit_on_queries(&queries);
        // Modify the committed values in the second layer.
        proof.inner_layers[1].evals_subset[0] += BaseField::one();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, vec![decommitment_value]);

        assert!(matches!(
            verification_result,
            Err(FriVerificationError::InnerLayerCommitmentInvalid { layer: 1 })
        ));
    }

    #[test]
    fn proof_with_invalid_last_layer_degree_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        const LOG_MAX_LAST_LAYER_DEGREE: u32 = 2;
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1, 7, 8], log_domain_size);
        let config = FriConfig::new(LOG_MAX_LAST_LAYER_DEGREE, LOG_BLOWUP_FACTOR, queries.len());
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
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
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1, 7, 8], log_domain_size);
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
        let bound = vec![CirclePolyDegreeBound::new(LOG_DEGREE)];
        let mut proof = prover.decommit_on_queries(&queries);
        // Compromise the last layer polynomial's first coefficient.
        proof.last_layer_poly[0] += BaseField::one();
        let verifier = FriVerifier::commit(&mut test_channel(), config, proof, bound).unwrap();

        let verification_result = verifier.decommit_on_queries(&queries, vec![decommitment_value]);

        assert!(matches!(
            verification_result,
            Err(FriVerificationError::LastLayerEvaluationsInvalid)
        ));
    }

    #[test]
    #[should_panic]
    fn decommit_queries_on_invalid_domain_fails_verification() {
        const LOG_DEGREE: u32 = 3;
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let config = FriConfig::new(1, LOG_BLOWUP_FACTOR, queries.len());
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::commit(&mut test_channel(), config, &[polynomial]);
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
    ) -> CPUCircleEvaluation<BaseField, BitReversedOrder> {
        let poly = CPUCirclePoly::new(vec![BaseField::one(); 1 << log_degree]);
        let coset = Coset::half_odds(log_degree + log_blowup_factor - 1);
        let domain = CircleDomain::new(coset);
        poly.evaluate(domain)
    }

    /// Returns the log degree bound of a polynomial.
    fn log_degree_bound<F: ExtensionOf<BaseField>>(
        polynomial: CPULineEvaluation<F, BitReversedOrder>,
    ) -> u32 {
        let polynomial = polynomial.bit_reverse();
        let coeffs = polynomial.interpolate().into_ordered_coefficients();
        let degree = coeffs.into_iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        (degree + 1).ilog2()
    }

    // TODO: Remove after SubcircleDomain integration.
    fn query_polynomial<F: ExtensionOf<BaseField>>(
        polynomial: &CPUCircleEvaluation<F, BitReversedOrder>,
        queries: &Queries,
    ) -> SparseCircleEvaluation<F> {
        let polynomial_log_size = polynomial.domain.log_size();
        let positions =
            get_opening_positions(queries, &[queries.log_domain_size, polynomial_log_size]);
        open_polynomial(polynomial, &positions[&polynomial_log_size])
    }

    fn open_polynomial<F: ExtensionOf<BaseField>>(
        polynomial: &CPUCircleEvaluation<F, BitReversedOrder>,
        positions: &SparseSubCircleDomain,
    ) -> SparseCircleEvaluation<F> {
        let polynomial = polynomial.clone().bit_reverse();

        let coset_evals = positions
            .iter()
            .map(|position| {
                let coset_domain = position.to_circle_domain(&polynomial.domain);
                let evals = coset_domain
                    .iter_indices()
                    .map(|p| polynomial.get_at(p))
                    .collect();
                let coset_eval = CPUCircleEvaluation::<F, NaturalOrder>::new(coset_domain, evals);
                coset_eval.bit_reverse()
            })
            .collect();

        SparseCircleEvaluation::new(coset_evals)
    }
}
