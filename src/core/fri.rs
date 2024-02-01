use std::cmp::Reverse;
use std::fmt::Debug;
use std::iter::zip;
use std::ops::RangeInclusive;

use thiserror::Error;

use super::fields::m31::BaseField;
use super::fields::{ExtensionOf, Field};
use super::poly::circle::CircleEvaluation;
use super::poly::line::{LineEvaluation, LinePoly};
use super::poly::BitReversedOrder;
// TODO(andrew): Create fri/ directory, move queries.rs there and split this file up.
use super::queries::Queries;
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::poly::line::LineDomain;
use crate::core::utils::bit_reverse_index;

/// FRI proof config
// TODO(andrew): Support different step sizes.
#[derive(Debug, Clone, Copy)]
pub struct FriConfig {
    log_blowup_factor: u32,
    log_last_layer_degree_bound: u32,
    // TODO(andrew): Add num_queries, fold_steps.
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
    pub fn new(log_last_layer_degree_bound: u32, log_blowup_factor: u32) -> Self {
        assert!(Self::LOG_LAST_LAYER_DEGREE_BOUND_RANGE.contains(&log_last_layer_degree_bound));
        assert!(Self::LOG_BLOWUP_FACTOR_RANGE.contains(&log_blowup_factor));
        Self {
            log_blowup_factor,
            log_last_layer_degree_bound,
        }
    }

    fn last_layer_domain_size(&self) -> usize {
        1 << (self.log_last_layer_degree_bound + self.log_blowup_factor)
    }
}

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
pub struct FriProver<F: ExtensionOf<BaseField>, H: Hasher> {
    inner_layers: Vec<FriLayerProver<F, H>>,
    last_layer_poly: LinePoly<F>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher<NativeType = u8>> FriProver<F, H> {
    /// Commits to multiple [CircleEvaluation]s.
    ///
    /// `evals` must be provided in descending order by size.
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
    /// * `evals` is empty or not sorted in ascending order by domain size.
    /// * An evaluation is not from a sufficiently low degree circle polynomial.
    /// * An evaluation's domain is smaller than the last layer.
    /// * An evaluation's domain is not a canonic circle domain.
    // TODO(andrew): Add docs for all evaluations needing to be from canonic domains.
    pub fn commit(config: FriConfig, columns: Vec<CircleEvaluation<F, BitReversedOrder>>) -> Self {
        assert!(columns.is_sorted_by_key(|e| Reverse(e.len())), "not sorted");
        assert!(columns.iter().all(|e| e.domain.is_canonic()), "not canonic");
        let (inner_layers, last_layer_evaluation) = Self::commit_inner_layers(config, columns);
        let last_layer_poly = Self::commit_last_layer(config, last_layer_evaluation);
        Self {
            inner_layers,
            last_layer_poly,
        }
    }

    /// Builds and commits to the inner FRI layers (all layers except the last layer).
    ///
    /// All `columns` must be provided in descending order by size.
    ///
    /// Returns all inner layers and the evaluation of the last layer.
    fn commit_inner_layers(
        config: FriConfig,
        columns: Vec<CircleEvaluation<F, BitReversedOrder>>,
    ) -> (
        Vec<FriLayerProver<F, H>>,
        LineEvaluation<F, BitReversedOrder>,
    ) {
        // Returns the length of the [LineEvaluation] a [CircleEvaluation] gets folded into.
        let folded_len = |e: &CircleEvaluation<_, _>| e.len() >> CIRCLE_TO_LINE_FOLD_STEP;

        let first_layer_size = folded_len(&columns[0]);
        let first_layer_domain = LineDomain::new(Coset::half_odds(first_layer_size.ilog2()));
        let mut layer_evaluation = LineEvaluation::new_zero(first_layer_domain);

        let mut columns = columns.into_iter().peekable();

        let mut layers = Vec::new();

        // Circle polynomials can all be folded with the same alpha.
        // TODO(andrew): draw random alpha from channel
        let circle_poly_alpha = F::one();

        while layer_evaluation.len() > config.last_layer_domain_size() {
            // Check for any columns (circle poly evaluations) that should be combined.
            while let Some(column) = columns.next_if(|c| folded_len(c) == layer_evaluation.len()) {
                fold_circle_into_line(&mut layer_evaluation, &column, circle_poly_alpha);
            }

            let layer = FriLayerProver::new(layer_evaluation);

            // TODO(andrew): add merkle root to channel
            // TODO(ohad): Add back once IntoSlice implemented for Field.
            // let _merkle_root = layer.merkle_tree.root();
            // TODO(andrew): draw random alpha from channel
            let alpha = F::one();
            let folded_layer_evaluation = fold_line(&layer.evaluation, alpha);

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
        config: FriConfig,
        evaluation: LineEvaluation<F, BitReversedOrder>,
    ) -> LinePoly<F> {
        assert_eq!(evaluation.len(), config.last_layer_domain_size());

        let evaluation = evaluation.bit_reverse();
        let mut coeffs = evaluation.interpolate().into_ordered_coefficients();

        let last_layer_degree_bound = 1 << config.log_last_layer_degree_bound;
        let zeros = coeffs.split_off(last_layer_degree_bound);
        assert!(zeros.iter().all(F::is_zero), "invalid degree");

        LinePoly::from_ordered_coefficients(coeffs)
        // TODO(andrew): Seed channel with coeffs.
    }

    /// # Panics
    ///
    /// Panics if the queries were sampled on the wrong domain size.
    pub fn decommit(self, queries: &Queries) -> FriProof<F, H> {
        let last_layer_poly = self.last_layer_poly;

        let first_layer_queries = queries.fold(CIRCLE_TO_LINE_FOLD_STEP);
        let inner_layers = self
            .inner_layers
            .into_iter()
            .scan(first_layer_queries, |layer_queries, layer| {
                let layer_proof = layer.decommit(layer_queries);
                *layer_queries = layer_queries.fold(FOLD_STEP);
                Some(layer_proof)
            })
            .collect::<Vec<_>>();

        FriProof {
            inner_layers,
            last_layer_poly,
        }
    }
}

pub struct FriVerifier<F: ExtensionOf<BaseField>, H: Hasher> {
    /// Alpha used to fold all circle polynomials to univariate polynomials.
    circle_poly_alpha: F,
    /// Domain size queries should be sampled from.
    expected_query_log_domain_size: u32,
    /// The list of degree bounds of all committed circle polynomials.
    column_bounds: Vec<CirclePolyDegreeBound>,
    inner_layers: Vec<FriLayerVerifier<F, H>>,
    last_layer_domain: LineDomain,
    last_layer_poly: LinePoly<F>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher<NativeType = u8>> FriVerifier<F, H> {
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
        config: FriConfig,
        proof: FriProof<F, H>,
        column_bounds: Vec<CirclePolyDegreeBound>,
    ) -> Result<Self, VerificationError> {
        assert!(column_bounds.is_sorted_by_key(|b| Reverse(*b)));

        let max_column_bound = column_bounds[0];
        let expected_query_log_domain_size =
            max_column_bound.log_degree_bound + config.log_blowup_factor;

        // Circle polynomials can all be folded with the same alpha.
        // TODO(andrew): Draw alpha from channel.
        let circle_poly_alpha = F::one();

        let mut inner_layers = Vec::new();
        let mut layer_bound = max_column_bound.fold_to_line();
        let mut layer_domain = LineDomain::new(Coset::half_odds(
            layer_bound.log_degree_bound + config.log_blowup_factor,
        ));

        for (layer_index, proof) in proof.inner_layers.into_iter().enumerate() {
            inner_layers.push(FriLayerVerifier {
                degree_bound: layer_bound,
                domain: layer_domain,
                // TODO(andrew): Seed channel with commitment.
                // TODO(andrew): Draw alpha from channel.
                folding_alpha: F::one(),
                layer_index,
                proof,
            });

            layer_bound = layer_bound
                .fold(FOLD_STEP)
                .ok_or(VerificationError::InvalidNumFriLayers)?;
            layer_domain = layer_domain.double();
        }

        if layer_bound.log_degree_bound != config.log_last_layer_degree_bound {
            return Err(VerificationError::InvalidNumFriLayers);
        }

        let last_layer_domain = layer_domain;
        let last_layer_poly = proof.last_layer_poly;

        if last_layer_poly.len() > (1 << config.log_last_layer_degree_bound) {
            return Err(VerificationError::LastLayerDegreeInvalid);
        }

        Ok(Self {
            circle_poly_alpha,
            column_bounds,
            expected_query_log_domain_size,
            inner_layers,
            last_layer_domain,
            last_layer_poly,
        })
    }

    /// Verifies the decommitment stage of FRI.
    ///
    /// The decommitment values need to be provided in the same order as their commitment.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The queries were sampled on the wrong domain size.
    /// * There aren't the same number of decommited values as degree bounds.
    // TODO(andrew): Finish docs.
    pub fn decommit(
        self,
        queries: &Queries,
        decommited_values: Vec<SparseCircleEvaluation<F>>,
    ) -> Result<(), VerificationError> {
        assert_eq!(queries.log_domain_size, self.expected_query_log_domain_size);
        assert_eq!(decommited_values.len(), self.column_bounds.len());

        let (last_layer_queries, last_layer_query_evals) =
            self.decommit_inner_layers(queries, decommited_values)?;

        self.decommit_last_layer(last_layer_queries, last_layer_query_evals)
    }

    /// Verifies all inner layer decommitments.
    ///
    /// Returns the queries and query evaluations needed for verifying the last FRI layer.
    fn decommit_inner_layers(
        &self,
        queries: &Queries,
        decommited_values: Vec<SparseCircleEvaluation<F>>,
    ) -> Result<(Queries, Vec<F>), VerificationError> {
        let circle_poly_alpha = self.circle_poly_alpha;
        let circle_poly_alpha_sq = circle_poly_alpha * circle_poly_alpha;

        let mut decommited_values = decommited_values.into_iter();
        let mut column_bounds = self.column_bounds.iter().copied().peekable();
        let mut layer_queries = queries.fold(CIRCLE_TO_LINE_FOLD_STEP);
        let mut layer_query_evals = vec![F::zero(); layer_queries.len()];

        for layer in self.inner_layers.iter() {
            // Check for column evals that need to folded into this layer.
            while column_bounds
                .next_if(|b| b.fold_to_line() == layer.degree_bound)
                .is_some()
            {
                let sparse_evaluation = decommited_values.next().unwrap();
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
        assert!(decommited_values.is_empty());

        Ok((layer_queries, layer_query_evals))
    }

    /// Verifies the last layer.
    fn decommit_last_layer(
        self,
        queries: Queries,
        query_evals: Vec<F>,
    ) -> Result<(), VerificationError> {
        let Self {
            last_layer_domain: domain,
            last_layer_poly,
            ..
        } = self;

        for (&query, eval) in zip(&*queries, query_evals) {
            let x = domain.at(bit_reverse_index(query, domain.log_size()));

            if eval != last_layer_poly.eval_at_point(x.into()) {
                return Err(VerificationError::LastLayerEvaluationsInvalid);
            }
        }

        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum VerificationError {
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
pub struct FriProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub inner_layers: Vec<FriLayerProof<F, H>>,
    pub last_layer_poly: LinePoly<F>,
}

/// Number of folds for univariate polynomials.
// TODO(andrew): Support different step sizes.
const FOLD_STEP: u32 = 1;

/// Number of folds when folding a circle polynomial to univariate polynomial.
const CIRCLE_TO_LINE_FOLD_STEP: u32 = 1;

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
///
/// The subset corresponds to the set of evaluations needed by a FRI verifier.
pub struct FriLayerProof<F: ExtensionOf<BaseField>, H: Hasher> {
    /// The subset stored corresponds to the set of evaluations the verifier doesn't have but needs
    /// to fold and verify the merkle decommitment.
    pub evals_subset: Vec<F>,
    pub decommitment: MerkleDecommitment<F, H>,
    pub commitment: H::Hash,
}

struct FriLayerVerifier<F: ExtensionOf<BaseField>, H: Hasher> {
    degree_bound: LinePolyDegreeBound,
    domain: LineDomain,
    folding_alpha: F,
    layer_index: usize,
    proof: FriLayerProof<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher<NativeType = u8>> FriLayerVerifier<F, H> {
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
        evals_at_queries: Vec<F>,
    ) -> Result<(Queries, Vec<F>), VerificationError> {
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
                    return Err(VerificationError::InnerLayerCommitmentInvalid {
                        layer: self.layer_index,
                    });
                }
            }

            let actual_decommitment_evals = sparse_evaluation
                .subline_evals
                .iter()
                .flat_map(|e| &**e)
                .copied()
                .collect::<Vec<F>>();

            if expected_decommitment_evals != actual_decommitment_evals {
                return Err(VerificationError::InnerLayerCommitmentInvalid {
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
            return Err(VerificationError::InnerLayerCommitmentInvalid {
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
        evals_at_queries: &[F],
    ) -> Result<SparseLineEvaluation<F>, VerificationError> {
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
                        VerificationError::InnerLayerEvaluationsInvalid {
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
            return Err(VerificationError::InnerLayerEvaluationsInvalid {
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
struct FriLayerProver<F: ExtensionOf<BaseField>, H: Hasher> {
    evaluation: LineEvaluation<F, BitReversedOrder>,
    merkle_tree: MerkleTree<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher<NativeType = u8>> FriLayerProver<F, H> {
    fn new(evaluation: LineEvaluation<F, BitReversedOrder>) -> Self {
        // TODO: Commit on slice.
        let merkle_tree = MerkleTree::commit(vec![evaluation.to_vec()]);
        #[allow(unreachable_code)]
        FriLayerProver {
            evaluation,
            merkle_tree,
        }
    }

    /// Generates a decommitment of the subline evaluations at the specified positions.
    fn decommit(self, queries: &Queries) -> FriLayerProof<F, H> {
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

                let eval = self.evaluation[eval_position];
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
pub struct SparseCircleEvaluation<F: ExtensionOf<BaseField>> {
    subcircle_evals: Vec<CircleEvaluation<F, BitReversedOrder>>,
}

impl<F: ExtensionOf<BaseField>> SparseCircleEvaluation<F> {
    /// # Panics
    ///
    /// Panics if the evaluation domain sizes don't equal the folding factor.
    pub fn new(subcircle_evals: Vec<CircleEvaluation<F, BitReversedOrder>>) -> Self {
        let folding_factor = 1 << CIRCLE_TO_LINE_FOLD_STEP;
        assert!(subcircle_evals.iter().all(|e| e.len() == folding_factor));
        Self { subcircle_evals }
    }

    fn fold(self, alpha: F) -> Vec<F> {
        self.subcircle_evals
            .into_iter()
            .map(|e| {
                let buffer_domain = LineDomain::new(e.domain.half_coset);
                let mut buffer = LineEvaluation::new(buffer_domain, vec![F::zero()]);
                fold_circle_into_line(&mut buffer, &e, alpha);
                buffer[0]
            })
            .collect()
    }
}

/// Holds a foldable subset of univariate polynomial evaluations.
struct SparseLineEvaluation<F: ExtensionOf<BaseField>> {
    subline_evals: Vec<LineEvaluation<F, BitReversedOrder>>,
}

impl<F: ExtensionOf<BaseField>> SparseLineEvaluation<F> {
    /// # Panics
    ///
    /// Panics if the evaluation domain sizes don't equal the folding factor.
    fn new(subline_evals: Vec<LineEvaluation<F, BitReversedOrder>>) -> Self {
        let folding_factor = 1 << FOLD_STEP;
        assert!(subline_evals.iter().all(|e| e.len() == folding_factor));
        Self { subline_evals }
    }

    #[allow(dead_code)]
    fn fold(self, alpha: F) -> Vec<F> {
        self.subline_evals
            .into_iter()
            .map(|e| fold_line(&e, alpha)[0])
            .collect()
    }
}

/// Folds a degree `d` polynomial into a degree `d/2` polynomial.
///
/// Let `evals` be a polynomial evaluated on a [LineDomain] `E`, `alpha` be a random field element
/// and `pi(x) = 2x^2 - 1` be the circle's x-coordinate doubling map. This function returns
/// `f' = f0 + alpha * f1` evaluated on `pi(E)` such that `2f(x) = f0(pi(x)) + x * f1(pi(x))`.
///
/// # Panics
///
/// Panics if there are less than two evaluations.
pub fn fold_line<F: ExtensionOf<BaseField>>(
    evals: &LineEvaluation<F, BitReversedOrder>,
    alpha: F,
) -> LineEvaluation<F, BitReversedOrder> {
    let n = evals.len();
    assert!(n >= 2, "too few evals");

    let domain = evals.domain();

    let folded_evals = evals
        .array_chunks()
        .enumerate()
        .map(|(i, &[f_x, f_neg_x])| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let x = domain.at(bit_reverse_index(i << FOLD_STEP, domain.log_size()));

            let (mut f0, mut f1) = (f_x, f_neg_x);
            ibutterfly(&mut f0, &mut f1, x.inverse());
            f0 + alpha * f1
        })
        .collect();

    LineEvaluation::new(domain.double(), folded_evals)
}

/// Folds and accumulates a degree `d` circle polynomial into a degree `d/2` univariate polynomial.
///
/// Let `src` be the evaluation of a circle polynomial `f` on a [CircleDomain] `E`. This function
/// computes evaluations of `f' = f0 + alpha * f1` on the x-coordinates of `E` such that
/// `2f(p) = f0(px) + py * f1(px)`. The evaluations of `f'` are accumulated into `dst` by the
/// formula `dst = dst * alpha^2 + f'`.
///
/// # Panics
///
/// Panics if `src` is not double the length of `dst`.
// TODO(andrew): Make folding factor generic.
// TODO(andrew): Fold directly into FRI layer to prevent allocation.
fn fold_circle_into_line<F: ExtensionOf<BaseField>>(
    dst: &mut LineEvaluation<F, BitReversedOrder>,
    src: &CircleEvaluation<F, BitReversedOrder>,
    alpha: F,
) {
    assert_eq!(src.len() >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

    let domain = src.domain;
    let alpha_sq = alpha * alpha;

    zip(&mut **dst, src.array_chunks())
        .enumerate()
        .for_each(|(i, (dst, &[f_p, f_neg_p]))| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let p = domain.at(bit_reverse_index(
                i << CIRCLE_TO_LINE_FOLD_STEP,
                domain.log_size(),
            ));

            // Calculate `f0(px)` and `f1(px)` such that `2f(p) = f0(px) + py * f1(px)`.
            let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
            ibutterfly(&mut f0_px, &mut f1_px, p.y.inverse());
            let f_prime = f0_px + alpha * f1_px;

            *dst = *dst * alpha_sq + f_prime;
        });
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::{One, Zero};

    use super::{SparseCircleEvaluation, VerificationError};
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::constraints::{EvalByEvaluation, PolyOracle};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::ExtensionOf;
    use crate::core::fri::{
        fold_circle_into_line, fold_line, CirclePolyDegreeBound, FriConfig, FriProver, FriVerifier,
        CIRCLE_TO_LINE_FOLD_STEP,
    };
    use crate::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};
    use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};
    use crate::core::poly::{BitReversedOrder, NaturalOrder};
    use crate::core::queries::Queries;
    use crate::core::utils::bit_reverse_index;

    /// Default blowup factor used for tests.
    const LOG_BLOWUP_FACTOR: u32 = 2;

    #[test]
    fn fold_line_works() {
        const DEGREE: usize = 8;
        // Coefficients are bit-reversed.
        let even_coeffs: [BaseField; DEGREE / 2] = [1, 2, 1, 3].map(BaseField::from_u32_unchecked);
        let odd_coeffs: [BaseField; DEGREE / 2] = [3, 5, 4, 1].map(BaseField::from_u32_unchecked);
        let poly = LinePoly::new([even_coeffs, odd_coeffs].concat());
        let even_poly = LinePoly::new(even_coeffs.to_vec());
        let odd_poly = LinePoly::new(odd_coeffs.to_vec());
        let alpha = BaseField::from_u32_unchecked(19283);
        let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2()));
        let drp_domain = domain.double();
        let evals = poly.evaluate(domain).bit_reverse();
        let two = BaseField::from_u32_unchecked(2);

        let drp_evals = fold_line(&evals, alpha);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        let drp_evals = drp_evals.bit_reverse();
        for (i, (&drp_eval, x)) in zip(&*drp_evals, drp_domain).enumerate() {
            let f_e = even_poly.eval_at_point(x);
            let f_o = odd_poly.eval_at_point(x);
            assert_eq!(drp_eval, two * (f_e + alpha * f_o), "mismatch at {i}");
        }
    }

    #[test]
    fn fold_circle_to_line_works() {
        const LOG_DEGREE: u32 = 4;
        let circle_evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let num_folded_evals = circle_evaluation.domain.size() >> CIRCLE_TO_LINE_FOLD_STEP;
        let alpha = BaseField::one();
        let folded_domain = LineDomain::new(circle_evaluation.domain.half_coset);

        let mut folded_evaluation =
            LineEvaluation::new(folded_domain, vec![BaseField::zero(); num_folded_evals]);
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
        let config = FriConfig::new(2, LOG_EXPECTED_BLOWUP_FACTOR);
        let evaluation = polynomial_evaluation(6, LOG_INVALID_BLOWUP_FACTOR);

        FriProver::<BaseField, Blake3Hasher>::commit(config, vec![evaluation]);
    }

    #[test]
    #[should_panic = "not canonic"]
    fn committing_evaluation_from_invalid_domain_fails() {
        let invalid_domain = CircleDomain::new(Coset::new(CirclePointIndex::generator(), 3));
        assert!(!invalid_domain.is_canonic(), "must be an invalid domain");
        let evaluation = CircleEvaluation::new(invalid_domain, vec![QM31::one(); 1 << 4]);

        FriProver::<QM31, Blake3Hasher>::commit(FriConfig::new(2, 2), vec![evaluation]);
    }

    #[test]
    fn valid_proof_passes_verification() -> Result<(), VerificationError> {
        const LOG_DEGREE: u32 = 3;
        let config = FriConfig::new(1, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial.clone()]);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let proof = prover.decommit(&queries);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let column_bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let verifier = FriVerifier::commit(config, proof, vec![column_bound]).unwrap();

        verifier.decommit(&queries, vec![decommitment_value])
    }

    #[test]
    fn valid_mixed_degree_proof_passes_verification() -> Result<(), VerificationError> {
        const LOG_DEGREES: [u32; 3] = [6, 5, 4];
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomials = LOG_DEGREES.map(|log_d| polynomial_evaluation(log_d, LOG_BLOWUP_FACTOR));
        let log_domain_size = polynomials[0].domain.log_size();
        let queries = Queries::from_positions(vec![7, 70], log_domain_size);
        let decommitment_values = polynomials
            .iter()
            .map(|p| query_polynomial(p, &queries))
            .collect();
        let prover = FriProver::<BaseField, Blake3Hasher>::commit(config, polynomials.to_vec());
        let proof = prover.decommit(&queries);
        let column_bounds = LOG_DEGREES.map(CirclePolyDegreeBound::new);
        let verifier = FriVerifier::commit(config, proof, column_bounds.to_vec()).unwrap();

        verifier.decommit(&queries, decommitment_values)
    }

    #[test]
    fn proof_with_removed_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(6, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1], log_domain_size);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let proof = prover.decommit(&queries);
        let column_bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        // Set verifier's config to expect one extra layer than prover config.
        let mut invalid_config = config;
        invalid_config.log_last_layer_degree_bound -= 1;

        let verifier = FriVerifier::commit(invalid_config, proof, vec![column_bound]);

        assert!(matches!(
            verifier,
            Err(VerificationError::InvalidNumFriLayers)
        ));
    }

    #[test]
    fn proof_with_added_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1], log_domain_size);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let proof = prover.decommit(&queries);
        let column_bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        // Set verifier's config to expect one less layer than prover config.
        let mut invalid_config = config;
        invalid_config.log_last_layer_degree_bound += 1;

        let verifier = FriVerifier::commit(invalid_config, proof, vec![column_bound]);

        assert!(matches!(
            verifier,
            Err(VerificationError::InvalidNumFriLayers)
        ));
    }

    #[test]
    fn proof_with_invalid_inner_layer_evaluation_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let column_bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let mut proof = prover.decommit(&queries);
        // Remove an evaluation from the second layer's proof.
        proof.inner_layers[1].evals_subset.pop();
        let verifier = FriVerifier::commit(config, proof, vec![column_bound]).unwrap();

        let verification_result = verifier.decommit(&queries, vec![decommitment_value]);

        assert!(matches!(
            verification_result,
            Err(VerificationError::InnerLayerEvaluationsInvalid { layer: 1 })
        ));
    }

    #[test]
    fn proof_with_invalid_inner_layer_decommitment_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let column_bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let mut proof = prover.decommit(&queries);
        // Modify the committed values in the second layer.
        proof.inner_layers[1].evals_subset[0] += QM31::one();
        let verifier = FriVerifier::commit(config, proof, vec![column_bound]).unwrap();

        let verification_result = verifier.decommit(&queries, vec![decommitment_value]);

        assert!(matches!(
            verification_result,
            Err(VerificationError::InnerLayerCommitmentInvalid { layer: 1 })
        ));
    }

    #[test]
    fn proof_with_invalid_last_layer_degree_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        const LOG_MAX_LAST_LAYER_DEGREE: u32 = 2;
        let config = FriConfig::new(LOG_MAX_LAST_LAYER_DEGREE, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1, 7, 8], log_domain_size);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let column_bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let mut proof = prover.decommit(&queries);
        let bad_last_layer_coeffs = vec![QM31::one(); 1 << (LOG_MAX_LAST_LAYER_DEGREE + 1)];
        proof.last_layer_poly = LinePoly::new(bad_last_layer_coeffs);

        let verifier = FriVerifier::commit(config, proof, vec![column_bound]);

        assert!(matches!(
            verifier,
            Err(VerificationError::LastLayerDegreeInvalid)
        ));
    }

    #[test]
    fn proof_with_invalid_last_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1, 7, 8], log_domain_size);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let column_bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let mut proof = prover.decommit(&queries);
        // Compromise the last layer polynomial's first coefficient.
        proof.last_layer_poly[0] += QM31::one();
        let verifier = FriVerifier::commit(config, proof, vec![column_bound]).unwrap();

        let verification_result = verifier.decommit(&queries, vec![decommitment_value]);

        assert!(matches!(
            verification_result,
            Err(VerificationError::LastLayerEvaluationsInvalid)
        ));
    }

    #[test]
    #[should_panic]
    fn decommit_queries_on_invalid_domain_fails_verification() {
        const LOG_DEGREE: u32 = 3;
        let config = FriConfig::new(1, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial.clone()]);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let proof = prover.decommit(&queries);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let column_bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        let verifier = FriVerifier::commit(config, proof, vec![column_bound]).unwrap();
        // Simulate the verifier sampling queries on a smaller domain.
        let mut invalid_queries = queries.clone();
        invalid_queries.log_domain_size -= 1;

        let _ = verifier.decommit(&invalid_queries, vec![decommitment_value]);
    }

    /// Returns an evaluation of a random polynomial with degree `2^log_degree`.
    ///
    /// The evaluation domain size is `2^(log_degree + log_blowup_factor)`.
    fn polynomial_evaluation<F: ExtensionOf<BaseField>>(
        log_degree: u32,
        log_blowup_factor: u32,
    ) -> CircleEvaluation<F, BitReversedOrder> {
        let poly = CirclePoly::new(vec![F::one(); 1 << log_degree]);
        let coset = Coset::half_odds(log_degree + log_blowup_factor - 1);
        let domain = CircleDomain::new(coset);
        poly.evaluate(domain).bit_reverse()
    }

    /// Returns the log degree bound of a polynomial.
    fn log_degree_bound<F: ExtensionOf<BaseField>>(
        polynomial: LineEvaluation<F, BitReversedOrder>,
    ) -> u32 {
        let polynomial = polynomial.bit_reverse();
        let coeffs = polynomial.interpolate().into_ordered_coefficients();
        let degree = coeffs.into_iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        (degree + 1).ilog2()
    }

    // TODO: Remove after SubcircleDomain integration.
    fn query_polynomial<F: ExtensionOf<BaseField>>(
        polynomial: &CircleEvaluation<F, BitReversedOrder>,
        queries: &Queries,
    ) -> SparseCircleEvaluation<F> {
        let domain = polynomial.domain;
        let polynomial = polynomial.clone().bit_reverse();
        let oracle = EvalByEvaluation::new(CirclePointIndex::zero(), &polynomial);

        // Get queries on the polynomial
        let polynomial_queries = queries.fold(queries.log_domain_size - domain.log_size());

        let coset_evals = polynomial_queries
            .iter()
            .map(|&query| {
                let position = bit_reverse_index(query, domain.log_size());
                let p = domain.index_at(position);
                let coset_domain = CircleDomain::new(Coset::new(p, 0));
                let evals = coset_domain
                    .iter_indices()
                    .map(|p| oracle.get_at(p))
                    .collect();
                let coset_eval = CircleEvaluation::<F, NaturalOrder>::new(coset_domain, evals);
                coset_eval.bit_reverse()
            })
            .collect();

        SparseCircleEvaluation::new(coset_evals)
    }
}
