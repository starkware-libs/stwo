use std::array;
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
// TODO(andrew): support different folding factors
#[derive(Debug, Clone, Copy)]
pub struct FriConfig {
    log_blowup_factor: u32,
    log_last_layer_degree_bound: u32,
    // TODO(andrew): Add num_queries, folding_factors.
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
    inner_layers: Vec<FriLayer<F, H>>,
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
    pub fn commit(config: FriConfig, evals: Vec<CircleEvaluation<F, BitReversedOrder>>) -> Self {
        assert!(evals.is_sorted_by_key(|e| Reverse(e.len())), "not sorted");
        assert!(evals.iter().all(|e| e.domain.is_canonic()), "not canonic");
        let (inner_layers, last_layer_evaluation) = Self::commit_inner_layers(config, evals);
        let last_layer_poly = Self::commit_last_layer(config, last_layer_evaluation);
        Self {
            inner_layers,
            last_layer_poly,
        }
    }

    /// Builds and commits to the inner FRI layers (all layers except the last layer).
    ///
    /// `evals` must be provided in descending order by size.
    ///
    /// Returns all inner layers and the evaluation for the last layer.
    ///
    /// # Panics
    ///
    /// Panics if `evals` is empty or if an evaluation's domain is smaller than or equal to the last
    /// layer's domain.
    fn commit_inner_layers(
        config: FriConfig,
        evals: Vec<CircleEvaluation<F, BitReversedOrder>>,
    ) -> (Vec<FriLayer<F, H>>, LineEvaluation<F, BitReversedOrder>) {
        // Returns the length of the [LineEvaluation] a [CircleEvaluation] gets folded into.
        let folded_len = |e: &CircleEvaluation<_, _>| e.len() >> LOG_CIRCLE_TO_LINE_FOLDING_FACTOR;
        let mut evals = evals.into_iter().peekable();
        let mut layer_size = evals.peek().map(folded_len).expect("no evaluation");

        let domain = LineDomain::new(Coset::half_odds(layer_size.ilog2()));
        let mut evaluation = LineEvaluation::new(domain, vec![F::zero(); layer_size]);

        let mut layers = Vec::new();

        // Circle polynomials can all be folded with the same alpha.
        // TODO(andrew): draw random alpha from channel
        let circle_poly_alpha = F::one();

        while evaluation.len() > config.last_layer_domain_size() {
            // Check for any evaluations that should be combined.
            while evals.peek().map(folded_len) == Some(layer_size) {
                let circle_evaluation = evals.next().unwrap();
                fold_circle_into_line(&mut evaluation, &circle_evaluation, circle_poly_alpha);
            }

            let layer = FriLayer::new(evaluation);

            // TODO(andrew): add merkle root to channel
            // TODO(ohad): Add back once IntoSlice implemented for Field.
            // let _merkle_root = layer.merkle_tree.root();
            // TODO(andrew): draw random alpha from channel
            let alpha = F::one();
            let folded_evaluation = fold_line(&layer.evaluation, alpha);

            evaluation = folded_evaluation;
            layer_size >>= LOG_FOLDING_FACTOR;
            layers.push(layer);
        }

        assert!(evals.next().is_none());
        (layers, evaluation)
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

    pub fn decommit(self, queries: &Queries) -> FriProof<F, H> {
        let last_layer_poly = self.last_layer_poly;
        let inner_layers = self
            .inner_layers
            .into_iter()
            .scan(queries.clone(), |queries, layer| {
                let layer_proof = layer.decommit(queries);
                *queries = queries.fold(LOG_FOLDING_FACTOR);
                Some(layer_proof)
            })
            .collect();
        FriProof {
            inner_layers,
            last_layer_poly,
        }
    }
}

pub struct FriVerifier<F: ExtensionOf<BaseField>, H: Hasher> {
    /// Alpha used to fold all circle polynomials to univariate polynomials.
    circle_poly_alpha: F,
    /// The list of degree bounds of all committed circle polynomials.
    column_degree_bounds: Vec<LogCirclePolyDegreeBound>,
    config: FriConfig,
    inner_layer_alphas: Vec<F>,
    proof: FriProof<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher<NativeType = u8>> FriVerifier<F, H> {
    /// Verifies the commitment stage of FRI.
    ///
    /// `column_degree_bounds` should be a list of degree bounds of all committed circle
    /// polynomials.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof contains an invalid number of FRI layers.
    /// * The degree of the last layer polynomial is too high.
    ///
    /// # Panics
    ///
    /// Panics if there are no degree bounds or if one is less than or equal to the last
    /// layer's degree bound.
    pub fn commit(
        config: FriConfig,
        proof: FriProof<F, H>,
        mut column_degree_bounds: Vec<LogCirclePolyDegreeBound>,
    ) -> Result<Self, VerificationError> {
        // Sort descending order
        column_degree_bounds.sort_by_key(|d| Reverse(*d));
        let log_first_layer_degree_bound = folded_circle_poly_degree(&column_degree_bounds[0]);
        let mut log_degree = log_first_layer_degree_bound;

        // Circle polynomials can all be folded with the same alpha.
        // TODO(andrew): Draw alpha from channel.
        let circle_poly_alpha = F::one();

        let mut inner_layer_alphas = Vec::new();

        for _ in &proof.inner_layers {
            // TODO(andrew): Seed channel with commitment.
            // TODO(andrew): Draw alpha from channel.
            let alpha = F::one();
            inner_layer_alphas.push(alpha);
            log_degree = log_degree.wrapping_sub(LOG_FOLDING_FACTOR);
        }

        if log_degree != config.log_last_layer_degree_bound {
            return Err(VerificationError::InvalidNumFriLayers);
        }

        let last_layer_degree_bound = 1 << config.log_last_layer_degree_bound;
        if proof.last_layer_poly.len() > last_layer_degree_bound {
            return Err(VerificationError::LastLayerDegreeInvalid);
        }

        Ok(Self {
            circle_poly_alpha,
            config,
            inner_layer_alphas,
            column_degree_bounds,
            proof,
        })
    }

    /// Verifies the decommitment stage of FRI.
    ///
    /// The decommitment values need to be provided in the same order as their commitment.
    ///
    /// # Panics
    ///
    /// Panics if there aren't the same number of decommited values as degree bounds.
    // TODO(andrew): Finish docs.
    pub fn decommit(
        self,
        queries: &Queries,
        decommited_values: Vec<SparseCircleEvaluation<F>>,
    ) -> Result<(), VerificationError> {
        assert_eq!(decommited_values.len(), self.column_degree_bounds.len());

        let (last_layer_domain, last_layer_queries, last_layer_evals) =
            self.decommit_inner_layers(queries, decommited_values)?;

        self.decommit_last_layer(last_layer_domain, last_layer_queries, last_layer_evals)
    }

    /// Verifies all inner layer decommitments.
    ///
    /// Returns the domain, query positions and evaluations needed for verifying the last FRI
    /// layer. Output is of the form: `(domain, query_positions, evaluations)`.
    fn decommit_inner_layers(
        &self,
        queries: &Queries,
        decommited_values: Vec<SparseCircleEvaluation<F>>,
    ) -> Result<(LineDomain, Queries, Vec<F>), VerificationError> {
        let log_blowup_factor = self.config.log_blowup_factor;

        let circle_poly_alpha = self.circle_poly_alpha;
        let circle_poly_alpha_sq = circle_poly_alpha * circle_poly_alpha;

        let mut queries = queries.clone();
        let mut alphas = self.inner_layer_alphas.iter().copied();
        let mut decommited_values = decommited_values.into_iter();
        let mut degrees = self.column_degree_bounds.iter().copied().peekable();
        let mut log_degree = folded_circle_poly_degree(&self.column_degree_bounds[0]);
        let mut evals = vec![F::zero(); queries.len()];
        let mut domain = LineDomain::new(Coset::half_odds(log_degree + log_blowup_factor));

        for (i, layer) in self.proof.inner_layers.iter().enumerate() {
            // Fold and combine circle polynomial evaluations.
            while degrees.peek().map(folded_circle_poly_degree) == Some(log_degree) {
                let sparse_evaluation = decommited_values.next().unwrap();
                let folded_evals = sparse_evaluation.fold(circle_poly_alpha);
                assert_eq!(folded_evals.len(), evals.len());

                for (eval, folded_eval) in zip(&mut evals, folded_evals) {
                    *eval = *eval * circle_poly_alpha_sq + folded_eval;
                }

                degrees.next();
            }

            // Extract the values needed to fold.
            let sparse_evaluation = layer
                .extract_evaluation(domain, &queries, &evals)
                .ok_or(VerificationError::InnerLayerEvaluationsInvalid { layer: i })?;

            // Verify the decommitment.
            if !layer.decommit(&queries, &sparse_evaluation) {
                return Err(VerificationError::InnerLayerCommitmentInvalid { layer: i });
            }

            let alpha = alphas.next().unwrap();
            let folded_evals = sparse_evaluation.fold(alpha);

            // Prepare the next layer.
            evals = folded_evals;
            queries = queries.fold(LOG_FOLDING_FACTOR);
            log_degree -= LOG_FOLDING_FACTOR;
            domain = domain.double();
        }

        // Check all values have been consumed.
        assert!(alphas.is_empty());
        assert!(degrees.is_empty());
        assert!(decommited_values.is_empty());

        Ok((domain, queries, evals))
    }

    /// Verifies the last layer.
    fn decommit_last_layer(
        self,
        domain: LineDomain,
        queries: Queries,
        evals: Vec<F>,
    ) -> Result<(), VerificationError> {
        for (&query, eval) in zip(&*queries, evals) {
            let x = domain.at(bit_reverse_index(query, domain.log_size()));
            if eval != self.proof.last_layer_poly.eval_at_point(x.into()) {
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

/// Log degree bound of a circle polynomial.
type LogCirclePolyDegreeBound = u32;

/// Log degree bound of a univariate (line) polynomial.
pub(crate) type LogLinePolyDegreeBound = u32;

/// Maps a circle polynomial's degree bound to the degree bound of the line polynomial it gets
/// folded into.
fn folded_circle_poly_degree(degree: &LogCirclePolyDegreeBound) -> LogLinePolyDegreeBound {
    degree - LOG_CIRCLE_TO_LINE_FOLDING_FACTOR
}

/// A FRI proof.
pub struct FriProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub inner_layers: Vec<FriLayerProof<F, H>>,
    pub last_layer_poly: LinePoly<F>,
}

/// Folding factor for univariate polynomials.
// TODO(andrew): Support multiple folding factors.
const LOG_FOLDING_FACTOR: u32 = 1;

/// Folding factor when folding a circle polynomial to univariate polynomial.
const LOG_CIRCLE_TO_LINE_FOLDING_FACTOR: u32 = 1;

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
///
/// The subset corresponds to the set of evaluations needed by a FRI verifier.
// TODO(andrew): Consider adding docs here explaining the idea of splitting the layer's evaluations
// into evaluations on multiple smaller cosets. Also perhaps coset isn't the best name because it
// clashes with [Coset].
pub struct FriLayerProof<F: ExtensionOf<BaseField>, H: Hasher> {
    /// Subset of all subcircle evaluations.
    ///
    /// The subset stored corresponds to the set of evaluations the verifier doesn't have but needs
    /// to verify the decommitment.
    pub evals_subset: Vec<F>,
    pub decommitment: MerkleDecommitment<F, H>,
    pub commitment: H::Hash,
}

impl<F: ExtensionOf<BaseField>, H: Hasher<NativeType = u8>> FriLayerProof<F, H> {
    /// Verifies the layer's decommitment.
    ///
    /// `queries` should be the queries into this layer. `evaluation` should be the evaluation
    /// returned by [Self::extract_evaluation].
    // TODO(andrew): implement and add docs
    // TODO(andrew): create FRI verification error type
    fn decommit(&self, queries: &Queries, evaluation: &SparseLineEvaluation<F>) -> bool {
        // All evals flattened.
        let evals = evaluation
            .coset_evals
            .iter()
            .flat_map(|e| e.iter())
            .copied()
            .collect::<Vec<F>>();

        let mut decommitment_values = Vec::new();

        // TODO: Remove leaf values from the decommitment.
        for leaf in self.decommitment.values() {
            // Ensure each leaf is a single value.
            if let [leaf] = *leaf {
                decommitment_values.push(leaf);
            } else {
                return false;
            }
        }

        if decommitment_values != evals {
            println!("WHAAAT??: {:?}", decommitment_values);
            println!("WHAAAT??: {:?}", evals);
            return false;
        }

        // Positions of all the flattened evals.
        let eval_positions = queries
            .fold(LOG_FOLDING_FACTOR)
            .iter()
            .flat_map(|folded_query| {
                const COSET_SIZE: usize = 1 << LOG_FOLDING_FACTOR;
                let coset_start = folded_query * COSET_SIZE;
                let coset_end = coset_start + COSET_SIZE;
                coset_start..coset_end
            })
            .collect::<Vec<usize>>();
        assert_eq!(eval_positions.len(), evals.len());

        self.decommitment.verify(self.commitment, &eval_positions)
    }

    /// Returns the coset evals needed for decommitment.
    ///
    /// `evals` must be the verifier's set of evals at the corresponding queries. `domain` should
    /// be the domain of the layer.
    ///
    /// # Errors
    ///
    /// Returns [None] if the proof contains an invalid number of evaluations.
    ///
    /// # Panics
    ///
    /// Panics if the number of queries doesn't match the number of evals.
    fn extract_evaluation(
        &self,
        domain: LineDomain,
        queries: &Queries,
        evals: &[F],
    ) -> Option<SparseLineEvaluation<F>> {
        const SUBCIRCLE_SIZE: usize = 1 << LOG_FOLDING_FACTOR;

        // Evals provided by the verifier.
        let mut verifier_evals = evals.iter().copied();

        // Evals stored in the proof.
        let mut proof_evals = self.evals_subset.iter().copied();

        let mut subcircle_evals = Vec::new();

        // Group queries by the subcircle they reside in.
        for query_group in queries.group_by(|a, b| a / SUBCIRCLE_SIZE == b / SUBCIRCLE_SIZE) {
            let subcircle_index = query_group[0] / SUBCIRCLE_SIZE;

            // Construct the subcircle domain
            let initial_index = bit_reverse_index(subcircle_index, domain.log_size());
            let initial = domain.coset().index_at(initial_index);
            let domain = LineDomain::new(Coset::new(initial, LOG_FOLDING_FACTOR));

            let mut subcircle_eval = LineEvaluation::new(domain, vec![F::zero(); SUBCIRCLE_SIZE]);
            let mut subcircle_queries = query_group.iter().map(|q| q % SUBCIRCLE_SIZE).peekable();

            // Insert the subcircle evals.
            for i in 0..SUBCIRCLE_SIZE {
                if subcircle_queries.next_if_eq(&i).is_some() {
                    subcircle_eval[i] = verifier_evals.next().unwrap();
                } else {
                    subcircle_eval[i] = verifier_evals.next()?;
                }
            }

            subcircle_evals.push(subcircle_eval);
        }

        Some(SparseLineEvaluation::new(subcircle_evals))
    }
}

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): support different folding factors
struct FriLayer<F: ExtensionOf<BaseField>, H: Hasher> {
    evaluation: LineEvaluation<F, BitReversedOrder>,
    merkle_tree: MerkleTree<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher<NativeType = u8>> FriLayer<F, H> {
    fn new(evaluation: LineEvaluation<F, BitReversedOrder>) -> Self {
        // TODO: Commit on slice.
        let merkle_tree = MerkleTree::commit(vec![evaluation.to_vec()]);
        #[allow(unreachable_code)]
        FriLayer {
            evaluation,
            merkle_tree,
        }
    }

    /// Generates a decommitment of the subcircle evaluations at the specified positions.
    fn decommit(self, queries: &Queries) -> FriLayerProof<F, H> {
        const SUBCIRCLE_SIZE: usize = 1 << LOG_FOLDING_FACTOR;

        let mut decommit_positions = Vec::new();
        let mut evals_subset = Vec::new();

        // Group queries by the subcircle they reside in.
        for query_group in queries.group_by(|a, b| a / SUBCIRCLE_SIZE == b / SUBCIRCLE_SIZE) {
            let subcircle_index = query_group[0] / SUBCIRCLE_SIZE;
            let mut subcircle_queries = query_group.iter().peekable();

            for i in 0..SUBCIRCLE_SIZE {
                // Add decommitment position.
                let eval_position = subcircle_index * SUBCIRCLE_SIZE + i;
                decommit_positions.push(eval_position);

                // Skip evals the verifier can calculate.
                if subcircle_queries.next_if_eq(&&eval_position).is_some() {
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
    coset_evals: Vec<CircleEvaluation<F, BitReversedOrder>>,
}

impl<F: ExtensionOf<BaseField>> SparseCircleEvaluation<F> {
    /// # Panics
    ///
    /// Panics if the coset sizes aren't the same as the folding factor.
    pub fn new(coset_evals: Vec<CircleEvaluation<F, BitReversedOrder>>) -> Self {
        let folding_factor = 1 << LOG_FOLDING_FACTOR;
        assert!(coset_evals.iter().all(|e| e.len() == folding_factor));
        Self { coset_evals }
    }

    fn fold(self, alpha: F) -> Vec<F> {
        self.coset_evals
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
    coset_evals: Vec<LineEvaluation<F, BitReversedOrder>>,
}

impl<F: ExtensionOf<BaseField>> SparseLineEvaluation<F> {
    /// # Panics
    ///
    /// Panics if the coset sizes aren't the same as the folding factor.
    fn new(coset_evals: Vec<LineEvaluation<F, BitReversedOrder>>) -> Self {
        let folding_factor = 1 << LOG_FOLDING_FACTOR;
        assert!(coset_evals.iter().all(|e| e.len() == folding_factor));
        Self { coset_evals }
    }

    fn fold(self, alpha: F) -> Vec<F> {
        self.coset_evals
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
    let log_folded_domain_size = domain.log_size() - LOG_FOLDING_FACTOR;

    let folded_evals = evals
        .array_chunks()
        .enumerate()
        .map(|(i, &[f_x, f_neg_x])| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let x = domain.at(bit_reverse_index(i, log_folded_domain_size));

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
    assert_eq!(src.len() >> LOG_CIRCLE_TO_LINE_FOLDING_FACTOR, dst.len());

    let domain = src.domain;
    let log_folded_domain_size = domain.log_size() - LOG_CIRCLE_TO_LINE_FOLDING_FACTOR;
    let alpha_sq = alpha * alpha;

    zip(&mut **dst, src.array_chunks())
        .enumerate()
        .for_each(|(i, (dst, &[f_p, f_neg_p]))| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            // TODO(andrew): Remove conditional and update bit_reverse_index to handle log_size = 0.
            let p = if log_folded_domain_size == 0 {
                domain.at(0)
            } else {
                domain.at(bit_reverse_index(i, log_folded_domain_size))
            };

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
        fold_circle_into_line, fold_line, FriConfig, FriProver, FriVerifier,
        LOG_CIRCLE_TO_LINE_FOLDING_FACTOR,
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
        let num_folded_evals = circle_evaluation.domain.size() >> LOG_CIRCLE_TO_LINE_FOLDING_FACTOR;
        let alpha = BaseField::one();
        let folded_domain = LineDomain::new(circle_evaluation.domain.half_coset);

        let mut folded_evaluation =
            LineEvaluation::new(folded_domain, vec![BaseField::zero(); num_folded_evals]);
        fold_circle_into_line(&mut folded_evaluation, &circle_evaluation, alpha);

        assert_eq!(
            log_degree_bound(folded_evaluation),
            LOG_DEGREE - LOG_CIRCLE_TO_LINE_FOLDING_FACTOR
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
    #[ignore = "verification issues"]
    fn valid_fri_proof_passes_verification() -> Result<(), VerificationError> {
        const LOG_DEGREE: u32 = 3;
        let config = FriConfig::new(1, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial.clone()]);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let proof = prover.decommit(&queries);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let verifier = FriVerifier::commit(config, proof, vec![LOG_DEGREE]).unwrap();

        verifier.decommit(&queries, vec![decommitment_value])
    }

    #[test]
    #[ignore = "verification issues"]
    fn mixed_degree_fri_proof_passes_verification() -> Result<(), VerificationError> {
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
        let verifier = FriVerifier::commit(config, proof, LOG_DEGREES.to_vec()).unwrap();

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
        // Set verifier's config to expect one extra layer than prover config.
        let mut invalid_config = config;
        invalid_config.log_last_layer_degree_bound -= 1;

        let verifier = FriVerifier::commit(invalid_config, proof, vec![LOG_DEGREE]);

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
        // Set verifier's config to expect one less layer than prover config.
        let mut invalid_config = config;
        invalid_config.log_last_layer_degree_bound += 1;

        let verifier = FriVerifier::commit(invalid_config, proof, vec![LOG_DEGREE]);

        assert!(matches!(
            verifier,
            Err(VerificationError::InvalidNumFriLayers)
        ));
    }

    #[test]
    #[ignore = "verification issues"]
    fn proof_with_invalid_inner_layer_evaluation_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let mut proof = prover.decommit(&queries);
        // Remove an evaluation from the second layer's proof.
        proof.inner_layers[1].evals_subset.pop();
        let verifier = FriVerifier::commit(config, proof, vec![LOG_DEGREE]).unwrap();

        let verification_result = verifier.decommit(&queries, vec![decommitment_value]);

        assert!(matches!(
            verification_result,
            Err(VerificationError::InnerLayerEvaluationsInvalid { layer: 1 })
        ));
    }

    #[test]
    #[ignore = "verification incomplete"]
    fn proof_with_invalid_inner_layer_decommitment_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![5], log_domain_size);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let mut proof = prover.decommit(&queries);
        // Modify the committed values in the second layer.
        proof.inner_layers[1].evals_subset[0] += QM31::one();
        let verifier = FriVerifier::commit(config, proof, vec![LOG_DEGREE]).unwrap();

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
        let mut proof = prover.decommit(&queries);
        let bad_last_layer_coeffs = vec![QM31::one(); 1 << (LOG_MAX_LAST_LAYER_DEGREE + 1)];
        proof.last_layer_poly = LinePoly::new(bad_last_layer_coeffs);

        let verifier = FriVerifier::commit(config, proof, vec![LOG_DEGREE]);

        assert!(matches!(
            verifier,
            Err(VerificationError::LastLayerDegreeInvalid)
        ));
    }

    #[test]
    #[ignore = "verification issues"]
    fn proof_with_invalid_last_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let log_domain_size = polynomial.domain.log_size();
        let queries = Queries::from_positions(vec![1, 7, 8], log_domain_size);
        let decommitment_value = query_polynomial(&polynomial, &queries);
        let prover = FriProver::<QM31, Blake3Hasher>::commit(config, vec![polynomial]);
        let mut proof = prover.decommit(&queries);
        // Compromise the last layer polynomial's first coefficient.
        proof.last_layer_poly[0] += QM31::one();
        let verifier = FriVerifier::commit(config, proof, vec![LOG_DEGREE]).unwrap();

        let verification_result = verifier.decommit(&queries, vec![decommitment_value]);

        assert!(matches!(
            verification_result,
            Err(VerificationError::LastLayerEvaluationsInvalid)
        ));
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

        let coset_evals = queries
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
