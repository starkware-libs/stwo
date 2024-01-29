use std::cmp::Reverse;
use std::fmt::Debug;
use std::iter::zip;
use std::ops::RangeInclusive;

use itertools::Itertools;
use thiserror::Error;

use super::circle::CirclePointIndex;
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

impl<F: ExtensionOf<BaseField>, H: Hasher> FriProver<F, H> {
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
        let mut evaluation = LineEvaluation::new(vec![F::zero(); layer_size]);

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

            let layer = FriLayer::new(&evaluation);

            // TODO(andrew): add merkle root to channel
            // TODO(ohad): Add back once IntoSlice implemented for Field.
            // let _merkle_root = layer.merkle_tree.root();
            // TODO(andrew): draw random alpha from channel
            let alpha = F::one();
            let folded_evaluation = fold_line(&evaluation, alpha);

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

        let domain = LineDomain::new(Coset::half_odds(evaluation.len().ilog2()));
        let evaluation = evaluation.bit_reverse();
        let mut coeffs = evaluation.interpolate(domain).into_ordered_coefficients();

        let max_num_coeffs = 1 << config.log_last_layer_degree_bound;
        let zeros = coeffs.split_off(max_num_coeffs);
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

impl<F: ExtensionOf<BaseField>, H: Hasher> FriVerifier<F, H> {
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
            if !layer.verify(&queries, &sparse_evaluation) {
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

impl<F: ExtensionOf<BaseField>, H: Hasher> FriLayerProof<F, H> {
    // TODO(andrew): implement and add docs
    // TODO(andrew): create FRI verification error type
    fn verify(&self, _queries: &Queries, _evaluation: &SparseLineEvaluation<F>) -> bool {
        todo!()
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
        const COSET_SIZE: usize = 1 << LOG_FOLDING_FACTOR;

        // Evals provided by the verifier.
        let mut verifier_evals = evals.iter().copied();

        // Evals stored in the proof.
        let mut proof_evals = self.evals_subset.iter().copied();

        let mut coset_evals = Vec::new();

        // Group queries by the coset they reside in.
        for coset_positions in queries.group_by(|a, b| a / COSET_SIZE == b / COSET_SIZE) {
            let mut coset_eval = [None; COSET_SIZE];

            // Insert the verifier's evals.
            for position in coset_positions {
                coset_eval[position % COSET_SIZE] = Some(verifier_evals.next().unwrap());
            }

            // Fill in the remaining evals using values stored in the proof.
            for eval in &mut coset_eval {
                if eval.is_none() {
                    *eval = Some(proof_evals.next()?);
                }
            }

            // Construct the domain.
            let coset_index = bit_reverse_index(coset_positions[0] / COSET_SIZE, domain.log_size());
            let domain_initial = domain.coset().index_at(coset_index);
            let domain = LineDomain::new(Coset::new(domain_initial, LOG_FOLDING_FACTOR));

            let coset_eval = LineEvaluation::new(domain, coset_eval.map(Option::unwrap).to_vec());
            coset_evals.push(coset_eval);
        }

        Some(SparseLineEvaluation::new(coset_evals))
    }
}

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): support different folding factors
struct FriLayer<F: ExtensionOf<BaseField>, H: Hasher> {
    /// Coset evaluations stored in column-major.
    subcircle_evals: [Vec<F>; 1 << LOG_FOLDING_FACTOR],
    _merkle_tree: MerkleTree<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriLayer<F, H> {
    fn new(evaluation: &LineEvaluation<F, BitReversedOrder>) -> Self {
        // TODO(andrew): With bit-reversed order coset evals are next to each other. Update.
        let (l, r) = evaluation.split_at(evaluation.len() / 2);
        let subcircle_evals = [l.to_vec(), r.to_vec()];
        // TODO(ohad): Add back once IntoSlice implemented for Field.
        // let merkle_tree = MerkleTree::commit(coset_evals.to_vec());
        #[allow(unreachable_code)]
        FriLayer {
            subcircle_evals,
            _merkle_tree: todo!(),
        }
    }

    /// Generates a decommitment of the subcircle evaluations at the specified positions.
    fn decommit(self, queries: &Queries) -> FriLayerProof<F, H> {
        const SUBCIRCLE_SIZE: usize = 1 << LOG_FOLDING_FACTOR;

        let mut evals_subset = Vec::new();

        // Group queries by the subcircle they reside in.
        for query_group in queries.group_by(|a, b| a / SUBCIRCLE_SIZE == b / SUBCIRCLE_SIZE) {
            let subcircle_index = query_group[0] / SUBCIRCLE_SIZE;
            let mut subcircle_queries = query_group.iter().map(|q| q % SUBCIRCLE_SIZE).peekable();

            for i in 0..SUBCIRCLE_SIZE {
                // Skip evals the verifier can calculate.
                if subcircle_queries.peek() == Some(&i) {
                    subcircle_queries.next();
                    continue;
                }

                let eval = self.subcircle_evals[i][subcircle_index];
                evals_subset.push(eval);
            }
        }

        // TODO(ohad): Add back once IntoSlice implemented for Field.
        // let position_set = positions.iter().copied().collect();
        // let decommitment = self.merkle_tree.generate_decommitment(position_set);
        // let commitment = self.merkle_tree.root();
        #[allow(unreachable_code)]
        FriLayerProof {
            evals_subset,
            decommitment: todo!(),
            commitment: todo!(),
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

    let folded_evals = evals
        .array_chunks()
        .enumerate()
        .map(|(i, &[f_x, f_neg_x])| {
            // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
            let x = domain.at(bit_reverse_index(i, domain.log_size()));

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
            let p = domain.at(bit_reverse_index(i, log_folded_domain_size));

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

    use crate::core::circle::Coset;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::ExtensionOf;
    use crate::core::fri::{fold_circle_into_line, fold_line, LOG_CIRCLE_TO_LINE_FOLDING_FACTOR};
    use crate::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};
    use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};
    use crate::core::poly::BitReversedOrder;

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
        let domain = LineDomain::new(Coset::half_odds(polynomial.len().ilog2()));
        let polynomial = polynomial.bit_reverse();
        let coeffs = polynomial.interpolate(domain).into_ordered_coefficients();
        let degree = coeffs.into_iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        (degree + 1).ilog2()
    }
}
