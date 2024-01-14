// TODO(andrew): File getting large consider splitting components across multiple files.
use core::array;
use std::fmt::Debug;
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::RangeInclusive;

use thiserror::Error;

use super::constraints::PolyOracle;
use super::fields::m31::BaseField;
use super::fields::{ExtensionOf, Field};
use super::poly::circle::CircleEvaluation;
use super::poly::line::{LineEvaluation, LinePoly};
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::poly::line::LineDomain;

/// FRI proof config
// TODO(andrew): support different folding factors
#[derive(Debug, Clone, Copy)]
pub struct FriConfig {
    log_blowup_factor: u32,
    log_last_layer_degree_bound: u32,
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
///
/// `Phase` is used to enforce the commitment phase is done before the query phase.
pub struct FriProver<F: ExtensionOf<BaseField>, H: Hasher, Phase = CommitmentPhase> {
    config: FriConfig,
    inner_layers: Vec<FriLayer<F, H>>,
    last_layer_poly: Option<LinePoly<F>>,
    _phase: PhantomData<Phase>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriProver<F, H, CommitmentPhase> {
    /// Creates a new FRI prover.
    pub fn new(config: FriConfig) -> Self {
        Self {
            config,
            inner_layers: Vec::new(),
            last_layer_poly: None,
            _phase: PhantomData,
        }
    }

    /// Commits to multiple [CircleEvaluation]s.
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
    /// * `evals` is empty or not sorted in ascending order by evaluation domain size.
    /// * A circle evaluation is not of a sufficiently low degree polynomial.
    /// * A circle evaluation's domain is smaller than the last layer.
    /// * A circle evaluation's domain is not canonic circle domain.
    pub fn commit(mut self, evals: Vec<CircleEvaluation<F>>) -> FriProver<F, H, QueryPhase> {
        assert!(evals.is_sorted_by_key(|e| e.len()), "not sorted");
        assert!(evals.iter().all(|e| e.domain.is_canonic()), "not canonic");
        let last_layer_evaluation = self.commit_inner_layers(evals);
        self.commit_last_layer(last_layer_evaluation);
        FriProver {
            config: self.config,
            inner_layers: self.inner_layers,
            last_layer_poly: self.last_layer_poly,
            _phase: PhantomData,
        }
    }

    /// Builds and commits to the inner FRI layers (all layers except the last layer).
    ///
    /// Returns the evaluation for the last layer.
    ///
    /// # Panics
    ///
    /// Panics if `evals` is empty or if an evaluation's domain is smaller than or equal to the last
    /// layer's domain.
    fn commit_inner_layers(&mut self, mut evals: Vec<CircleEvaluation<F>>) -> LineEvaluation<F> {
        // Returns the length of the [LineEvaluation] a [CircleEvaluation] gets folded into.
        let folded_len = |e: &CircleEvaluation<F>| e.len() >> LOG_CIRCLE_TO_LINE_FOLDING_FACTOR;
        let mut layer_size = evals.last().map(folded_len).expect("no evaluation");
        let mut evaluation: Option<LineEvaluation<F>> = None;

        while layer_size > self.config.last_layer_domain_size() {
            // Check for any evaluations that should be combined.
            while evals.last().map(folded_len) == Some(layer_size) {
                // TODO(andrew): draw random alpha from channel
                // TODO(andrew): Use powers of alpha instead of drawing for each circle polynomial.
                let alpha = F::one();
                let folded_evaluation = fold_circle_to_line(&evals.pop().unwrap(), alpha);
                assert_eq!(folded_evaluation.len(), layer_size);

                match evaluation.as_deref_mut() {
                    Some(evaluation) => zip(evaluation, folded_evaluation)
                        .for_each(|(eval, folded_eval)| *eval += alpha * folded_eval),
                    None => evaluation = Some(folded_evaluation),
                }
            }

            let layer = FriLayer::new(evaluation.as_ref().unwrap());

            // TODO(andrew): add merkle root to channel
            // TODO(ohad): Add back once IntoSlice implemented for Field.
            // let _merkle_root = layer.merkle_tree.root();
            // TODO(andrew): draw random alpha from channel
            let alpha = F::one();
            let folded_evaluation = fold_line(evaluation.as_ref().unwrap(), alpha);

            evaluation = Some(folded_evaluation);
            layer_size >>= LOG_FOLDING_FACTOR;
            self.inner_layers.push(layer)
        }

        assert!(evals.is_empty());
        evaluation.unwrap()
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
    fn commit_last_layer(&mut self, evaluation: LineEvaluation<F>) {
        assert_eq!(evaluation.len(), self.config.last_layer_domain_size());
        let max_num_coeffs = evaluation.len() >> self.config.log_blowup_factor;
        let domain = LineDomain::new(Coset::half_odds(evaluation.len().ilog2()));
        let mut coeffs = evaluation.interpolate(domain).into_ordered_coefficients();
        let zeros = coeffs.split_off(max_num_coeffs);
        assert!(zeros.iter().all(F::is_zero), "invalid degree");
        self.last_layer_poly = Some(LinePoly::from_ordered_coefficients(coeffs));
        // TODO(andrew): seed channel with last layer polynomial's coefficients.
    }
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriProver<F, H, QueryPhase> {
    pub fn into_proof(self, query_positions: &[usize]) -> FriProof<F, H> {
        let last_layer_poly = self.last_layer_poly.unwrap();
        let inner_layers = self
            .inner_layers
            .into_iter()
            .scan(query_positions.to_vec(), |positions, layer| {
                let num_layer_cosets = layer.coset_evals[0].len();
                let folded_positions = fold_positions(positions, num_layer_cosets);
                let layer_proof = layer.into_proof(positions, &folded_positions);
                *positions = folded_positions;
                Some(layer_proof)
            })
            .collect();
        FriProof {
            inner_layers,
            last_layer_poly,
        }
    }
}

/// Commitment phase for [FriProver].
pub struct CommitmentPhase;

/// Query phase for [FriProver].
pub struct QueryPhase;

pub struct FriVerifier<F: ExtensionOf<BaseField>, H: Hasher> {
    /// Alphas used for folding.
    alphas: Vec<F>,
    config: FriConfig,
    /// The list of degree bounds of all committed circle polynomials.
    poly_degree_bounds: Vec<LogCirclePolyDegreeBound>,
    proof: FriProof<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriVerifier<F, H> {
    /// Creates a new FRI verifier.
    ///
    /// This verifier can verify multiple polynomials, with different degrees, are each low degree.
    /// `degree_bounds` should be a list of degree bounds of all committed circle polynomials.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof contains an invalid number of FRI layers.
    /// * The last layer polynomial's degree is too high.
    ///
    /// # Panics
    ///
    /// Panics if there are no degree bounds or if a degree bound is less than the last layer.
    pub fn new(
        config: FriConfig,
        proof: FriProof<F, H>,
        mut poly_degree_bounds: Vec<LogCirclePolyDegreeBound>,
    ) -> Result<Self, VerificationError> {
        poly_degree_bounds.sort();
        let mut alphas = Vec::new();
        let mut degree_bounds = poly_degree_bounds.clone();
        let mut log_degree = degree_bounds.last().map(folded_circle_poly_degree).unwrap();

        for _ in &proof.inner_layers {
            // Draw alphas for the evaluations that will be combined into this layer.
            while degree_bounds.last().map(folded_circle_poly_degree) == Some(log_degree) {
                degree_bounds.pop();
                // TODO(andrew): draw random alpha from channel
                let alpha = F::one();
                alphas.push(alpha);
            }

            // TODO(andrew): Seed channel with commitment.
            // TODO(andrew): Draw alpha from channel.
            let alpha = F::one();
            alphas.push(alpha);

            log_degree = log_degree.wrapping_sub(LOG_FOLDING_FACTOR);
        }

        // Check there aren't an invalid number of layer proofs.
        if log_degree != config.log_last_layer_degree_bound {
            return Err(VerificationError::InvalidNumFriLayers);
        }

        // Ensure there aren't any circle polynomials left out.
        assert!(degree_bounds.is_empty());

        // Check the degree of the last layer's polynomial for the last layer.
        let last_layer_degree_bound = 1 << config.log_last_layer_degree_bound;
        if proof.last_layer_poly.len() > last_layer_degree_bound {
            return Err(VerificationError::LastLayerDegreeInvalid);
        }

        Ok(Self {
            config,
            alphas,
            poly_degree_bounds,
            proof,
        })
    }

    /// Verifies the FRI commitment.
    ///
    /// The polynomial oracles need to be provided in the same order as the commitment.
    pub fn verify(
        self,
        query_positions: &[usize],
        poly_oracles: &[impl PolyOracle<F>],
    ) -> Result<(), VerificationError> {
        if poly_oracles.len() != self.poly_degree_bounds.len() {
            return Err(VerificationError::InvalidNumPolynomials {
                expected: self.poly_degree_bounds.len(),
                given: poly_oracles.len(),
            });
        }

        // Obtain the evaluations needed for verification.
        let sparse_evaluations = query_oracles(
            self.config.log_blowup_factor,
            query_positions,
            &self.poly_degree_bounds,
            poly_oracles,
        );

        let (last_layer_domain, last_layer_positions, last_layer_evals) =
            self.verify_inner_layers(query_positions.to_vec(), sparse_evaluations)?;
        self.verify_last_layer(last_layer_domain, last_layer_positions, last_layer_evals)
    }

    /// Verifies all layers except the last layer.
    ///
    /// Returns the domain, query positions and evaluations needed for verifying the last FRI
    /// layer. Output is of the form: `(domain, query_positions, evaluations)`.
    // TODO: Separate into two functions. 1st to generate the oracle queries and 2nd to get the
    // answers.
    fn verify_inner_layers(
        &self,
        mut positions: Vec<usize>,
        mut sparse_evaluations: Vec<SparseCircleEvaluation<F>>,
    ) -> Result<(LineDomain, Vec<usize>, Vec<F>), VerificationError> {
        let log_blowup_factor = self.config.log_blowup_factor;

        let mut alphas = self.alphas.iter().copied();
        let mut degrees = self.poly_degree_bounds.clone();
        let mut log_degree = degrees.last().map(folded_circle_poly_degree).unwrap();
        let mut evals: Option<Vec<F>> = None;
        let mut domain = LineDomain::new(Coset::half_odds(log_degree + log_blowup_factor));

        for (i, layer) in self.proof.inner_layers.iter().enumerate() {
            let folded_size = domain.size() >> LOG_FOLDING_FACTOR;
            let folded_positions = fold_positions(&positions, folded_size);

            // Fold and combine circle polynomial evaluations.
            while degrees.last().map(folded_circle_poly_degree) == Some(log_degree) {
                degrees.pop();
                let alpha = alphas.next().unwrap();
                let sparse_evaluation = sparse_evaluations.pop().unwrap();
                let folded_evals = sparse_evaluation.fold(alpha);
                assert_eq!(folded_evals.len(), folded_positions.len());

                match evals.as_mut() {
                    Some(evals) => zip(evals, folded_evals)
                        .for_each(|(eval, folded_eval)| *eval += alpha * folded_eval),
                    None => evals = Some(folded_evals),
                }
            }

            // Extract the values needed to fold.
            let coset_evals = layer
                .coset_evals(
                    &positions,
                    &folded_positions,
                    domain.size(),
                    evals.as_ref().unwrap(),
                )
                .ok_or(VerificationError::InnerLayerEvaluationsInvalid { layer: i })?;

            // Verify the decommitment.
            if !layer.verify(&folded_positions, &coset_evals) {
                return Err(VerificationError::InnerLayerCommitmentInvalid { layer: i });
            }

            let alpha = alphas.next().unwrap();
            let folded_evals = layer.fold_evals(domain, &folded_positions, &coset_evals, alpha);

            // Prepare the next layer.
            evals = Some(folded_evals);
            positions = folded_positions;
            log_degree -= LOG_FOLDING_FACTOR;
            domain = domain.double();
        }

        // Check all values have been consumed.
        assert!(alphas.next().is_none());
        assert!(degrees.is_empty());
        assert!(sparse_evaluations.is_empty());

        Ok((domain, positions, evals.unwrap()))
    }

    /// Verifies the last layer.
    fn verify_last_layer(
        self,
        domain: LineDomain,
        positions: Vec<usize>,
        evals: Vec<F>,
    ) -> Result<(), VerificationError> {
        let last_layer_poly = self.proof.last_layer_poly;
        for (position, eval) in zip(positions, evals) {
            let x = domain.at(position);
            if eval != last_layer_poly.eval_at_point(x.into()) {
                return Err(VerificationError::LastLayerEvaluationsInvalid);
            }
        }
        Ok(())
    }
}

/// # Panics
///
/// Panics if there is a different amount of oracles and degree bounds.
pub(crate) fn query_oracles<F: ExtensionOf<BaseField>>(
    log_blowup_factor: u32,
    positions: &[usize],
    log_poly_degree_bounds: &[u32],
    poly_oracles: &[impl PolyOracle<F>],
) -> Vec<SparseCircleEvaluation<F>> {
    assert_eq!(poly_oracles.len(), log_poly_degree_bounds.len());
    assert!(log_poly_degree_bounds.is_sorted());
    assert!(positions.is_sorted());

    zip(log_poly_degree_bounds, poly_oracles)
        .map(|(log_degree_bound, oracle)| {
            let domain = CanonicCoset::new(log_degree_bound + log_blowup_factor).circle_domain();
            let folded_size = domain.size() >> LOG_CIRCLE_TO_LINE_FOLDING_FACTOR;
            let folded_positions = fold_positions(positions, folded_size);
            SparseCircleEvaluation::new(domain, oracle, &folded_positions)
        })
        .collect()
}

/// Holds a foldable subset of circle polynomial evaluations.
pub(crate) struct SparseCircleEvaluation<F: ExtensionOf<BaseField>> {
    coset_evals: Vec<CircleEvaluation<F>>,
}

impl<F: ExtensionOf<BaseField>> SparseCircleEvaluation<F> {
    fn new(domain: CircleDomain, oracle: &impl PolyOracle<F>, positions: &[usize]) -> Self {
        Self {
            coset_evals: positions
                .iter()
                .map(|position| {
                    let p = domain.index_at(*position);
                    let coset_domain = CircleDomain::new(Coset::new(p, 0));
                    let coset_evals = domain.iter_indices().map(|p| oracle.get_at(p)).collect();
                    CircleEvaluation::new(coset_domain, coset_evals)
                })
                .collect(),
        }
    }

    fn fold(self, alpha: F) -> Vec<F> {
        self.coset_evals
            .into_iter()
            .map(|e| fold_circle_to_line(&e, alpha)[0])
            .collect()
    }
}

/// Log degree bound of a circle polynomial.
pub(crate) type LogCirclePolyDegreeBound = u32;

/// Log degree bound of a univariate (line) polynomial.
pub(crate) type LogLinePolyDegreeBound = u32;

/// Maps a circle polynomial's degree bound to the degree bound of the line polynomial it gets
/// folded into.
fn folded_circle_poly_degree(degree: &LogCirclePolyDegreeBound) -> LogLinePolyDegreeBound {
    degree - LOG_CIRCLE_TO_LINE_FOLDING_FACTOR
}

#[derive(Error, Debug)]
pub enum VerificationError {
    #[error("proof contains an invalid number of FRI layers")]
    InvalidNumFriLayers,
    #[error("provided an invalid number of polynomials (expected {expected}, given {given}")]
    InvalidNumPolynomials { expected: usize, given: usize },
    #[error("queries do not resolve to their commitment in layer {layer}")]
    InnerLayerCommitmentInvalid { layer: usize },
    #[error("evaluations are invalid in layer {layer}")]
    InnerLayerEvaluationsInvalid { layer: usize },
    #[error("degree of last layer is invalid")]
    LastLayerDegreeInvalid,
    #[error("evaluations in the last layer are invalid")]
    LastLayerEvaluationsInvalid,
}

/// A FRI proof.
pub struct FriProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub inner_layers: Vec<FriLayerProof<F, H>>,
    pub last_layer_poly: LinePoly<F>,
}

/// Log folding factor for univariate (line) polynomials.
// TODO(andrew): Support multiple folding factors.
const LOG_FOLDING_FACTOR: u32 = 1;

/// Log folding factor for circle polynomials.
const LOG_CIRCLE_TO_LINE_FOLDING_FACTOR: u32 = 1;

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): support different folding factors
struct FriLayer<F: ExtensionOf<BaseField>, H: Hasher> {
    /// Coset evaluations stored in column-major.
    coset_evals: [Vec<F>; 1 << LOG_FOLDING_FACTOR],
    _merkle_tree: MerkleTree<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriLayer<F, H> {
    fn new(evaluation: &LineEvaluation<F>) -> Self {
        let (l, r) = evaluation.split_at(evaluation.len() / 2);
        let coset_evals = [l.to_vec(), r.to_vec()];
        // TODO(ohad): Add back once IntoSlice implemented for Field.
        // let merkle_tree = MerkleTree::commit(coset_evals.to_vec());
        #[allow(unreachable_code)]
        FriLayer {
            coset_evals,
            _merkle_tree: todo!(),
        }
    }

    /// Decommits to the coset evaluation at the specified positions.
    fn into_proof(self, positions: &[usize], folded_positions: &[usize]) -> FriLayerProof<F, H> {
        const COSET_SIZE: usize = 1 << LOG_FOLDING_FACTOR;
        let num_cosets = self.coset_evals[0].len();
        let coset_evals_subset = folded_positions
            .iter()
            .flat_map(|&folded_position| {
                let mut coset_evals: [Option<F>; COSET_SIZE] =
                    array::from_fn(|i| Some(self.coset_evals[i][folded_position]));

                // Remove evals the verifier will be able to calculate.
                for position in positions {
                    if position % num_cosets == folded_position {
                        coset_evals[position % COSET_SIZE] = None;
                    }
                }

                coset_evals
            })
            .flatten()
            .collect();
        // TODO(ohad): Add back once IntoSlice implemented for Field.
        // let position_set = positions.iter().copied().collect();
        // let decommitment = self.merkle_tree.generate_decommitment(position_set);
        // let commitment = self.merkle_tree.root();
        #[allow(unreachable_code)]
        FriLayerProof {
            coset_evals_subset,
            commitment: todo!(),
            decommitment: todo!(),
        }
    }
}

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
#[derive(Clone)]
pub struct FriLayerProof<F: ExtensionOf<BaseField>, H: Hasher> {
    /// Subset of coset evaluations.
    ///
    /// The subset stored corresponds to the set of evaluations the verifier doesn't have but needs
    /// in order to verify the decommitment.
    pub coset_evals_subset: Vec<F>,
    pub decommitment: MerkleDecommitment<F, H>,
    pub commitment: H::Hash,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriLayerProof<F, H> {
    /// Returns the validity of the merkle tree decommitment at the positions.
    pub fn verify(
        &self,
        _positions: &[usize],
        _coset_evals: &[[F; 1 << LOG_FOLDING_FACTOR]],
    ) -> bool {
        todo!()
    }

    /// Returns the coset evals in this layer needed for decommitment.
    ///
    /// `evals` must be the verifier's set of evals at the corresponding positions. `layer_size`
    /// should be the total number of evaluations in the entire layer.
    ///
    /// # Errors
    ///
    /// Returns [None] if the proof contains an invalid number of evaluations.
    ///
    /// # Panics
    ///
    /// Panics if the positions are not sorted in ascending order or if the number of positions
    /// doesn't match the number of evals.
    fn coset_evals(
        &self,
        positions: &[usize],
        folded_positions: &[usize],
        layer_size: usize,
        evals: &[F],
    ) -> Option<Vec<[F; 1 << LOG_FOLDING_FACTOR]>> {
        const COSET_SIZE: usize = 1 << LOG_FOLDING_FACTOR;

        assert!(positions.is_sorted());
        assert!(folded_positions.is_sorted());
        assert_eq!(positions.len(), evals.len());

        let num_cosets = layer_size / COSET_SIZE;

        // The evals provided by the verifier.
        let mut verifier_evals = evals.iter().copied();

        // The evals stored in the proof.
        let mut proof_evals = self.coset_evals_subset.iter().copied();

        let mut all_coset_evals = Vec::new();

        for &folded_position in folded_positions {
            let mut coset_evals = [None; COSET_SIZE];

            // Insert the verifier's evals.
            for position in positions {
                if position % num_cosets == folded_position {
                    coset_evals[position % COSET_SIZE] = Some(verifier_evals.next().unwrap());
                }
            }

            // Fill in the remaining evals using values stored in the proof.
            for eval in &mut coset_evals {
                if eval.is_none() {
                    *eval = Some(proof_evals.next()?);
                }
            }

            all_coset_evals.push(coset_evals.map(Option::unwrap))
        }

        // Check all the verifier's evals have been consumed.
        assert!(verifier_evals.next().is_none());

        // TODO(andrew): Do we even care if the proof stores too many evaluations?
        if proof_evals.next().is_some() {
            return None;
        }

        Some(all_coset_evals)
    }

    /// Returns the folded coset evaluations.
    ///
    /// `domain` must be the evaluation domain of the entire layer. `positions` should be the
    /// positions of cosets.
    ///
    /// # Panics
    ///
    /// Panics if the positions are not sorted in ascending order or if the number of positions
    /// doesn't match the number of coset evaluations.
    fn fold_evals(
        &self,
        domain: LineDomain,
        positions: &[usize],
        coset_evals: &[[F; 1 << LOG_FOLDING_FACTOR]],
        alpha: F,
    ) -> Vec<F> {
        assert!(positions.is_sorted());
        assert_eq!(positions.len(), coset_evals.len());
        zip(coset_evals, positions)
            .map(|(&[f_x, f_neg_x], &position)| {
                let coset_x = domain.at(position);
                let (mut f0, mut f1) = (f_x, f_neg_x);
                ibutterfly(&mut f0, &mut f1, coset_x.inverse());
                f0 + alpha * f1
            })
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
    evals: &LineEvaluation<F>,
    alpha: F,
) -> LineEvaluation<F> {
    let n = evals.len();
    assert!(n >= 2, "too few evals");

    let (l, r) = evals.split_at(n / 2);
    // TODO: Change LineDomain to be defined by its size and to always be canonic.
    let domain = LineDomain::new(Coset::half_odds(n.ilog2()));
    let folded_evals = zip(domain, zip(l, r))
        .map(|(x, (&f_x, &f_neg_x))| {
            let (mut f0, mut f1) = (f_x, f_neg_x);
            ibutterfly(&mut f0, &mut f1, x.inverse());
            f0 + alpha * f1
        })
        .collect();

    LineEvaluation::new(folded_evals)
}

/// Folds a degree `d` circle polynomial into a degree `d/2` univariate polynomial.
///
/// Let `evals` be the evaluation of a circle polynomial `f` on a [CircleDomain] `E`. This function
/// returns a univariate polynomial `f' = f0 + alpha * f1` evaluated on the x-coordinates of `E`
/// such that `2f(p) = f0(px) + py * f1(px)`.
fn fold_circle_to_line<F: ExtensionOf<BaseField>>(
    evals: &CircleEvaluation<F>,
    alpha: F,
) -> LineEvaluation<F> {
    let (l, r) = evals.split_at(evals.len() / 2);
    let folded_evals = zip(evals.domain, zip(l, r))
        .map(|(p, (&f_p, &f_neg_p))| {
            // Calculate `f0(px)` and `f1(px)` such that `2f(p) = f0(px) + py * f1(px)`.
            let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
            ibutterfly(&mut f0_px, &mut f1_px, p.y.inverse());
            f0_px + alpha * f1_px
        })
        .collect();

    LineEvaluation::new(folded_evals)
}

// TODO(andrew): support different folding factors
fn fold_positions(positions: &[usize], n: usize) -> Vec<usize> {
    let mut folded_positions = positions.iter().map(|p| p % n).collect::<Vec<usize>>();
    folded_positions.sort_unstable();
    folded_positions.dedup();
    folded_positions
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::One;

    use super::{FriConfig, FriProver, VerificationError};
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::constraints::EvalByEvaluation;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::ExtensionOf;
    use crate::core::fri::{fold_circle_to_line, fold_line, FriVerifier, LOG_FOLDING_FACTOR};
    use crate::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};
    use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};

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
        let evals = poly.evaluate(domain);
        let two = BaseField::from_u32_unchecked(2);

        let drp_evals = fold_line(&evals, alpha);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        for (i, (&drp_eval, x)) in zip(&*drp_evals, drp_domain).enumerate() {
            let f0 = even_poly.eval_at_point(x);
            let f1 = odd_poly.eval_at_point(x);
            assert_eq!(drp_eval, two * (f0 + alpha * f1), "mismatch at {i}");
        }
    }

    #[test]
    fn fold_circle_to_line_works() {
        const LOG_DEGREE: u32 = 4;
        let circle_evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let alpha = BaseField::one();

        let folded_evaluation = fold_circle_to_line(&circle_evaluation, alpha);

        assert_eq!(
            log_degree_bound(folded_evaluation),
            LOG_DEGREE - LOG_FOLDING_FACTOR
        );
    }

    #[test]
    #[should_panic = "invalid degree"]
    #[ignore = "commit not implemented"]
    fn committing_high_degree_polynomial_fails() {
        const LOG_EXPECTED_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR;
        const LOG_INVALID_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR - 1;
        let config = FriConfig::new(2, LOG_EXPECTED_BLOWUP_FACTOR);
        let prover = FriProver::<M31, Blake3Hasher>::new(config);
        let evaluation = polynomial_evaluation(6, LOG_INVALID_BLOWUP_FACTOR);

        prover.commit(vec![evaluation]);
    }

    #[test]
    #[should_panic = "not canonic"]
    fn committing_evaluation_from_invalid_domain_fails() {
        let invalid_domain = CircleDomain::new(Coset::new(CirclePointIndex::generator(), 3));
        assert!(!invalid_domain.is_canonic(), "must be an invalid domain");
        let evaluation = CircleEvaluation::new(invalid_domain, vec![QM31::one(); 1 << 4]);
        let prover = FriProver::<QM31, Blake3Hasher>::new(FriConfig::new(2, 2));

        prover.commit(vec![evaluation]);
    }

    #[test]
    #[ignore = "verification incomplete"]
    fn valid_fri_proof_passes_verification() -> Result<(), VerificationError> {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::new(config);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let oracle = EvalByEvaluation::new(CirclePointIndex::zero(), &polynomial);
        let prover = prover.commit(vec![polynomial.clone()]);
        let query_positions = [1, 8, 7];
        let proof = prover.into_proof(&query_positions);
        let verifier = FriVerifier::new(config, proof, vec![LOG_DEGREE]).unwrap();

        verifier.verify(&query_positions, &[oracle])
    }

    #[test]
    #[ignore = "verification incomplete"]
    fn mixed_degree_fri_proof_passes_verification() -> Result<(), VerificationError> {
        const DEGREES: [u32; 3] = [4, 5, 6];
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::new(config);
        let polynomials = DEGREES.map(|log_d| polynomial_evaluation(log_d, LOG_BLOWUP_FACTOR));
        let oracles = polynomials
            .iter()
            .map(|p| EvalByEvaluation::new(CirclePointIndex::zero(), p))
            .collect::<Vec<_>>();
        let prover = prover.commit(polynomials.to_vec());
        let query_positions = [1, 8, 7];
        let proof = prover.into_proof(&query_positions);
        let verifier = FriVerifier::new(config, proof, DEGREES.to_vec()).unwrap();

        verifier.verify(&query_positions, &oracles)
    }

    #[test]
    #[ignore = "verification incomplete"]
    fn proof_with_removed_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let prover_config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::new(prover_config);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let oracle = EvalByEvaluation::new(CirclePointIndex::zero(), &polynomial);
        let prover = prover.commit(vec![polynomial.clone()]);
        let query_positions = [1, 8, 7];
        let proof = prover.into_proof(&query_positions);
        // Set verifier config to expect one extra layer than prover config.
        let mut verifier_config = prover_config;
        verifier_config.log_last_layer_degree_bound -= 1;
        let verifier = FriVerifier::new(verifier_config, proof, vec![LOG_DEGREE]).unwrap();

        let verification_result = verifier.verify(&query_positions, &[oracle]);

        assert!(matches!(
            verification_result,
            Err(VerificationError::InvalidNumFriLayers)
        ));
    }

    #[test]
    #[ignore = "verification incomplete"]
    fn proof_with_added_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let prover_config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::new(prover_config);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let oracle = EvalByEvaluation::new(CirclePointIndex::zero(), &polynomial);
        let prover = prover.commit(vec![polynomial.clone()]);
        let query_positions = [1, 8, 7];
        let proof = prover.into_proof(&query_positions);
        // Set verifier config to expect one less layer than prover config.
        let mut verifier_config = prover_config;
        verifier_config.log_last_layer_degree_bound += 1;
        let verifier = FriVerifier::new(verifier_config, proof, vec![LOG_DEGREE]).unwrap();

        let verification_result = verifier.verify(&query_positions, &[oracle]);

        assert!(matches!(
            verification_result,
            Err(VerificationError::InvalidNumFriLayers)
        ));
    }

    #[test]
    #[ignore = "verification incomplete"]
    fn proof_with_invalid_inner_layer_evaluation_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::new(config);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let oracle = EvalByEvaluation::new(CirclePointIndex::zero(), &polynomial);
        let prover = prover.commit(vec![polynomial.clone()]);
        let query_position = [5];
        let mut proof = prover.into_proof(&query_position);
        // Compromises the evaluation in the second layer but retains a valid merkle decommitment.
        proof.inner_layers[1] = proof.inner_layers[0].clone();
        let verifier = FriVerifier::new(config, proof, vec![LOG_DEGREE]).unwrap();

        let verification_result = verifier.verify(&query_position, &[oracle]);

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
        let prover = FriProver::<QM31, Blake3Hasher>::new(config);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let oracle = EvalByEvaluation::new(CirclePointIndex::zero(), &polynomial);
        let prover = prover.commit(vec![polynomial.clone()]);
        let query_position = [5];
        let mut proof = prover.into_proof(&query_position);
        // Modify the committed values in the second layer.
        proof.inner_layers[1].coset_evals_subset[0] += QM31::one();
        let verifier = FriVerifier::new(config, proof, vec![LOG_DEGREE]).unwrap();

        let verification_result = verifier.verify(&query_position, &[oracle]);

        assert!(matches!(
            verification_result,
            Err(VerificationError::InnerLayerCommitmentInvalid { layer: 1 })
        ));
    }

    #[test]
    #[ignore = "verification incomplete"]
    fn proof_with_invalid_last_layer_degree_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        const LOG_MAX_LAST_LAYER_DEGREE: u32 = 2;
        let config = FriConfig::new(LOG_MAX_LAST_LAYER_DEGREE, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::new(config);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let prover = prover.commit(vec![polynomial.clone()]);
        let mut proof = prover.into_proof(&[1, 8, 7]);
        let bad_last_layer_coeffs = vec![QM31::one(); 1 << (LOG_MAX_LAST_LAYER_DEGREE + 1)];
        proof.last_layer_poly = LinePoly::new(bad_last_layer_coeffs);

        let verifier = FriVerifier::new(config, proof, vec![LOG_DEGREE]);

        assert!(matches!(
            verifier,
            Err(VerificationError::LastLayerDegreeInvalid)
        ));
    }

    #[test]
    #[ignore = "verification incomplete"]
    fn proof_with_invalid_last_layer_fails_verification() {
        const LOG_DEGREE: u32 = 6;
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR);
        let prover = FriProver::<QM31, Blake3Hasher>::new(config);
        let polynomial = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
        let oracle = EvalByEvaluation::new(CirclePointIndex::zero(), &polynomial);
        let prover = prover.commit(vec![polynomial.clone()]);
        let query_positions = [1, 8, 7];
        let mut proof = prover.into_proof(&query_positions);
        // Compromise the last layer polynomial's first coefficient.
        proof.last_layer_poly[0] += QM31::one();
        let verifier = FriVerifier::new(config, proof, vec![LOG_DEGREE]).unwrap();

        let verification_result = verifier.verify(&query_positions, &[oracle]);

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
    ) -> CircleEvaluation<F> {
        let poly = CirclePoly::new(vec![F::one(); 1 << log_degree]);
        let coset = Coset::half_odds(log_degree + log_blowup_factor - 1);
        let domain = CircleDomain::new(coset);
        poly.evaluate(domain)
    }

    /// Returns the log degree bound of a polynomial.
    fn log_degree_bound<F: ExtensionOf<BaseField>>(polynomial: LineEvaluation<F>) -> u32 {
        let domain = LineDomain::new(Coset::half_odds(polynomial.len().ilog2()));
        let coeffs = polynomial.interpolate(domain).into_ordered_coefficients();
        let degree = coeffs.into_iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        (degree + 1).ilog2()
    }
}
