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
use crate::core::poly::circle::CircleDomain;
use crate::core::poly::line::LineDomain;

/// FRI proof config
// TODO(andrew): support different folding factors
#[derive(Debug, Clone, Copy)]
pub struct FriConfig {
    log_last_layer_degree_bound: u32,
    log_blowup_factor: u32,
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
            log_last_layer_degree_bound,
            log_blowup_factor,
        }
    }

    fn max_last_layer_domain_size(&self) -> usize {
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
    /// * An evaluation is not from a sufficiently low degree polynomial.
    /// * An evaluation domain is smaller than or equal to 2x the maximum last layer domain size.
    /// * An evaluation domain is not canonic circle domain.
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
    /// Returns the evaluation of the last layer.
    ///
    /// # Panics
    ///
    /// Panics if `evals` is empty or if an evaluation domain is smaller than or equal to the
    /// maximum last layer domain size.
    fn commit_inner_layers(&mut self, mut evals: Vec<CircleEvaluation<F>>) -> LineEvaluation<F> {
        let mut evaluation = {
            // TODO(andrew): draw from channel
            let alpha = F::one();
            let first_circle_evaluaiton = evals.pop().expect("requires an evaluation");
            fold_circle_to_line(&first_circle_evaluaiton, alpha)
        };

        while evaluation.len() > self.config.max_last_layer_domain_size() {
            let folded_len = |e: &CircleEvaluation<F>| e.len() >> LOG_CIRCLE_TO_LINE_FOLDING_FACTOR;

            // Check for any evaluations that should be combined.
            while evals.last().map(folded_len) == Some(evaluation.len()) {
                // TODO(andrew): draw random alpha from channel
                let alpha = F::one();
                let folded_evaluation = fold_circle_to_line(&evals.pop().unwrap(), alpha);
                assert_eq!(folded_evaluation.len(), evaluation.len());
                for (i, eval) in folded_evaluation.into_iter().enumerate() {
                    evaluation[i] += alpha * eval;
                }
            }

            let layer = FriLayer::new(&evaluation);
            // TODO(andrew): add merkle root to channel
            // TODO(ohad): Add back once IntoSlice implemented for Field.
            // let _merkle_root = layer.merkle_tree.root();
            // TODO(andrew): draw random alpha from channel
            let alpha = F::one();
            evaluation = fold_line(&evaluation, alpha);
            self.inner_layers.push(layer)
        }

        assert!(evals.is_empty());
        evaluation
    }

    /// Builds and commits to the last layer.
    ///
    /// The layer is committed to by sending the verifier all the coefficients of the polynomial in
    /// the last layer.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation domain size exceeds the maximum last layer domain size.
    /// * The evaluation is not of sufficiently low degree.
    fn commit_last_layer(&mut self, evaluation: LineEvaluation<F>) {
        assert!(evaluation.len() <= self.config.max_last_layer_domain_size());
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
                fold_positions(positions, num_layer_cosets);
                Some(layer.into_proof(positions))
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
    /// The FRI config
    config: FriConfig,
    /// The FRI proof.
    proof: FriProof<F, H>,
    /// The list of degree bounds of all committed circle polynomials.
    degrees: Vec<LogCirclePolyDegree>,
    /// Alphas used for foldings.
    alphas: Vec<F>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriVerifier<F, H> {
    /// Creates a new FRI verifier.
    ///
    /// This verifier can verify multiple polynomials, with different degrees, are each low degree.
    /// `degrees` should be a list of degree bounds of all committed circle polynomials.
    ///
    /// # Errors
    ///
    /// An `Err` will be returned if:
    /// * The proof contains an invalid number of FRI layers.
    /// * The last layer polynomial's degree is too high.
    ///
    /// # Panics
    ///
    /// Panics if `degrees` is empty or if a degree bound is less than the maximum last layer
    /// degree bound.
    pub fn new(
        config: FriConfig,
        proof: FriProof<F, H>,
        mut degrees: Vec<LogCirclePolyDegree>,
    ) -> Result<Self, VerificationError> {
        degrees.sort();
        let mut alphas = Vec::new();

        {
            let mut layer_proofs = proof.inner_layers.iter();
            let mut degrees = degrees.clone();
            let mut log_degree = degrees.last().map(folded_circle_poly_degree).unwrap();
            while log_degree > config.log_last_layer_degree_bound {
                // Check a proof exists for this layer.
                if layer_proofs.next().is_none() {
                    return Err(VerificationError::InvalidNumFriLayers);
                }

                // Draw alphas for the evaluations that will be combined into this layer.
                while degrees.last().map(folded_circle_poly_degree) == Some(log_degree) {
                    degrees.pop();
                    // TODO(andrew): draw random alpha from channel
                    let alpha = F::one();
                    alphas.push(alpha);
                }

                // TODO(andrew): Seed channel with commitment.
                // TODO(andrew): Draw alpha from channel.
                let alpha = F::one();
                alphas.push(alpha);

                log_degree -= LOG_FOLDING_FACTOR;
            }

            // Check there aren't too many layer proofs.
            if layer_proofs.next().is_some() {
                return Err(VerificationError::InvalidNumFriLayers);
            }

            // Ensure there aren't any circle polynomials left out.
            assert!(degrees.is_empty());

            // Check the degree of the last layer's polynomial for the last layer.
            let last_layer_degree_bound = 1 << log_degree;
            if proof.last_layer_poly.len() > last_layer_degree_bound {
                return Err(VerificationError::LastLayerDegreeInvalid);
            }
        }

        Ok(Self {
            config,
            alphas,
            degrees,
            proof,
        })
    }

    /// Verifies the FRI commitment.
    ///
    /// The polynomials need to be provided in the same order as their commitment.
    pub fn verify(
        self,
        query_positions: &[usize],
        circle_polynomials: Vec<impl PolyOracle<F>>,
    ) -> Result<(), VerificationError> {
        if circle_polynomials.len() != self.degrees.len() {
            return Err(VerificationError::InvalidNumPolynomials {
                expected: self.degrees.len(),
                given: circle_polynomials.len(),
            });
        }

        let (last_layer_domain, last_layer_positions, last_layer_evals) =
            self.verify_inner_layers(query_positions.to_vec(), circle_polynomials)?;
        self.verify_last_layer(last_layer_domain, last_layer_positions, last_layer_evals)
    }

    /// Verifies all layers except the last layer.
    ///
    /// Returns the domain, query positions and evaluations needed for verifying the last FRI
    /// layer. Output is of the form: `(domain, query_positions, evaluations)`.
    fn verify_inner_layers(
        &self,
        mut positions: Vec<usize>,
        mut circle_polynomials: Vec<impl PolyOracle<F>>,
    ) -> Result<(LineDomain, Vec<usize>, Vec<F>), VerificationError> {
        let log_blowup_factor = self.config.log_blowup_factor;

        let mut alphas = self.alphas.iter().copied();
        let mut degrees = self.degrees.clone();
        let mut log_degree = degrees.last().map(folded_circle_poly_degree).unwrap();
        let mut evals = vec![F::zero(); positions.len()];
        let mut domain = LineDomain::new(Coset::half_odds(log_degree + log_blowup_factor));

        for (i, layer) in self.proof.inner_layers.iter().enumerate() {
            let mut folded_positions = positions.clone();
            fold_positions(&mut folded_positions, domain.size());

            // Verify the decommitment.
            if !layer.verify(&folded_positions) {
                return Err(VerificationError::InnerLayerCommitmentInvalid { layer: i });
            }

            // Fold and combine circle polynomial evaluations.
            while degrees.last().map(folded_circle_poly_degree) == Some(log_degree) {
                degrees.pop();
                let circle_polynomial = circle_polynomials.pop().unwrap();
                let alpha = alphas.next().unwrap();
                let circle_domain = CircleDomain::new(domain.coset());
                for (eval, &position) in zip(&mut evals, &positions) {
                    let p_index = circle_domain.index_at(position);
                    let p = p_index.to_point();
                    let f_p = circle_polynomial.get_at(p_index);
                    let f_neg_p = circle_polynomial.get_at(-p_index);
                    let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
                    ibutterfly(&mut f0_px, &mut f1_px, p.y.inverse());
                    let folded_eval = f0_px + alpha * f1_px;
                    *eval += alpha * folded_eval;
                }
            }

            // Verify the evals match.
            if evals != layer.get_query_evals(&positions, &folded_positions, domain.log_size()) {
                return Err(VerificationError::InnerLayerEvaluationsInvalid { layer: i });
            }

            // Prepare the next layer.
            evals = layer.fold_evals(&folded_positions, domain, alphas.next().unwrap());
            positions = folded_positions;
            log_degree -= LOG_FOLDING_FACTOR;
            domain = domain.double();
        }

        // Check all values have been consumed.
        assert!(alphas.next().is_none());
        assert!(degrees.is_empty());
        assert!(circle_polynomials.is_empty());

        Ok((domain, positions, evals))
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

/// Degree bound of a circle polynomial stored as `log2(degree_bound)`.
pub(crate) type LogCirclePolyDegree = u32;

/// Degree bound of a univariate (line) polynomial stored as `log2(degree_bound)`.
pub(crate) type LogLinePolyDegree = u32;

/// Maps a circle polynomial's degree bound to the degree bound of the line polynomial it gets
/// folded into.
fn folded_circle_poly_degree(degree: &LogCirclePolyDegree) -> LogLinePolyDegree {
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

/// Folding factor for univariate polynomials.
// TODO(andrew): Remove constant and support multiple folding factors.
const LOG_FOLDING_FACTOR: u32 = 1;

/// Folding factor when folding a circle polynomial to univariate polynomial.
const LOG_CIRCLE_TO_LINE_FOLDING_FACTOR: u32 = 1;

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
///
/// The subset corresponds to the set of evaluations needed by a FRI verifier.
#[derive(Clone)]
pub struct FriLayerProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub coset_evals: Vec<[F; 1 << LOG_FOLDING_FACTOR]>,
    pub decommitment: MerkleDecommitment<F, H>,
    pub commitment: H::Hash,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriLayerProof<F, H> {
    /// Returns the validity of the merkle tree decommitment at the positions.
    pub fn verify(&self, _positions: &[usize]) -> bool {
        todo!()
    }

    /// Returns the evaluations at the corresponding query positions.
    ///
    /// Domain size must be the evaluation domain size of the entire layer.
    /// Adapted from <https://github.com/facebook/winterfell/blob/main/fri/src/verifier/mod.rs#L327>.
    ///
    /// # Panics
    ///
    /// Panics if all positions aren't sorted in ascending order.
    fn get_query_evals(
        &self,
        positions: &[usize],
        folded_positions: &[usize],
        log_domain_size: u32,
    ) -> Vec<F> {
        assert!(positions.is_sorted());
        assert!(folded_positions.is_sorted());

        positions
            .iter()
            .map(|position| {
                let folded_domain_size = 1 << (log_domain_size - LOG_FOLDING_FACTOR);
                let folded_position = position % folded_domain_size;
                let i = folded_positions.binary_search(&folded_position).unwrap();
                self.coset_evals[i][position % (1 << LOG_FOLDING_FACTOR)]
            })
            .collect()
    }

    /// Returns the folded coset evaluations.
    ///
    /// `domain` must be the evaluation domain of the entire layer.
    ///
    /// # Panics
    ///
    /// If the positions are not sorted in ascending order or if the number of positions doesn't
    /// match the number of coset evaluations.
    fn fold_evals(&self, folded_positions: &[usize], domain: LineDomain, alpha: F) -> Vec<F> {
        assert!(folded_positions.is_sorted());
        assert_eq!(folded_positions.len(), self.coset_evals.len());
        zip(&self.coset_evals, folded_positions)
            .map(|(&[f_x, f_neg_x], &position)| {
                let coset_x = domain.at(position);
                let (mut f0, mut f1) = (f_x, f_neg_x);
                ibutterfly(&mut f0, &mut f1, coset_x.inverse());
                f0 + alpha * f1
            })
            .collect()
    }
}

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
    fn into_proof(self, positions: &[usize]) -> FriLayerProof<F, H> {
        let coset_evals = positions
            .iter()
            .map(|i| {
                let eval0 = self.coset_evals[0][*i];
                let eval1 = self.coset_evals[1][*i];
                [eval0, eval1]
            })
            .collect();
        // TODO(ohad): Add back once IntoSlice implemented for Field.
        // let position_set = positions.iter().copied().collect();
        // let decommitment = self.merkle_tree.generate_decommitment(position_set);
        // let commitment = self.merkle_tree.root();
        #[allow(unreachable_code)]
        FriLayerProof {
            coset_evals,
            decommitment: todo!(),
            commitment: todo!(),
        }
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
    // TODO: Either add domain to [LineEvaluation] or consider changing LineDomain to be defined by
    // its size and to always be canonic - since it may only be used by FRI.
    let domain = LineDomain::new(Coset::half_odds(n.ilog2()));
    let folded_evals = zip(domain.iter(), zip(l, r))
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
    let folded_evals = zip(evals.domain.iter(), zip(l, r))
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
fn fold_positions(positions: &mut Vec<usize>, n: usize) {
    positions.iter_mut().for_each(|p| *p %= n);
    positions.sort_unstable();
    positions.dedup();
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
        for (i, (&drp_eval, x)) in zip(&*drp_evals, drp_domain.iter()).enumerate() {
            let f_e = even_poly.eval_at_point(x);
            let f_o = odd_poly.eval_at_point(x);
            assert_eq!(drp_eval, two * (f_e + alpha * f_o), "mismatch at {i}");
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
        // not a canonic domain
        let invalid_domain = CircleDomain::new(Coset::new(CirclePointIndex(1), 3));
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

        verifier.verify(&query_positions, vec![oracle])
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
            .collect();
        let prover = prover.commit(polynomials.to_vec());
        let query_positions = [1, 8, 7];
        let proof = prover.into_proof(&query_positions);
        let verifier = FriVerifier::new(config, proof, DEGREES.to_vec()).unwrap();

        verifier.verify(&query_positions, oracles)
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

        let verification_result = verifier.verify(&query_positions, vec![oracle]);

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

        let verification_result = verifier.verify(&query_positions, vec![oracle]);

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

        let verification_result = verifier.verify(&query_position, vec![oracle]);

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
        proof.inner_layers[1].coset_evals[0][0] += QM31::one();
        let verifier = FriVerifier::new(config, proof, vec![LOG_DEGREE]).unwrap();

        let verification_result = verifier.verify(&query_position, vec![oracle]);

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

        let verification_result = verifier.verify(&query_positions, vec![oracle]);

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

    /// Returns the degree bound of a polynomial as `log2(degree_bound)`.
    fn log_degree_bound<F: ExtensionOf<BaseField>>(polynomial: LineEvaluation<F>) -> u32 {
        let domain = LineDomain::new(Coset::half_odds(polynomial.len().ilog2()));
        let coeffs = polynomial.interpolate(domain).into_ordered_coefficients();
        let degree = coeffs.into_iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        (degree + 1).ilog2()
    }
}
