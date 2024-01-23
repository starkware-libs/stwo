use std::cmp::Reverse;
use std::fmt::Debug;
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::RangeInclusive;

use super::fields::m31::BaseField;
use super::fields::{ExtensionOf, Field};
use super::poly::circle::CircleEvaluation;
use super::poly::line::{LineEvaluation, LinePoly};
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::poly::line::LineDomain;

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

    /// Commits to multiple [LineEvaluation]s.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `evals` is empty or not sorted in ascending order by evaluation domain size.
    /// * An evaluation domain is smaller than or equal to the max last layer domain size.
    /// * An evaluation is not of sufficiently low degree.
    pub fn commit(mut self, evals: Vec<LineEvaluation<F>>) -> FriProver<F, H, QueryPhase> {
        assert!(evals.is_sorted_by_key(|evaluation| Reverse(evaluation.len())));
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
    /// Panics if:
    /// * `evals` is empty.
    /// * An evaluation domain is smaller than or equal to the max last layer domain size.
    fn commit_inner_layers(&mut self, mut evals: Vec<LineEvaluation<F>>) -> LineEvaluation<F> {
        let mut evaluation = evals.pop().expect("require at least one evaluation");
        while evaluation.len() > self.config.last_layer_domain_size() {
            // Aggregate all evaluations that have the same domain size.
            while let Some(true) = evals.last().map(|e| e.len() == evaluation.len()) {
                // TODO(andrew): draw random alpha from channel
                let alpha = F::one();
                for (i, &eval) in evals.pop().unwrap().iter().enumerate() {
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
    /// The layer is committed to by sending the verifier all the coefficients of the remaining
    /// polynomial.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation's domain size exceeds the maximum last layer domain size.
    /// * The evaluation is not of sufficiently low degree.
    fn commit_last_layer(&mut self, evaluation: LineEvaluation<F>) {
        assert_eq!(evaluation.len(), self.config.last_layer_domain_size());
        let max_num_coeffs = evaluation.len() >> self.config.log_blowup_factor;
        let domain = LineDomain::new(Coset::half_odds(evaluation.len().ilog2()));
        let mut coeffs = evaluation.interpolate(domain).into_ordered_coefficients();
        let zeros = coeffs.split_off(max_num_coeffs);
        assert!(zeros.iter().all(F::is_zero), "invalid degree");
        self.last_layer_poly = Some(LinePoly::from_ordered_coefficients(coeffs));
        // TODO(andrew): seed channel with last_layer_poly
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

/// A FRI proof.
pub struct FriProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub inner_layers: Vec<FriLayerProof<F, H>>,
    pub last_layer_poly: LinePoly<F>,
}

/// Folding factor for univariate polynomials.
// TODO(andrew): Support multiple folding factors.
const LOG_FOLDING_FACTOR: u32 = 1;

/// Folding factor when folding a circle polynomial to univariate polynomial.
const _LOG_CIRCLE_TO_LINE_FOLDING_FACTOR: u32 = 1;

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
///
/// The subset corresponds to the set of evaluations needed by a FRI verifier.
pub struct FriLayerProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub coset_evals: Vec<[F; 1 << LOG_FOLDING_FACTOR]>,
    pub decommitment: MerkleDecommitment<F, H>,
    pub commitment: H::Hash,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriLayerProof<F, H> {
    // TODO(andrew): implement and add docs
    // TODO(andrew): create FRI verification error type
    pub fn verify(&self, _positions: &[usize]) -> Result<(), String> {
        todo!()
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
// TODO(andrew): Fold directly into FRI layer to prevent allocation.
fn _fold_circle_to_line<F: ExtensionOf<BaseField>>(
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
fn fold_positions(positions: &mut Vec<usize>, n: usize) {
    positions.iter_mut().for_each(|p| *p %= n);
    positions.sort_unstable();
    positions.dedup();
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::One;

    use crate::core::circle::Coset;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::ExtensionOf;
    use crate::core::fri::{_fold_circle_to_line, fold_line, _LOG_CIRCLE_TO_LINE_FOLDING_FACTOR};
    use crate::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};
    use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};

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
        let evals = poly.evaluate(domain);
        let two = BaseField::from_u32_unchecked(2);

        let drp_evals = fold_line(&evals, alpha);

        assert_eq!(drp_evals.len(), DEGREE / 2);
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
        let alpha = BaseField::one();

        let folded_evaluation = _fold_circle_to_line(&circle_evaluation, alpha);

        assert_eq!(
            log_degree_bound(folded_evaluation),
            LOG_DEGREE - _LOG_CIRCLE_TO_LINE_FOLDING_FACTOR
        );
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
