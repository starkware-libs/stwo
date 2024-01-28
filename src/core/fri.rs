use std::cmp::Reverse;
use std::fmt::Debug;
use std::iter::zip;
use std::ops::RangeInclusive;

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
    inner_layers: Vec<FriLayerProver<F, H>>,
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
    ) -> (
        Vec<FriLayerProver<F, H>>,
        LineEvaluation<F, BitReversedOrder>,
    ) {
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

            let layer = FriLayerProver::new(&evaluation);

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
    pub fn verify(&self, _queries: &Queries) -> Result<(), String> {
        todo!()
    }
}

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): support different folding factors
struct FriLayerProver<F: ExtensionOf<BaseField>, H: Hasher> {
    /// Coset evaluations stored in column-major.
    subcircle_evals: [Vec<F>; 1 << LOG_FOLDING_FACTOR],
    _merkle_tree: MerkleTree<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriLayerProver<F, H> {
    fn new(evaluation: &LineEvaluation<F, BitReversedOrder>) -> Self {
        // TODO(andrew): With bit-reversed order coset evals are next to each other. Update.
        let (l, r) = evaluation.split_at(evaluation.len() / 2);
        let subcircle_evals = [l.to_vec(), r.to_vec()];
        // TODO(ohad): Add back once IntoSlice implemented for Field.
        // let merkle_tree = MerkleTree::commit(coset_evals.to_vec());
        #[allow(unreachable_code)]
        FriLayerProver {
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
            let mut subcircle_queries = query_group
                .iter()
                .map(|q| q & (SUBCIRCLE_SIZE - 1))
                .peekable();

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

    // TODO: Change LineDomain to be defined by its size and to always be canonic.
    let domain = LineDomain::new(Coset::half_odds(n.ilog2()));
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

    LineEvaluation::new(folded_evals)
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

        let mut folded_evaluation = LineEvaluation::new(vec![BaseField::zero(); num_folded_evals]);
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
