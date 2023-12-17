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
use crate::core::constraints::coset_vanishing;
use crate::core::fft::ibutterfly;
use crate::core::poly::circle::CircleDomain;
use crate::core::poly::line::LineDomain;

/// FRI proof config
// TODO(andrew): support different folding factors
#[derive(Debug, Clone, Copy)]
pub struct FriConfig {
    last_layer_degree_bits: u32,
    blowup_factor_bits: u32,
}

impl FriConfig {
    const MIN_LAST_LAYER_DEGREE_BITS: u32 = 0;
    const MAX_LAST_LAYER_DEGREE_BITS: u32 = 10;
    const LAST_LAYER_DEGREE_BITS_RANGE: RangeInclusive<u32> =
        Self::MIN_LAST_LAYER_DEGREE_BITS..=Self::MAX_LAST_LAYER_DEGREE_BITS;

    const MIN_BLOWUP_FACTOR_BITS: u32 = 1;
    const MAX_BLOWUP_FACTOR_BITS: u32 = 16;
    const BLOWUP_FACTOR_BITS_RANGE: RangeInclusive<u32> =
        Self::MIN_BLOWUP_FACTOR_BITS..=Self::MAX_BLOWUP_FACTOR_BITS;

    /// Creates a new FRI configuration.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `last_layer_degree_bits` is greater than 10.
    /// * `blowup_factor_bits` is equal to zero or greater than 16.
    pub fn new(last_layer_degree_bits: u32, blowup_factor_bits: u32) -> Self {
        assert!(Self::LAST_LAYER_DEGREE_BITS_RANGE.contains(&last_layer_degree_bits));
        assert!(Self::BLOWUP_FACTOR_BITS_RANGE.contains(&blowup_factor_bits));
        Self {
            last_layer_degree_bits,
            blowup_factor_bits,
        }
    }

    fn max_last_layer_domain_size(&self) -> usize {
        1 << (self.last_layer_degree_bits + self.blowup_factor_bits)
    }
}

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
///
/// `Phase` is used to enforce the commitment phase is done before the query phase.
pub struct FriProver<F: ExtensionOf<BaseField>, H: Hasher, Phase = CommitmentPhase> {
    config: FriConfig,
    layers: Vec<FriLayer<F, H>>,
    remainder: Option<LinePoly<F>>,
    _phase: PhantomData<Phase>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriProver<F, H, CommitmentPhase> {
    /// Creates a new FRI prover.
    pub fn new(config: FriConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
            remainder: None,
            _phase: PhantomData,
        }
    }

    /// Commits to multiple [CircleEvaluation]s.
    ///
    /// Mixed degree STARKs involve polynomials evaluated on multiple domains of different size and
    /// structure. Combining evaluations on different domains into an evaluation on a single domain
    /// can be inefficient. Instead you can commit to multiple evaluations over different domains
    /// individually and the necessary shifts and combining is taken care of at the appropriate
    /// FRI layer.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `evals` is empty or not sorted in ascending order by evaluation domain size.
    /// * An evaluation domain is smaller than or equal to the maximum last layer domain size.
    /// * An evaluation is not of sufficiently low degree.
    pub fn commit(mut self, evals: Vec<CircleEvaluation<F>>) -> FriProver<F, H, QueryPhase> {
        assert!(evals.is_sorted_by_key(|e| e.len()));
        let last_layer_evaluation = self.commit_inner_layers(evals);
        self.commit_last_layer(last_layer_evaluation);
        FriProver {
            config: self.config,
            layers: self.layers,
            remainder: self.remainder,
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
    /// * An evaluation domain is smaller than or equal to the maximum last layer domain size.
    // TODO(andrew): Consider folding circle evaluations on a canonical domain differently as they
    // only needed to be folded into line evaluations.
    fn commit_inner_layers(&mut self, mut evals: Vec<CircleEvaluation<F>>) -> LineEvaluation<F> {
        let mut line_evaluation = {
            // TODO(andrew): draw from channel
            let alpha = F::one();
            let first_circle_evaluaiton = evals.pop().expect("requires an evaluation");
            fold_circle_to_line(&first_circle_evaluaiton, alpha)
        };

        const CIRCLE_TO_LINE_FOLDING_FACTOR: usize = 4;
        let folded_len = |e: &CircleEvaluation<F>| e.len() / CIRCLE_TO_LINE_FOLDING_FACTOR;

        while line_evaluation.len() > self.config.max_last_layer_domain_size() {
            // Check for any evaluations that should be combined.
            while evals.last().map(folded_len) == Some(line_evaluation.len()) {
                // TODO(andrew): draw random alpha from channel
                let alpha = F::one();
                let folded_evaluation = fold_circle_to_line(&evals.pop().unwrap(), alpha);
                assert_eq!(folded_evaluation.len(), line_evaluation.len());
                for (i, eval) in folded_evaluation.into_iter().enumerate() {
                    line_evaluation[i] += alpha * eval;
                }
            }

            let layer = FriLayer::new(&line_evaluation);
            // TODO(andrew): add merkle root to channel
            // TODO(ohad): Add back once IntoSlice implemented for Field.
            // let _merkle_root = layer.merkle_tree.root();
            // TODO(andrew): draw random alpha from channel
            let alpha = F::one();
            line_evaluation = fold_line(&line_evaluation, alpha);
            self.layers.push(layer)
        }

        assert!(evals.is_empty());
        line_evaluation
    }

    /// Builds and commits to the FRI remainder polynomial's coefficients (the last FRI layer).
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation domain size exceeds the maximum last layer domain size.
    /// * The evaluation is not of sufficiently low degree.
    fn commit_last_layer(&mut self, evaluation: LineEvaluation<F>) {
        assert!(evaluation.len() <= self.config.max_last_layer_domain_size());
        let num_remainder_coeffs = evaluation.len() >> self.config.blowup_factor_bits;
        let domain = LineDomain::new(Coset::half_odds(evaluation.len().ilog2() as usize));
        let mut coeffs = evaluation.interpolate(domain).into_natural_coefficients();
        let zeros = coeffs.split_off(num_remainder_coeffs);
        assert!(zeros.iter().all(F::is_zero), "invalid degree");
        self.remainder = Some(LinePoly::from_natural_coefficients(coeffs));
        // TODO(andrew): seed channel with remainder
    }
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriProver<F, H, QueryPhase> {
    pub fn into_proof(self, query_positions: &[usize]) -> FriProof<F, H> {
        let remainder = self.remainder.unwrap();
        let layer_proofs = self
            .layers
            .into_iter()
            .scan(query_positions.to_vec(), |positions, layer| {
                let num_layer_cosets = layer.coset_evals[0].len();
                fold_positions(positions, num_layer_cosets);
                Some(layer.into_proof(positions))
            })
            .collect();
        FriProof {
            layer_proofs,
            remainder,
        }
    }
}

/// Commitment phase for [FriProver].
pub struct CommitmentPhase;

/// Query phase for [FriProver].
pub struct QueryPhase;

/// A FRI proof.
pub struct FriProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub layer_proofs: Vec<FriLayerProof<F, H>>,
    pub remainder: LinePoly<F>,
}

// TODO(andrew): Remove constant and support multiple folding factors.
const FRI_STEP_SIZE: usize = 2;

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
///
/// The subset corresponds to the set of evaluations needed by a FRI verifier.
pub struct FriLayerProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub coset_evals: Vec<[F; FRI_STEP_SIZE]>,
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
    coset_evals: [Vec<F>; FRI_STEP_SIZE],
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

/// Folds a degree `<d` polynomial evaluated on [LineDomain] `E` into a degree `<d/2` polynomial
/// evaluated on `2E`.
///
/// Example: Our evaluation domain is the x-coordinates of `E = c + <G>`, `alpha` is a random field
/// element and `pi(x) = 2x^2 - 1` is the circle's x-coordinate doubling map. We have evaluations of
/// a polynomial `f` on `E` (i.e `evals`) and we can compute the evaluations of `f' = fe +
/// alpha * fo` over `E' = { pi(x) | x in E }` such that `f(x) = (fe(pi(x)) + x * fo(pi(x))) / 2`.
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
    let domain = LineDomain::new(Coset::half_odds(n.ilog2() as usize));
    let folded_evals = zip(domain.iter(), zip(l, r))
        .map(|(x, (&f_x, &f_neg_x))| {
            let (mut f_e, mut f_o) = (f_x, f_neg_x);
            ibutterfly(&mut f_e, &mut f_o, x.inverse());
            f_e + alpha * f_o
        })
        .collect();

    LineEvaluation::new(folded_evals)
}

/// Folds evaluations of a polynomial on a [CircleDomain] into evaluations on a [LineDomain].
///
/// This folds a degree `<d` polynomial evaluated on any [CircleDomain] of size `n` into a
/// polynomial of degree `<d/4` evaluated on a canonic [LineDomain] (a domain of the form
/// `{p_x | p in +-G_4n + <G_n>}`) of size `n / 4` .
///
/// Let `evals` be the evaluations of a polynomial `f` on the domain `E = +-c + <G_n>` and `v0`,
/// `v1` be polynomials that vanish on `g + <G>` and `-g + <G>` respectively. We can obtain the
/// evals of `f0` and `f1` on `E' = +-G_n + <G_n/2>` such that `f(p) = v1(p) * f0(p-c+G_n) + v0(p) *
/// f1(p+c-G_n)`. This function returns `f' = f00 + alpha * f01 + alpha^2 * f10 + alpha^3 * f11`
/// evaluated on the x-coordinates of `E'` such that `f0(p) = (f00(px) + py * f01(px)) / 2` and
/// `f1(p) = (f10(px) + py * f11(px)) / 2`.
///
/// # Panics
///
/// Panics if there are less than four evaluations.
fn fold_circle_to_line<F: ExtensionOf<BaseField>>(
    evals: &CircleEvaluation<F>,
    alpha: F,
) -> LineEvaluation<F> {
    let n = evals.len();
    assert!(n >= 4, "too few evals");

    // TODO: Faster to do all these operations in a single pass.
    let (f0, f1) = split_and_shift(evals);
    let (f00, f01) = split_to_line(&f0);
    let (f10, f11) = split_to_line(&f1);

    let [a0, a1, a2, a3] = [F::one(), alpha, alpha.pow(2), alpha.pow(3)];
    let folded_evals = zip(zip(f00, f01), zip(f10, f11))
        .map(|((f00, f01), (f10, f11))| f00 * a0 + f01 * a1 + f10 * a2 + f11 * a3)
        .collect();

    LineEvaluation::new(folded_evals)
}

/// Splits a polynomial evaluated on a circle domain of size `n` into two shifted polynomials
/// evaluated on a canonic circle domain (domains of the form `+-G_2n + <G_n>`) of size `n / 2`.
///
/// Let `evals` be the evaluations of a polynomial `f` on the domain `E = +-c + <G_n>` and `v0`,
/// `v1` be polynomials that vanish on `c + <G_n>` and `-c + <G_n>` respectively. This function
/// returns polynomials `f0` and `f1` evaluated on `E' = +-G_n + <G_n/2>` such that `f(p) = v1(p) *
/// f0(p-c+G_n) + v0(p) * f1(p+c-G_n)`.
///
/// # Panics
///
/// Panics if there are less than two evaluations.
fn split_and_shift<F: ExtensionOf<BaseField>>(
    evals: &CircleEvaluation<F>,
) -> (CircleEvaluation<F>, CircleEvaluation<F>) {
    let n = evals.len();
    assert!(n >= 2, "too few evals");

    // TODO: This whole function can be a 4-value uninterleave
    let (_e0, _e1) = evals.split_at(n / 2);
    let half_coset = evals.domain.half_coset;
    let half_coset_conjugate = half_coset.conjugate();
    let (f0_evals, f1_evals) = zip(half_coset, half_coset_conjugate)
        .enumerate()
        .map(|(i, (p0, p1))| {
            // v0(p1) and v1(p0) will alternate between 1 and -1
            let _v0_p1 = coset_vanishing(half_coset, p1);
            let _v1_p0 = coset_vanishing(half_coset_conjugate, p0);
            // (_e0[i] / _v1_p0, _e1[i] / _v0_p1)
            (_e0[i], _e1[i])
        })
        .unzip();

    let reorder = |evals: Vec<F>| -> Vec<F> {
        #[allow(clippy::tuple_array_conversions)]
        let (even_evals, mut odd_evals): (Vec<F>, Vec<F>) =
            evals.array_chunks().map(|[e, o]| (e, o)).unzip();
        odd_evals.reverse();
        #[allow(clippy::tuple_array_conversions)]
        [even_evals, odd_evals].concat()
    };

    // Reorder the evals in the same order as the domain
    let f0_evals = reorder(f0_evals);
    let f1_evals = reorder(f1_evals);

    let canonic_domain = CircleDomain::new(Coset::half_odds(n.ilog2() as usize - 2));
    let f0 = CircleEvaluation::new(canonic_domain, f0_evals);
    let f1 = CircleEvaluation::new(canonic_domain, f1_evals);

    (f0, f1)
}

/// Splits a circle polynomial evaluated on a [CircleDomain] of size `n` into two univariate
/// polynomials evaluated on a [LineDomain] of size `n/2`.
///
/// Let `evals` be the evaluations be of a polynomial `f` on the domain `E = +-c + <G>`. This
/// function returns the evaluations of `f0` and `f1` on the x-coordinates of `c + <G>` and
/// `-c - <G>` respectively such that `f(p) = (f0(px) + py * f1(px)) / 2`.
///
/// # Panics
///
/// Panics if there are less than two evaluations.
fn split_to_line<F: ExtensionOf<BaseField>>(
    evals: &CircleEvaluation<F>,
) -> (LineEvaluation<F>, LineEvaluation<F>) {
    let n = evals.len();
    assert!(n >= 2, "too few evals");

    let (l, r) = evals.split_at(n / 2);
    let (f0_evals, f1_evals) = zip(evals.domain.iter(), zip(l, r))
        .map(|(p, (&f_p, &f_neg_p))| {
            // Cauclate `f0(p_x)` and `f1(p_x)` such that `f(p) = (f0(p_x) + p_y * f1(p_x)) / 2`.
            let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
            ibutterfly(&mut f0_px, &mut f1_px, p.y.inverse());
            (f0_px, f1_px)
        })
        .unzip();

    // TODO: Either add domain to [LineEvaluation] or consider changing LineDomain to be defined by
    // its size and to always be canonic - since it may only be used by FRI.
    let f0 = LineEvaluation::new(f0_evals);
    let f1 = LineEvaluation::new(f1_evals);

    (f0, f1)
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

    use super::{FriConfig, FriProver};
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::core::circle::Coset;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::ExtensionOf;
    use crate::core::fri::{fold_circle_to_line, fold_line};
    use crate::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};
    use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};

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
        let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2() as usize));
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
        const BLOWUP_FACTOR_BITS: u32 = 2;
        const DEGREE_BITS: u32 = 4;
        let circle_evaluation = polynomial_evaluation(DEGREE_BITS, BLOWUP_FACTOR_BITS);
        let alpha = BaseField::one();

        let folded_evaluation = fold_circle_to_line(&circle_evaluation, alpha);

        assert_eq!(degree_bound_bits(folded_evaluation), DEGREE_BITS - 1);
    }

    #[test]
    #[should_panic]
    fn committing_high_degree_polynomial_fails() {
        const EXPECTED_BLOWUP_FACTOR_BITS: u32 = 2;
        const INVALID_BLOWUP_FACTOR_BITS: u32 = 1;
        let config = FriConfig::new(2, EXPECTED_BLOWUP_FACTOR_BITS);
        let prover = FriProver::<M31, Blake3Hasher>::new(config);
        let evaluation = polynomial_evaluation(6, INVALID_BLOWUP_FACTOR_BITS);

        prover.commit(vec![evaluation]);
    }

    #[test]
    #[ignore = "verification not implemented"]
    fn valid_fri_proof_passes_verification() {
        const BLOWUP_FACTOR_BITS: u32 = 2;
        let config = FriConfig::new(2, BLOWUP_FACTOR_BITS);
        let evaluation = polynomial_evaluation(6, BLOWUP_FACTOR_BITS);
        let prover = FriProver::<QM31, Blake3Hasher>::new(config);
        let prover = prover.commit(vec![evaluation]);
        let query_positions = [1, 8, 7];
        let _proof = prover.into_proof(&query_positions);

        todo!("verify proof");
    }

    #[test]
    #[ignore = "verification not implemented"]
    fn mixed_degree_fri_proof_passes_verification() {
        const BLOWUP_FACTOR_BITS: u32 = 2;
        let config = FriConfig::new(4, BLOWUP_FACTOR_BITS);
        let midex_degree_evals = vec![
            polynomial_evaluation(6, BLOWUP_FACTOR_BITS),
            polynomial_evaluation(4, BLOWUP_FACTOR_BITS),
            polynomial_evaluation(1, BLOWUP_FACTOR_BITS),
            polynomial_evaluation(0, BLOWUP_FACTOR_BITS),
        ];
        let prover = FriProver::<QM31, Blake3Hasher>::new(config);
        let prover = prover.commit(midex_degree_evals);
        let query_positions = [1, 8, 7];
        let _proof = prover.into_proof(&query_positions);

        todo!("verify proof");
    }

    #[test]
    #[ignore = "verification not implemented"]
    fn invalid_fri_proof_fails_verification() {
        todo!()
    }

    /// Returns an evaluation of a random polynomial with degree `2^degree_bits`.
    ///
    /// The evaluation domain size is `2^(degree_bits + blowup_factor_bits)`.
    fn polynomial_evaluation<F: ExtensionOf<BaseField>>(
        degree_bits: u32,
        blowup_factor_bits: u32,
    ) -> CircleEvaluation<F> {
        let poly = CirclePoly::new(degree_bits as usize, vec![F::one(); 1 << degree_bits]);
        let coset = Coset::half_odds((degree_bits + blowup_factor_bits - 1) as usize);
        let domain = CircleDomain::new(coset);
        poly.evaluate(domain)
    }

    /// Returns the degree bound of a polynomial as `log2(degree_bound)`.
    // TODO: move to test module
    fn degree_bound_bits<F: ExtensionOf<BaseField>>(polynomial: LineEvaluation<F>) -> u32 {
        let domain = LineDomain::new(Coset::half_odds(polynomial.len().ilog2() as usize));
        let coeffs = polynomial.interpolate(domain).into_natural_coefficients();
        let degree = coeffs.into_iter().rposition(|c| !c.is_zero()).unwrap_or(0);
        (degree + 1).ilog2()
    }
}
