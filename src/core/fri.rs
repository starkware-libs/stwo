use std::cmp::Reverse;
use std::fmt::Debug;
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::RangeInclusive;

use super::fields::m31::BaseField;
use super::fields::{ExtensionOf, Field};
use super::poly::line::{LineEvaluation, LinePoly};
use crate::commitment_scheme::hasher::ComplexHasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
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
pub struct FriProver<F: ExtensionOf<BaseField>, H: ComplexHasher, Phase = CommitmentPhase> {
    config: FriConfig,
    layers: Vec<FriLayer<F, H>>,
    remainder: Option<LinePoly<F>>,
    _phase: PhantomData<Phase>,
}

impl<F: ExtensionOf<BaseField>, H: ComplexHasher> FriProver<F, H, CommitmentPhase> {
    /// Creates a new FRI prover.
    pub fn new(config: FriConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
            remainder: None,
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
    /// * An evaluation domain is smaller than or equal to the max last layer domain size.
    fn commit_inner_layers(&mut self, mut evals: Vec<LineEvaluation<F>>) -> LineEvaluation<F> {
        let mut evaluation = evals.pop().expect("require at least one evaluation");
        while evaluation.len() > self.config.max_last_layer_domain_size() {
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
            evaluation = apply_drp(&evaluation, alpha);
            self.layers.push(layer)
        }

        assert!(evals.is_empty());
        evaluation
    }

    /// Builds and commits to the FRI remainder polynomial's coefficients (the last FRI layer).
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The domain size of the evaluation exceeds the max last layer domain size.
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

impl<F: ExtensionOf<BaseField>, H: ComplexHasher> FriProver<F, H, QueryPhase> {
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
pub struct FriProof<F: ExtensionOf<BaseField>, H: ComplexHasher> {
    pub layer_proofs: Vec<FriLayerProof<F, H>>,
    pub remainder: LinePoly<F>,
}

const FRI_STEP_SIZE: usize = 2;

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
///
/// The subset corresponds to the set of evaluations needed by a FRI verifier.
pub struct FriLayerProof<F: ExtensionOf<BaseField>, H: ComplexHasher> {
    pub coset_evals: Vec<[F; FRI_STEP_SIZE]>,
    pub decommitment: MerkleDecommitment<F, H>,
    pub commitment: H::Hash,
}

impl<F: ExtensionOf<BaseField>, H: ComplexHasher> FriLayerProof<F, H> {
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
struct FriLayer<F: ExtensionOf<BaseField>, H: ComplexHasher> {
    /// Coset evaluations stored in column-major.
    coset_evals: [Vec<F>; FRI_STEP_SIZE],
    _merkle_tree: MerkleTree<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: ComplexHasher> FriLayer<F, H> {
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

/// Performs a degree respecting projection (DRP) on a polynomial.
///
/// Example: Our evaluation domain is the x-coordinates of `E = c + <G>`, `alpha` is a random field
/// element and `pi(x) = 2x^2 - 1` is the circle's x-coordinate doubling map. We have evaluations of
/// a polynomial `f` on `E` (i.e `evals`) and we can compute the evaluations of `f' = 2 * (fe +
/// alpha * fo)` over `E' = { pi(x) | x in E }` such that `f(x) = fe(pi(x)) + x * fo(pi(x))`.
///
/// `evals` should be polynomial evaluations over a [LineDomain] stored in natural order. The return
/// evaluations are evaluations over a [LineDomain] of half the size stored in natural order.
///
/// # Panics
///
/// Panics if the number of evaluations is not greater than or equal to two.
pub fn apply_drp<F: ExtensionOf<BaseField>>(
    evals: &LineEvaluation<F>,
    alpha: F,
) -> LineEvaluation<F> {
    let n = evals.len();
    assert!(n >= 2);
    let (l, r) = evals.split_at(n / 2);
    let domain = LineDomain::new(Coset::half_odds(n.ilog2() as usize));
    let drp_evals = zip(zip(l, r), domain.iter())
        .map(|((&f_x, &f_neg_x), x)| {
            let (mut f_e, mut f_o) = (f_x, f_neg_x);
            ibutterfly(&mut f_e, &mut f_o, x.inverse());
            f_e + alpha * f_o
        })
        .collect();
    LineEvaluation::new(drp_evals)
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

    use super::{FriConfig, FriProver};
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::core::circle::Coset;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::ExtensionOf;
    use crate::core::fri::apply_drp;
    use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};

    #[test]
    fn drp_works() {
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

        let drp_evals = apply_drp(&evals, alpha);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        for (i, (&drp_eval, x)) in zip(&*drp_evals, drp_domain.iter()).enumerate() {
            let f_e = even_poly.eval_at_point(x);
            let f_o = odd_poly.eval_at_point(x);
            assert_eq!(drp_eval, two * (f_e + alpha * f_o), "mismatch at {i}");
        }
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
    ) -> LineEvaluation<F> {
        let poly = LinePoly::new(vec![F::one(); 1 << degree_bits]);
        let coset = Coset::half_odds((degree_bits + blowup_factor_bits) as usize);
        let domain = LineDomain::new(coset);
        poly.evaluate(domain)
    }
}
