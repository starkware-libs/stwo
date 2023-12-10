use std::cmp::Reverse;
use std::fmt::Debug;
use std::iter::zip;
use std::marker::PhantomData;

use super::fields::m31::BaseField;
use super::fields::{ExtensionOf, Field};
use super::poly::line::{LineEvaluation, LinePoly};
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::poly::line::LineDomain;

/// FRI proof options
// TODO(andrew): support different folding factors
#[derive(Debug, Clone, Copy)]
pub struct FriOptions {
    max_remainder_coeffs_bits: u32,
    blowup_factor_bits: u32,
}

impl FriOptions {
    /// Creates new FRI options.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `max_remainder_coeffs_bits` is greater than six.
    /// * `blowup_factor_bits` is equal to zero or greater than four.
    pub fn new(max_remainder_coeffs_bits: u32, blowup_factor_bits: u32) -> Self {
        assert!(max_remainder_coeffs_bits <= 6);
        assert!(blowup_factor_bits != 0 && blowup_factor_bits <= 4);
        Self {
            max_remainder_coeffs_bits,
            blowup_factor_bits,
        }
    }

    fn max_remainder_domain_size(&self) -> usize {
        1 << (self.max_remainder_coeffs_bits + self.blowup_factor_bits)
    }
}

/// Commitment phase for [FriProver].
pub struct Commitment;

/// Query phase for [FriProver].
pub struct Query;

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
///
/// `Phase` is used for enforce the commitment phase is done before the query phase.
pub struct FriProver<F: ExtensionOf<BaseField>, H: Hasher, Phase = Commitment> {
    options: FriOptions,
    layers: Vec<FriLayer<F, H>>,
    remainder: Option<LinePoly<F>>,
    _phase: PhantomData<Phase>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriProver<F, H, Commitment> {
    pub fn new(options: FriOptions) -> Self {
        Self {
            options,
            layers: Vec::new(),
            remainder: None,
            _phase: PhantomData,
        }
    }

    /// Builds FRI layers from multiple [LineEvaluation]s.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `evals` is empty.
    /// * Each evaluation is not of sufficiently low degree.
    pub fn build_layers(mut self, mut evals: Vec<LineEvaluation<F>>) -> FriProver<F, H, Query> {
        // Sort in ascending order by evaluation domain size.
        evals.sort_by_key(|eval| Reverse(eval.len()));

        // build FRI layers
        let mut evaluation = evals.pop().expect("require at least one evaluation");
        while evaluation.len() > self.options.max_remainder_domain_size() {
            // Aggregate all evaluations that have the same domain size.
            while let Some(true) = evals.last().map(|e| e.len() == evaluation.len()) {
                for (i, &eval) in evals.pop().unwrap().iter().enumerate() {
                    evaluation[i] += eval;
                }
            }

            let layer = FriLayer::new(&evaluation);
            // TODO(andrew): add merkle root to channel
            let _merkle_root = layer.merkle_tree.root();
            // TODO(andrew): draw random alpha from channel
            let alpha = F::one();
            evaluation = apply_drp(&evaluation, alpha);
            self.layers.push(layer)
        }

        // Add our folded evaluation to the set of evaluations.
        evals.push(evaluation);

        self.set_remainder(evals)
    }

    /// Obtains the coefficients of the FRI remainder polynomial (i.e. the last FRI layer).
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `evals` is empty.
    /// * The domain size of an evaluation exceeds the maximum remainder domain size.
    /// * Each evaluation is not sufficiently low degree.
    fn set_remainder(self, mut evals: Vec<LineEvaluation<F>>) -> FriProver<F, H, Query> {
        let Self {
            options, layers, ..
        } = self;

        // Sort in ascending order by evaluation domain size.
        evals.sort_by_key(|eval| Reverse(eval.len()));

        let largest_domain_size = evals.last().expect("require at least one evaluation").len();
        assert!(largest_domain_size <= options.max_remainder_domain_size());
        let num_remainder_coeffs = largest_domain_size >> options.blowup_factor_bits;
        let mut remainder_coeffs = vec![F::zero(); num_remainder_coeffs];

        // Aggregate all polynomials into the remainder.
        for evaluation in evals {
            let domain = LineDomain::new(Coset::half_odds(evaluation.len().ilog2() as usize));
            let expected_num_coeffs = evaluation.len() >> options.blowup_factor_bits;
            let mut coeffs = bit_reverse(evaluation.interpolate(domain).into_coefficients());
            let zeros = coeffs.split_off(expected_num_coeffs);
            assert!(zeros.iter().all(F::is_zero), "invalid degree");
            zip(&mut remainder_coeffs, coeffs).for_each(|(rem_coeff, coeff)| *rem_coeff += coeff);
        }

        // Reorder coefficients to create a [LinePoly].
        let remainder = Some(LinePoly::new(bit_reverse(remainder_coeffs)));

        FriProver {
            options,
            layers,
            remainder,
            _phase: PhantomData,
        }
    }
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriProver<F, H, Query> {
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

/// A FRI proof.
pub struct FriProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub layer_proofs: Vec<FriLayerProof<F, H>>,
    pub remainder: LinePoly<F>,
}

/// Stores a subset of evaluations in a [FriLayer] with their corresponding merkle decommitments.
///
/// The subset corresponds to the set of evaluations needed by a FRI verifier.
pub struct FriLayerProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub coset_evals: Vec<[F; 2]>,
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
    coset_evals: [Vec<F>; 2],
    merkle_tree: MerkleTree<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> FriLayer<F, H> {
    fn new(evaluation: &LineEvaluation<F>) -> Self {
        let (l, r) = evaluation.split_at(evaluation.len() / 2);
        let coset_evals = [l.to_vec(), r.to_vec()];
        let merkle_tree = MerkleTree::commit(coset_evals.to_vec());
        FriLayer {
            coset_evals,
            merkle_tree,
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
        let position_set = positions.iter().copied().collect();
        let decommitment = self.merkle_tree.generate_decommitment(position_set);
        let commitment = self.merkle_tree.root();
        FriLayerProof {
            coset_evals,
            decommitment,
            commitment,
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
fn apply_drp<F: ExtensionOf<BaseField>>(evals: &LineEvaluation<F>, alpha: F) -> LineEvaluation<F> {
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

/// Bit reverses a slice.
///
/// # Panics
///
/// Panics if the length of the slice is not a power of two.
fn bit_reverse<T, U: AsMut<[T]>>(mut v: U) -> U {
    let n = v.as_mut().len();
    assert!(n.is_power_of_two());
    let n_bits = n.ilog2();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - n_bits);
        if j > i {
            v.as_mut().swap(i, j);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::{FriOptions, FriProver};
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
    fn fri_prover_with_high_degree_polynomial_fails() {
        const EXPECTED_BLOWUP_FACTOR_BITS: u32 = 2;
        const INVALID_BLOWUP_FACTOR_BITS: u32 = 1;
        let options = FriOptions::new(2, EXPECTED_BLOWUP_FACTOR_BITS);
        let prover = FriProver::<M31, Blake3Hasher>::new(options);
        let evaluation = polynomial_evaluation(6, INVALID_BLOWUP_FACTOR_BITS);

        prover.build_layers(vec![evaluation]);
    }

    #[test]
    #[ignore = "verification not implemented"]
    fn valid_fri_proof_passes_verification() {
        const BLOWUP_FACTOR_BITS: u32 = 2;
        let options = FriOptions::new(2, BLOWUP_FACTOR_BITS);
        let evaluation = polynomial_evaluation(6, BLOWUP_FACTOR_BITS);
        let prover = FriProver::<QM31, Blake3Hasher>::new(options);
        let prover = prover.build_layers(vec![evaluation]);
        let query_positions = [1, 8, 7];
        let _proof = prover.into_proof(&query_positions);

        todo!("verify proof");
    }

    #[test]
    #[ignore = "verification not implemented"]
    fn mixed_degree_fri_proof_passes_verification() {
        const BLOWUP_FACTOR_BITS: u32 = 2;
        let options = FriOptions::new(4, BLOWUP_FACTOR_BITS);
        let midex_degree_evals = vec![
            polynomial_evaluation(6, BLOWUP_FACTOR_BITS),
            polynomial_evaluation(4, BLOWUP_FACTOR_BITS),
            polynomial_evaluation(1, BLOWUP_FACTOR_BITS),
            polynomial_evaluation(0, BLOWUP_FACTOR_BITS),
        ];
        let prover = FriProver::<QM31, Blake3Hasher>::new(options);
        let prover = prover.build_layers(midex_degree_evals);
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
