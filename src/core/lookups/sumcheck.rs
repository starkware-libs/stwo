use std::iter::zip;

use num_traits::{One, Zero};
use thiserror::Error;

use super::utils::UnivariatePolynomial;
use crate::core::channel::Channel;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::horner_eval;

/// The sum-check protocol enables proving claims about the sum of a multivariate polynomial
/// `g(x_1, ..., x_n)` over the boolean hypercube. This trait provides methods for evaluating sums
/// and making transformations on `g` in the context of the protocol. It is indented to be used in
/// conjunction with [`prove()`] to generate proofs.
pub trait SumcheckOracle: Sized {
    /// Returns the number of variables in `g`.
    fn num_variables(&self) -> usize;

    /// Computes the sum of `g(x_1, x_2, ..., x_n)` over all `(x_2, ..., x_n)` in `{0, 1}^{n-1}`,
    /// effectively reducing the sum over `g` to a univariate polynomial in `x_1`.
    ///
    /// `claim` equals the claimed sum of `g(x_1, x_2, ..., x_n)` over all `(x_1, ..., x_n)` in
    /// `{0, 1}^{n}`. It is passed since knowing it can help optimize the implementation: Let `f`
    /// denote the univariate polynomial we want to return. Note that `claim = f(0) + f(1)` so
    /// knowing `claim` and either `f(0)` or `f(1)` allows us to immediately determine the other
    /// value with a single subtraction.
    fn univariate_sum(&self, claim: SecureField) -> UnivariatePolynomial<SecureField>;

    /// Returns a transformed oracle where the first variable of `g` is fixed to `challenge`.
    ///
    /// The returned oracle represents the multivariate polynomial `g'`, defined as
    /// `g'(x_2, ..., x_n) = g(challenge, x_2, ..., x_n)`.
    fn fix_first(self, challenge: SecureField) -> Self;
}

/// Generates a sum-check protocol proof.
///
/// The sum-check protocol enables proving claims about the sum of a multivariate polynomial
/// `g(x_1, ..., x_n)` over the boolean hypercube `{0, 1}^n`. Operations on `g` are abstracted by
/// [`SumcheckOracle`]. The proof, list of challenges (variable assignment) and the finalized oracle
/// (the oracle with all challenges applied to the variables) are returned.
///
/// Output is of the form: `(proof, variable_assignment, finalized_oracle, claimed_eval)`
pub fn prove<O: SumcheckOracle>(
    mut claim: SecureField,
    mut oracle: O,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>, O, SecureField) {
    let mut round_polynomials = Vec::new();
    let mut assignment = Vec::new();

    for _round in 0..oracle.num_variables() {
        let round_polynomial = oracle.univariate_sum(claim);
        channel.mix_felts(&round_polynomial);

        let challenge = channel.draw_felt();
        oracle = oracle.fix_first(challenge);
        claim = round_polynomial.eval_at_point(challenge);
        round_polynomials.push(round_polynomial);
        assignment.push(challenge);
    }

    let proof = SumcheckProof { round_polynomials };

    (proof, assignment, oracle, claim)
}

pub fn prove_batch<O: SumcheckOracle>(
    mut claims: Vec<SecureField>,
    mut oracles: Vec<O>,
    lambda: SecureField,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>, Vec<O>, Vec<SecureField>) {
    let num_variables = oracles[0].num_variables();
    assert!(oracles.iter().all(|o| o.num_variables() == num_variables));
    assert_eq!(claims.len(), oracles.len());

    let mut round_polynomials = Vec::new();
    let mut assignment = Vec::new();

    for _round in 0..num_variables {
        let round_polys = zip(&oracles, &claims)
            .map(|(oracle, &claim)| oracle.univariate_sum(claim))
            .collect::<Vec<UnivariatePolynomial<SecureField>>>();

        let round_polynomial = random_linear_combination(&round_polys, lambda);

        assert_eq!(
            round_polynomial.eval_at_point(SecureField::zero())
                + round_polynomial.eval_at_point(SecureField::one()),
            horner_eval(&claims, lambda)
        );

        channel.mix_felts(&round_polynomial);

        let challenge = channel.draw_felt();

        oracles = oracles
            .into_iter()
            .map(|oracle| oracle.fix_first(challenge))
            .collect();

        claims = round_polys
            .iter()
            .map(|round_poly| round_poly.eval_at_point(challenge))
            .collect();

        round_polynomials.push(round_polynomial);
        assignment.push(challenge);
    }

    let proof = SumcheckProof { round_polynomials };

    (proof, assignment, oracles, claims)
}

fn random_linear_combination(
    polynomials: &[UnivariatePolynomial<SecureField>],
    lambda: SecureField,
) -> UnivariatePolynomial<SecureField> {
    polynomials
        .iter()
        .rfold(Zero::zero(), |acc, poly| acc * lambda + poly.clone())
}

/// Partially verifies the sum-check protocol by validating the provided proof against the claim.
///
/// Only "partial" since it does not fully verify the prover's claimed evaluation on the variable
/// assignment but checks if the sum of the round polynomials evaluated on `0` and `1` matches the
/// claim for each round. If the proof passes these checks, the variable assignment and the prover's
/// claimed evaluation are returned for the caller to validate otherwise an [`Err`] is returned.
///
/// Output is of the form `(variable_assignment, claimed_eval)`.
pub fn partially_verify(
    mut claim: SecureField,
    proof: &SumcheckProof,
    channel: &mut impl Channel,
) -> Result<(Vec<SecureField>, SecureField), SumcheckError> {
    let mut assignment = Vec::new();

    for (round, round_polynomial) in proof.round_polynomials.iter().enumerate() {
        if round_polynomial.degree() > MAX_DEGREE {
            return Err(SumcheckError::DegreeInvalid { round });
        }

        let sum = round_polynomial.eval_at_point(SecureField::zero())
            + round_polynomial.eval_at_point(SecureField::one());

        if claim != sum {
            return Err(SumcheckError::SumInvalid { claim, sum, round });
        }

        channel.mix_felts(round_polynomial);
        let challenge = channel.draw_felt();
        claim = round_polynomial.eval_at_point(challenge);
        assignment.push(challenge);
    }

    Ok((assignment, claim))
}

#[derive(Debug, Clone)]
pub struct SumcheckProof {
    pub round_polynomials: Vec<UnivariatePolynomial<SecureField>>,
}

/// Max degree of polynomials the verifier accepts in each round of the protocol.
const MAX_DEGREE: usize = 3;

/// Sum-check protocol verification error.
///
/// Round 0 corresponds to the first round.
#[derive(Error, Debug)]
pub enum SumcheckError {
    #[error("degree of the polynomial in round {round} is too high")]
    DegreeInvalid { round: usize },
    #[error("sum does not match the claim in round {round} (sum {sum}, claim {claim})")]
    SumInvalid {
        claim: SecureField,
        sum: SecureField,
        round: usize,
    },
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::{partially_verify, prove};
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::CPUBackend;
    // use crate::core::backend::avx512::AVX512Backend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::mle::Mle;

    #[test]
    fn cpu_sumcheck_works() {
        let values = test_channel().draw_felts(1 << 9);
        let claim = values.iter().sum();
        let mle = Mle::<CPUBackend, SecureField>::new(values.clone());
        let (proof, ..) = prove(claim, mle.clone(), &mut test_channel());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, mle.eval_at_point(&assignment));
    }

    #[test]
    fn simd_sumcheck_works() {
        let values = test_channel().draw_felts(1 << 9);
        let claim = values.iter().sum();
        let mle = Mle::<SimdBackend, SecureField>::new(values.iter().copied().collect());
        let (proof, ..) = prove(claim, mle.clone(), &mut test_channel());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, mle.eval_at_point(&assignment));
    }

    #[test]
    fn invalid_sumcheck_proof_fails() {
        let values = test_channel().draw_felts(8);
        let claim = values.iter().sum::<SecureField>();
        // Compromise the first value.
        let mut invalid_values = values.to_vec();
        invalid_values[0] += SecureField::one();
        let invalid_claim = invalid_values.iter().sum::<SecureField>();
        let invalid_mle = Mle::<CPUBackend, SecureField>::new(invalid_values.clone());
        let (invalid_proof, ..) = prove(invalid_claim, invalid_mle, &mut test_channel());

        assert!(partially_verify(claim, &invalid_proof, &mut test_channel()).is_err());
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}

// g(x) = sum_y eq(x, y) * p(x, 0) * p(x, 1)
// g(x) = sum_y eq(x, y) * p(x, 0) * p(x, 1)
// g(x) = sum_y eq(x, y) * (Z - t0(x)) * (Z - t1(y))
