use std::iter::zip;

use num_traits::{One, Zero};
use thiserror::Error;

use super::utils::UnivariatePoly;
use crate::core::channel::Channel;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::horner_eval;

/// The sum-check protocol enables proving claims about the sum of a multivariate polynomial
/// `g(x_1, ..., x_n)` over the boolean hypercube. This trait provides methods for evaluating sums
/// and making transformations on `g` in the context of the protocol. It is indented to be used in
/// conjunction with [`prove_batch()`] to generate proofs.
pub trait MultivariatePolyOracle: Sized {
    /// Returns the number of variables in `g`.
    fn n_variables(&self) -> usize;

    /// Computes the sum of `g(x_1, x_2, ..., x_n)` over all `(x_2, ..., x_n)` in `{0, 1}^{n-1}`,
    /// effectively reducing the sum over `g` to a univariate polynomial in `x_1`.
    ///
    /// `claim` equals the claimed sum of `g(x_1, x_2, ..., x_n)` over all `(x_1, ..., x_n)` in
    /// `{0, 1}^{n}`. It is passed since knowing it can help optimize the implementation: Let `f`
    /// denote the univariate polynomial we want to return. Note that `claim = f(0) + f(1)` so
    /// knowing `claim` and either `f(0)` or `f(1)` allows us to determine the other.
    fn sum_as_poly_in_first_variable(&self, claim: SecureField) -> UnivariatePoly<SecureField>;

    /// Returns a transformed oracle where the first variable of `g` is fixed to `challenge`.
    ///
    /// The returned oracle represents the multivariate polynomial `g'`, defined as
    /// `g'(x_2, ..., x_n) = g(challenge, x_2, ..., x_n)`.
    fn fix_first_variable(self, challenge: SecureField) -> Self;
}

/// Performs sum-check on a random linear combinations of multiple multivariate polynomials.
///
/// Let the multivariate polynomials be `g_1`, ..., `g_n`. A single sumcheck is performed on
/// multivariate polynomial `h = g_1 + lambda * g_2 + ... + lambda^(n-1) * g_n`. The degree of each
/// `g_i` should not exceed [`MAX_DEGREE`] in any variable.  The sumcheck proof of `h`, list of
/// challenges (variable assignment) and the finalized oracles (i.e. the `g_i` with all variables
/// fixed to the challenges) are returned.
///
/// Output is of the form: `(proof, variable_assignment, finalized_oracles, claimed_evals)`
///
/// # Panics
///
/// Panics if:
/// - No multivariate polynomials are provided.
/// - The input multivariate polynomials don't have the same number of variables.
/// - There aren't the same number of multivariate polynomials and claims.
/// - The degree of the any multivariate polynomial exceeds [`MAX_DEGREE`] in any variable.
/// - The round polynomials are inconsistent with their corresponding claimed sum on `0` and `1`.
pub fn prove_batch<O: MultivariatePolyOracle>(
    mut claims: Vec<SecureField>,
    mut multivariate_polys: Vec<O>,
    lambda: SecureField,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>, Vec<O>, Vec<SecureField>) {
    let n_vars = multivariate_polys[0].n_variables();
    assert!(multivariate_polys.iter().all(|o| o.n_variables() == n_vars));
    assert_eq!(claims.len(), multivariate_polys.len());

    let mut round_polys = Vec::new();
    let mut assignment = Vec::new();

    for _round in 0..n_vars {
        let this_round_polys = zip(&multivariate_polys, &claims)
            .map(|(mvp, &claim)| mvp.sum_as_poly_in_first_variable(claim))
            .collect::<Vec<UnivariatePoly<SecureField>>>();

        let round_poly = random_linear_combination(&this_round_polys, lambda);

        assert!(round_poly.degree() <= MAX_DEGREE,);

        assert_eq!(
            round_poly.eval_at_point(Zero::zero()) + round_poly.eval_at_point(One::one()),
            horner_eval(&claims, lambda),
        );

        channel.mix_felts(&round_poly);

        let challenge = channel.draw_felt();

        multivariate_polys = multivariate_polys
            .into_iter()
            .map(|mvp| mvp.fix_first_variable(challenge))
            .collect();

        claims = this_round_polys
            .iter()
            .map(|round_poly| round_poly.eval_at_point(challenge))
            .collect();

        round_polys.push(round_poly);
        assignment.push(challenge);
    }

    let proof = SumcheckProof { round_polys };

    (proof, assignment, multivariate_polys, claims)
}

fn random_linear_combination(
    polys: &[UnivariatePoly<SecureField>],
    lambda: SecureField,
) -> UnivariatePoly<SecureField> {
    polys
        .iter()
        .rfold(Zero::zero(), |acc, poly| acc * lambda + poly.clone())
}

/// Partially verifies a sum-check proof.
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

    for (round, round_poly) in proof.round_polys.iter().enumerate() {
        if round_poly.degree() > MAX_DEGREE {
            return Err(SumcheckError::DegreeInvalid { round });
        }

        // TODO: optimize this by sending one less coefficient, and computing it from the
        // claim, instead of checking the claim. (Can also be done by quotienting).
        let sum = round_poly.eval_at_point(Zero::zero()) + round_poly.eval_at_point(One::one());

        if claim != sum {
            return Err(SumcheckError::SumInvalid { claim, sum, round });
        }

        channel.mix_felts(round_poly);
        let challenge = channel.draw_felt();
        claim = round_poly.eval_at_point(challenge);
        assignment.push(challenge);
    }

    Ok((assignment, claim))
}

#[derive(Debug, Clone)]
pub struct SumcheckProof {
    pub round_polys: Vec<UnivariatePoly<SecureField>>,
}

/// Max degree of polynomials the verifier accepts in each round of the protocol.
pub const MAX_DEGREE: usize = 3;

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

    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::backend::CPUBackend;
    // use crate::core::backend::avx512::AVX512Backend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::sumcheck::{partially_verify, prove_batch};

    #[test]
    fn sumcheck_works() {
        let values = test_channel().draw_felts(32);
        let claim = values.iter().sum();
        let mle = Mle::<CPUBackend, SecureField>::new(values);
        let lambda = SecureField::one();
        let (proof, ..) = prove_batch(vec![claim], vec![mle.clone()], lambda, &mut test_channel());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, mle.eval_at_point(&assignment));
    }

    #[test]
    fn batch_sumcheck_works() {
        let mut channel = test_channel();
        let values0 = channel.draw_felts(32);
        let values1 = channel.draw_felts(32);
        let claim0 = values0.iter().sum();
        let claim1 = values1.iter().sum();
        let mle0 = Mle::<CPUBackend, SecureField>::new(values0.clone());
        let mle1 = Mle::<CPUBackend, SecureField>::new(values1.clone());
        let lambda = channel.draw_felt();
        let (proof, ..) = prove_batch(
            vec![claim0, claim1],
            vec![mle0.clone(), mle1.clone()],
            lambda,
            &mut test_channel(),
        );

        let (assignment, eval) =
            partially_verify(claim0 + lambda * claim1, &proof, &mut test_channel()).unwrap();

        let eval0 = mle0.eval_at_point(&assignment);
        let eval1 = mle1.eval_at_point(&assignment);
        assert_eq!(eval, eval0 + lambda * eval1);
    }

    #[test]
    fn invalid_sumcheck_proof_fails() {
        let values = test_channel().draw_felts(8);
        let claim = values.iter().sum::<SecureField>();
        let lambda = SecureField::one();
        // Compromise the first value.
        let mut invalid_values = values;
        invalid_values[0] += SecureField::one();
        let invalid_claim = invalid_values.iter().sum::<SecureField>();
        let invalid_mle = Mle::<CPUBackend, SecureField>::new(invalid_values.clone());
        let (invalid_proof, ..) = prove_batch(
            vec![invalid_claim],
            vec![invalid_mle],
            lambda,
            &mut test_channel(),
        );

        assert!(partially_verify(claim, &invalid_proof, &mut test_channel()).is_err());
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
