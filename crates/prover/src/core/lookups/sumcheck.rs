//! Sum-check protocol that proves and verifies claims about `sum_x g(x)` for all x in `{0, 1}^n`.
//!
//! [`MultivariatePolyOracle`] provides methods for evaluating sums and making transformations on
//! `g` in the context of the protocol. It is intended to be used in conjunction with
//! [`prove_batch()`] to generate proofs.

use std::iter::zip;

use itertools::Itertools;
use num_traits::{One, Zero};
use thiserror::Error;

use super::utils::UnivariatePoly;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;

/// Something that can be seen as a multivariate polynomial `g(x_0, ..., x_{n-1})`.
pub trait MultivariatePolyOracle: Sized {
    /// Returns the number of variables in `g`.
    fn n_variables(&self) -> usize;

    /// Computes the sum of `g(x_0, x_1, ..., x_{n-1})` over all `(x_1, ..., x_{n-1})` in
    /// `{0, 1}^(n-1)`, effectively reducing the sum over `g` to a univariate polynomial in `x_0`.
    ///
    /// `claim` equals the claimed sum of `g(x_0, x_2, ..., x_{n-1})` over all `(x_0, ..., x_{n-1})`
    /// in `{0, 1}^n`. Knowing the claim can help optimize the implementation: Let `f` denote the
    /// univariate polynomial we want to return. Note that `claim = f(0) + f(1)` so knowing `claim`
    /// and either `f(0)` or `f(1)` allows determining the other.
    fn sum_as_poly_in_first_variable(&self, claim: SecureField) -> UnivariatePoly<SecureField>;

    /// Returns a transformed oracle where the first variable of `g` is fixed to `challenge`.
    ///
    /// The returned oracle represents the multivariate polynomial `g'`, defined as
    /// `g'(x_1, ..., x_{n-1}) = g(challenge, x_1, ..., x_{n-1})`.
    fn fix_first_variable(self, challenge: SecureField) -> Self;
}

/// Performs sum-check on a random linear combinations of multiple multivariate polynomials.
///
/// Let the multivariate polynomials be `g_0, ..., g_{n-1}`. A single sum-check is performed on
/// multivariate polynomial `h = g_0 + lambda * g_1 + ... + lambda^(n-1) * g_{n-1}`. The `g_i`s do
/// not need to have the same number of variables. `g_i`s with less variables are folded in the
/// latest possible round of the protocol. For instance with `g_0(x, y, z)` and `g_1(x, y)`
/// sum-check is performed on `h(x, y, z) = g_0(x, y, z) + lambda * g_1(y, z)`. Claim `c_i` should
/// equal the claimed sum of `g_i(x_0, ..., x_{j-1})` over all `(x_0, ..., x_{j-1})` in `{0, 1}^j`.
///
/// The degree of each `g_i` should not exceed [`MAX_DEGREE`] in any variable.  The sum-check proof
/// of `h`, list of challenges (variable assignment) and the constant oracles (i.e. the `g_i` with
/// all variables fixed to the their corresponding challenges) are returned.
///
/// Output is of the form: `(proof, variable_assignment, constant_poly_oracles, claimed_evals)`
///
/// # Panics
///
/// Panics if:
/// - No multivariate polynomials are provided.
/// - There aren't the same number of multivariate polynomials and claims.
/// - The degree of any multivariate polynomial exceeds [`MAX_DEGREE`] in any variable.
/// - The round polynomials are inconsistent with their corresponding claimed sum on `0` and `1`.
// TODO: Consider returning constant oracles as separate type.
pub fn prove_batch<O: MultivariatePolyOracle>(
    mut claims: Vec<SecureField>,
    mut multivariate_polys: Vec<O>,
    lambda: SecureField,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>, Vec<O>, Vec<SecureField>) {
    let n_variables = multivariate_polys.iter().map(O::n_variables).max().unwrap();
    assert_eq!(claims.len(), multivariate_polys.len());

    let mut round_polys = Vec::new();
    let mut assignment = Vec::new();

    // Update the claims for the sum over `h`'s hypercube.
    for (claim, multivariate_poly) in zip(&mut claims, &multivariate_polys) {
        let n_unused_variables = n_variables - multivariate_poly.n_variables();
        *claim *= BaseField::from(1 << n_unused_variables);
    }

    // Prove sum-check rounds
    for round in 0..n_variables {
        let n_remaining_rounds = n_variables - round;

        let this_round_polys = zip(&multivariate_polys, &claims)
            .enumerate()
            .map(|(i, (multivariate_poly, &claim))| {
                let round_poly = if n_remaining_rounds == multivariate_poly.n_variables() {
                    multivariate_poly.sum_as_poly_in_first_variable(claim)
                } else {
                    (claim / BaseField::from(2)).into()
                };

                let eval_at_0 = round_poly.eval_at_point(SecureField::zero());
                let eval_at_1 = round_poly.eval_at_point(SecureField::one());
                assert_eq!(eval_at_0 + eval_at_1, claim, "i={i}, round={round}");
                assert!(round_poly.degree() <= MAX_DEGREE, "i={i}, round={round}");

                round_poly
            })
            .collect_vec();

        let round_poly = random_linear_combination(&this_round_polys, lambda);

        channel.mix_felts(&round_poly);

        let challenge = channel.draw_felt();

        claims = this_round_polys
            .iter()
            .map(|round_poly| round_poly.eval_at_point(challenge))
            .collect();

        multivariate_polys = multivariate_polys
            .into_iter()
            .map(|multivariate_poly| {
                if n_remaining_rounds != multivariate_poly.n_variables() {
                    return multivariate_poly;
                }

                multivariate_poly.fix_first_variable(challenge)
            })
            .collect();

        round_polys.push(round_poly);
        assignment.push(challenge);
    }

    let proof = SumcheckProof { round_polys };

    (proof, assignment, multivariate_polys, claims)
}

/// Returns `p_0 + alpha * p_1 + ... + alpha^(n-1) * p_{n-1}`.
fn random_linear_combination(
    polys: &[UnivariatePoly<SecureField>],
    alpha: SecureField,
) -> UnivariatePoly<SecureField> {
    polys
        .iter()
        .rfold(Zero::zero(), |acc, poly| acc * alpha + poly.clone())
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
#[derive(Error, Debug)]
pub enum SumcheckError {
    #[error("degree of the polynomial in round {round} is too high")]
    DegreeInvalid { round: RoundIndex },
    #[error("sum does not match the claim in round {round} (sum {sum}, claim {claim})")]
    SumInvalid {
        claim: SecureField,
        sum: SecureField,
        round: RoundIndex,
    },
}

/// Sum-check round index where 0 corresponds to the first round.
pub type RoundIndex = usize;

#[cfg(test)]
mod tests {

    use num_traits::One;

    use crate::core::backend::CpuBackend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::Field;
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::sumcheck::{partially_verify, prove_batch};

    #[test]
    fn sumcheck_works() {
        let values = test_channel().draw_felts(32);
        let claim = values.iter().sum();
        let mle = Mle::<CpuBackend, SecureField>::new(values);
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
        let mle0 = Mle::<CpuBackend, SecureField>::new(values0.clone());
        let mle1 = Mle::<CpuBackend, SecureField>::new(values1.clone());
        let lambda = channel.draw_felt();
        let claims = vec![claim0, claim1];
        let mles = vec![mle0.clone(), mle1.clone()];
        let (proof, ..) = prove_batch(claims, mles, lambda, &mut test_channel());

        let claim = claim0 + lambda * claim1;
        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        let eval0 = mle0.eval_at_point(&assignment);
        let eval1 = mle1.eval_at_point(&assignment);
        assert_eq!(eval, eval0 + lambda * eval1);
    }

    #[test]
    fn batch_sumcheck_with_different_n_variables() {
        let mut channel = test_channel();
        let values0 = channel.draw_felts(64);
        let values1 = channel.draw_felts(32);
        let claim0 = values0.iter().sum();
        let claim1 = values1.iter().sum();
        let mle0 = Mle::<CpuBackend, SecureField>::new(values0.clone());
        let mle1 = Mle::<CpuBackend, SecureField>::new(values1.clone());
        let lambda = channel.draw_felt();
        let claims = vec![claim0, claim1];
        let mles = vec![mle0.clone(), mle1.clone()];
        let (proof, ..) = prove_batch(claims, mles, lambda, &mut test_channel());

        let claim = claim0 + lambda * claim1.double();
        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        let eval0 = mle0.eval_at_point(&assignment);
        let eval1 = mle1.eval_at_point(&assignment[1..]);
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
        let invalid_claim = vec![invalid_values.iter().sum::<SecureField>()];
        let invalid_mle = vec![Mle::<CpuBackend, SecureField>::new(invalid_values.clone())];
        let (invalid_proof, ..) =
            prove_batch(invalid_claim, invalid_mle, lambda, &mut test_channel());

        assert!(partially_verify(claim, &invalid_proof, &mut test_channel()).is_err());
    }

    fn test_channel() -> Blake2sChannel {
        Blake2sChannel::default()
    }
}
