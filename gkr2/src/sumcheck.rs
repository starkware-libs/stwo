use num_traits::{One, Zero};
use prover_research::core::channel::Channel;
use prover_research::core::fields::qm31::SecureField;
use thiserror::Error;

use crate::utils::Polynomial;

/// The sum-check protocol enables proving claims about the sum of a multivariate polynomial
/// `g(x_1, ..., x_n)` over the boolean hypercube `{0, 1}^n`. This trait provides methods for
/// evaluating sums and making transformations on `g` in the context of the protocol. It is
/// indented to be used in conjunction with [`prove()`] to generate proofs.
pub trait SumcheckOracle: Sized {
    /// Returns the number of variables in `g`.
    fn num_variables(&self) -> usize;

    /// Computes the sum of `g(x_1, x_2, ..., x_n)` over all possible values `(x_2, ..., x_n)` in
    /// `{0, 1}^{n-1}`, effectively reducing the sum over `g` to a univariate polynomial in `x_1`.
    ///
    /// `claim` is a constant that equals the claimed sum of `g(x_1, x_2, ..., x_n)` over all
    /// possible values `(x_1, ..., x_n)` in `{0, 1}^{n}`. Knowing `claim` can help optimize the
    /// implementation: Let `f` denote the univariate polynomial we want to return. Our goal is
    /// obtaining the coefficients of `f` in the most optimal way. Assume that `deg(f) = 1`,
    /// therefore we need at least two evaluations to obtain `f`'s coefficients via interpolation.
    /// We could choose to evaluate `f(0)` and then we can use the claim to get `f(1)` using only
    /// only a single subtraction due to the fact `f(0) + f(1) = claim`.
    fn univariate_sum(&self, claim: SecureField) -> Polynomial<SecureField>;

    /// Returns a transformed oracle where the first variable of `g` is fixed to `challenge`.
    ///
    /// The returned oracle represents the multivariate polynomial `g_t`, defined as
    /// `g_t(x_2, ..., x_n) = g(challenge, x_2, ..., x_n)`.
    fn fix_first(self, challenge: SecureField) -> Self;
}

/// Generates a proof of the sum-check protocol.
///
/// The sum-check protocol enables proving claims about the sum of a multivariate polynomial
/// `g(x_1, ..., x_n)` over the boolean hypercube `{0, 1}^n`. Operations on `g` are abstracted by
/// [`SumcheckOracle`]. The proof, list of challenges (variable assignment) and the finalized oracle
/// (the oracle with all challenges applied to the variables) are returned.
///
/// Output is of the form: `(proof, variable_assignment, finalized_oracle)`
pub fn prove<O: SumcheckOracle>(
    claim: SecureField,
    oracle: O,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>, O) {
    let mut round_polynomials = Vec::new();
    let mut challenges = Vec::new();

    let mut round_oracle = oracle;
    let mut round_claim = claim;

    for _round in 0..oracle.num_variables() {
        let round_polynomial = oracle.univariate_sum(claim);
        channel.mix_felts(&round_polynomial);

        let challenge = channel.draw_felt();
        round_oracle = round_oracle.fix_first(challenge);
        round_claim = round_polynomial.eval(challenge);
        round_polynomials.push(round_polynomial);
        challenges.push(challenge);
    }

    (SumcheckProof { round_polynomials }, challenges, oracle)
}

/// Partially verifies the sum-check protocol by validating the provided proof against the claim.
///
/// Only "partial" since it does not fully verify the prover's claimed evaluation on the variable
/// assignment but checks if the sum of the round polynomials evaluated on `0` and `1` matches the
/// claim for each round. If the proof passes these checks, the variable assignment and the prover's
/// claimed evaluation are returned for the caller to validate otherwise [`None`] is returned.
///
/// Output is of the form `(variable_assignment, claimed_eval)`.
// TODO: Is checking that each univariate round polynomial is <5 safe. I think it only impacts a
// few bits of security but not any more? Keeping it fixed keeps the implementation a little
// simpler.
pub fn partially_verify(
    claim: SecureField,
    proof: &SumcheckProof,
    channel: &mut impl Channel,
) -> Result<(Vec<SecureField>, SecureField), SumcheckError> {
    let mut assignment = Vec::new();
    let mut round_claim = claim;

    for (round, round_polynomial) in (1..).zip(&proof.round_polynomials) {
        if round_polynomial.degree() > MAX_DEGREE {
            return Err(SumcheckError::DegreeInvalid { round });
        }

        let eval0 = round_polynomial.eval(SecureField::zero());
        let eval1 = round_polynomial.eval(SecureField::one());
        let sum = eval0 + eval1;

        if round_claim != sum {
            return Err(SumcheckError::SumInvalid {
                claim: round_claim,
                sum,
                round,
            });
        }

        channel.mix_felts(round_polynomial);
        let challenge = channel.draw_felt();
        assignment.push(challenge);
        round_claim = round_polynomial.eval(challenge);
    }

    Ok((assignment, round_claim))
}

/// Max degree of univariate polynomials [`partially_verify()`] accepts in each round of the
/// sum-check protocol.
const MAX_DEGREE: usize = 3;

/// Error encountered during sum-check protocol verification.
///
/// Round 1 corresponds to the first round.
#[derive(Error, Debug)]
pub enum SumcheckError {
    #[error("degree of the univariate polynomial in round {round} is too high")]
    DegreeInvalid { round: usize },
    #[error("sum does not match the claim in round {round} (sum {sum}, claim {claim})")]
    SumInvalid {
        claim: SecureField,
        sum: SecureField,
        round: usize,
    },
}

#[derive(Debug, Clone)]
pub struct SumcheckProof {
    pub round_polynomials: Vec<Polynomial<SecureField>>,
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
    use prover_research::commitment_scheme::hasher::Hasher;
    use prover_research::core::channel::{Blake2sChannel, Channel};
    use prover_research::core::fields::m31::BaseField;
    use prover_research::core::fields::qm31::SecureField;

    use super::prove;
    use crate::mle::Mle;
    use crate::sumcheck::partially_verify;

    #[test]
    fn sumcheck_works() {
        let vals = [1, 2, 3, 4, 5, 6, 7, 8].map(|v| BaseField::from(v).into());
        let claim = vals.iter().copied().sum::<SecureField>();
        let mle = Mle::new(vals.to_vec());
        let (proof, ..) = prove(claim, mle.clone(), &mut test_channel());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, mle.eval(&assignment));
    }

    #[test]
    fn invalid_sumcheck_proof_fails() {
        let vals = [1, 2, 3, 4, 5, 6, 7, 8].map(|v| BaseField::from(v).into());
        let claim = vals.iter().copied().sum::<SecureField>();
        // Compromise the first value.
        let mut invalid_mle = Mle::new(vals.to_vec());
        invalid_mle[0] += SecureField::one();
        let (invalid_proof, ..) = prove(claim, invalid_mle, &mut test_channel());

        assert!(partially_verify(claim, &invalid_proof, &mut test_channel()).is_err());
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
