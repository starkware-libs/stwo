use num_traits::{One, Zero};
use prover_research::core::channel::Channel;
use prover_research::core::fields::qm31::SecureField;
use thiserror::Error;

use crate::utils::Polynomial;

/// The sum-check protocol enables proving claims about the sum of a multivariate polynomial
/// `g(x_1, ..., x_n)` over the boolean hypercube `{0, 1}^n`. This trait should be implemented on an
/// oracle capable of evaluating sums and transformations of `g` in the context of the protocol.
/// This trait is used in conjunction with [`sumcheck::prove()`] to generate proofs.
///
/// [`sumcheck::prove()`]: prove
pub trait SumcheckOracle {
    /// Returns the number of variables in `g` (determines the number of rounds in the protocol).
    fn num_variables(&self) -> usize;

    /// Computes the sum of `g(x_1, x_2, ..., x_n)` over all possible values `(x_2, ..., x_n)` in
    /// `{0, 1}^{n-1}`, effectively reducing the sum over `g` to a univariate polynomial in `x_1`.
    fn univariate_sum(&self) -> Polynomial<SecureField>;

    /// Returns a new oracle where the first variable of `g` is fixed to `challenge`.
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
    mut oracle: O,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>, O) {
    let mut round_polynomials = Vec::new();
    let mut challenges = Vec::new();

    for _round in 0..oracle.num_variables() {
        let round_polynomial = oracle.univariate_sum();
        channel.mix_felts(&round_polynomial);
        let challenge = channel.draw_felt();
        round_polynomials.push(round_polynomial);
        challenges.push(challenge);
        oracle = oracle.fix_first(challenge);
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
pub fn partially_verify(
    claim: SecureField,
    proof: &SumcheckProof,
    channel: &mut impl Channel,
) -> Result<(Vec<SecureField>, SecureField), SumcheckError> {
    let mut assignment = Vec::new();
    let mut round_claim = claim;

    for (round, round_polynomial) in (1..).zip(&proof.round_polynomials) {
        let r0 = round_polynomial.eval(SecureField::zero());
        let r1 = round_polynomial.eval(SecureField::one());

        if round_claim != r0 + r1 {
            return Err(SumcheckError {
                claim: round_claim,
                sum: r0 + r1,
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

/// Error encountered during sum-check protocol verification.
///
/// Round 1 corresponds to the first round.
// TODO: Change to enum and add error for round polynomial degree.
#[derive(Error, Debug)]
#[error("sum does not match the claim in round {round} (sum {sum}, claim {claim})")]
pub struct SumcheckError {
    claim: SecureField,
    sum: SecureField,
    round: usize,
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
    use crate::mle::MultiLinearExtension;
    use crate::sumcheck::partially_verify;

    #[test]
    fn sumcheck_works() {
        let vals = [1, 2, 3, 4, 5, 6, 7, 8].map(|v| BaseField::from(v).into());
        let claim = vals.iter().copied().sum::<SecureField>();
        let mle = MultiLinearExtension::new(vals.to_vec());
        let (proof, ..) = prove(mle.clone(), &mut test_channel());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, mle.eval(&assignment));
    }

    #[test]
    fn invalid_sumcheck_proof_fails() {
        let vals = [1, 2, 3, 4, 5, 6, 7, 8].map(|v| BaseField::from(v).into());
        let claim = vals.iter().copied().sum::<SecureField>();
        // Compromise the first value.
        let mut invalid_mle = MultiLinearExtension::new(vals.to_vec());
        invalid_mle[0] += SecureField::one();
        let (invalid_proof, ..) = prove(invalid_mle, &mut test_channel());

        assert!(partially_verify(claim, &invalid_proof, &mut test_channel()).is_err());
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
