use num_traits::{One, Zero};
use prover_research::core::channel::Channel;
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::qm31::SecureField;

use crate::multivariate::{hypercube_sum, MultivariatePolynomial};
use crate::utils::Polynomial;

pub struct SumcheckProof {
    round_polynomials: Vec<Polynomial<SecureField>>,
}

/// Partially verifies the sumcheck protocol.
///
/// Returns the variable assignment and provers claimed evaluation of the given assignment. Only
/// performs partial verification since this function doesn't make check the evaluating the
/// polynomial with the returned variable assignment equals the provers claim.
///
/// Output is of the form `(variable_assignment, claimed_eval)`.
pub fn partially_verify(
    claim: SecureField,
    proof: &SumcheckProof,
    channel: &mut impl Channel,
) -> Option<(Vec<SecureField>, SecureField)> {
    let mut challenges = Vec::new();

    let eval = proof
        .round_polynomials
        .iter()
        .try_fold(claim, |round_claim, round_poly| {
            let r0 = round_poly.eval(SecureField::one());
            let r1 = round_poly.eval(-SecureField::one());

            if round_claim != r0 + r1 {
                return None;
            }

            channel.mix_felts(round_poly);
            let challenge = channel.draw_random_secure_felts()[0];
            challenges.push(challenge);
            Some(round_poly.eval(challenge))
        })?;

    Some((challenges, eval))
}

/// Returns proof and the evaluation point.
pub fn prove(
    g: &impl MultivariatePolynomial,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>) {
    let num_variables = g.num_variables();

    let mut round_polynomials = Vec::new();
    let mut challenges = Vec::new();

    for round in 0..num_variables {
        // TODO: Messy.
        let eval_at = |x| {
            let mut assignment = vec![SecureField::zero(); num_variables];
            assignment[0..round].clone_from_slice(&challenges);

            hypercube_sum(num_variables - round - 1, |hypercube_assignment| {
                assignment[round] = x;
                assignment[round + 1..].clone_from_slice(hypercube_assignment);
                g.eval(&assignment)
            })
        };

        let x0 = BaseField::zero().into();
        let x1 = BaseField::one().into();
        let x2 = (-BaseField::one()).into();
        let x3 = BaseField::from(2).into();
        let x4 = BaseField::from(4).into();

        let y0 = eval_at(x0);
        let y1 = eval_at(x1);
        let y2 = eval_at(x2);
        let y3 = eval_at(x3);
        let y4 = eval_at(x4);

        let round_polynomial =
            Polynomial::interpolate_lagrange(&[x0, x1, x2, x3, x4], &[y0, y1, y2, y3, y4]);
        println!("RP degree: {}", round_polynomial.degree());
        channel.mix_felts(&round_polynomial);
        round_polynomials.push(round_polynomial);

        challenges.push(channel.draw_random_secure_felts()[0]);
    }

    (SumcheckProof { round_polynomials }, challenges)
}

#[cfg(test)]
mod tests {

    use num_traits::One;
    use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
    use prover_research::commitment_scheme::hasher::Hasher;
    use prover_research::core::channel::{Blake2sChannel, Channel};
    use prover_research::core::fields::qm31::SecureField;
    use prover_research::core::fields::Field;
    use prover_research::fibonacci::verify_proof;

    use super::prove;
    use crate::multivariate::{self, hypercube_sum, MultivariatePolynomial};
    use crate::sumcheck::partially_verify;

    #[test]
    fn sumcheck_works() {
        // `f(x1, x2, x3) = 2x1^2 + x1 * x2 + x2 * x3`.
        let f = multivariate::from_const_fn(|[x1, x2, x3]| (x1 * x1).double() + x2 * x1 + x2 * x3);
        let claim = hypercube_sum(3, |assignment| f.eval(assignment));
        let (proof, _) = prove(&f, &mut test_channel());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, f.eval(&assignment));
    }

    #[test]
    fn invalid_sumcheck_proof_fails() {
        // `f(x1, x2, x3) = 2x1^2 + x1 * x2 + x2 * x3`.
        let f = multivariate::from_const_fn(|[x1, x2, x3]| (x1 * x1).double() + x2 * x1 + x2 * x3);
        let claim = hypercube_sum(3, |assignment| f.eval(assignment)) + SecureField::one();
        let (proof, _) = prove(&f, &mut test_channel());

        assert!(partially_verify(claim, &proof, &mut test_channel()).is_none());
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
