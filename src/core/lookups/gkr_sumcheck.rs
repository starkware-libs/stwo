use std::time::Instant;

use num_traits::{One, Zero};
use thiserror::Error;

use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::{eq, Polynomial};

/// The sum-check protocol enables proving claims about the sum of a multivariate polynomial
/// `g(x_1, ..., x_n)` over the boolean hypercube `{0, 1}^n`. This trait provides methods for
/// evaluating sums and making transformations on `g` in the context of the protocol. It is
/// indented to be used in conjunction with [`prove()`] to generate proofs.
///
/// This oracle is designed specifically for use with the multivariate polynomials in GKR protocol
/// which have the form `g(x_1, .., x_n) = eq(x_1, .., x_n, z_1, .., z_n) * f(x_1, .., x_n)`. These
/// multivariate polynomials must not exceed degree 3 in any `x_i` hence 'Cubic'.
pub trait CubicGkrSumcheckOracle: Sized {
    /// Returns the number of variables in `g`.
    fn num_variables(&self) -> usize;

    /// Evaluates the sum of `eq(x_1, ..., x_n, z_1, ..., z_n) * f(x_1, x_2, ..., x_n)` over all
    /// possible values `(x_2, ..., x_n)` in `{0, 1}^{n-1}` at `x_1 = 0` and `x_1 = 2`.
    fn univariate_sum_evals(&self) -> UnivariateEvals;

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
/// [`CubicGkrSumcheckOracle`]. The proof, list of challenges (variable assignment) and the
/// finalized oracle (the oracle with all challenges applied to the variables) are returned.
///
/// Output is of the form: `(proof, variable_assignment, finalized_oracle)`
pub fn prove<O: CubicGkrSumcheckOracle>(
    claim: SecureField,
    oracle: O,
    channel: &mut impl Channel,
    z: &[SecureField],
) -> (SumcheckProof, Vec<SecureField>, O) {
    let mut round_evals = Vec::new();
    let mut challenges = Vec::new();

    let mut round_oracle = oracle;
    let mut round_claim = claim;

    for round in 0..round_oracle.num_variables() {
        let evals = round_oracle.univariate_sum_evals();
        channel.mix_felts(&[evals.e0, evals.e2]);

        let challenge = channel.draw_felt();
        round_oracle = round_oracle.fix_first(challenge);
        round_claim = eval_round_polynomial(challenge, round_claim, z[round], evals);
        round_evals.push(evals);
        challenges.push(challenge);
    }

    let proof = SumcheckProof { round_evals };

    (proof, challenges, round_oracle)
}

/// Evaluates the sum-check round polynomial at `x`.
///
/// Let `f` be the univariate sum `g(x_1, .., x_n) = eq(x_1, .., x_n, z_1, .., z_n) * h(x_1, ..,
/// x_n)` over all possible values `(x_2, .., x_n)` in `{0, 1}^{n-1}` (the round polynomial). This
/// function returns `f` evaluated at `x`. The verifier is able to reconstruct `f` from the values
/// passed to this function:
///
/// - `claim`: the claimed sum of `g(x_1, .., x_n)` over all `(x_1, .., x_n)` in `{0, 1}^{n-1}`.
/// - `univariate_sum_evals`: Let `t` be the sum `eq(x_2, .., x_n, z_2, .., z_n) * h(x_1, .., x_n)`
///   over all possible values `(x_2, .., x_n)` in `{0, 1}^{n-1}`. `univariate_sum_evals` represents
///   the evaluations `t(0)` and `t(2)`. Note that `f(x) = eq(x, z_1) * t(x)`.
/// - `z`: randomness `z_1` needed to evaluate `eq(x_1, z_1)`.
///
/// Note that `claim = f(0) + f(1)`, which can be rearranged to obtain `t(1)` (since the verifier
/// has `t(0)` and can calculate `eq(x, z_1)` effieicntly). We now have `t(0)`, `t(1)` and `t(2)`
/// and can interpolate the quadratic polynomial `t`. Knowing `t` we can compute `f(x) = eq(x,
/// z_1) * t(x)`. This idea and algorithm is from <https://eprint.iacr.org/2024/108.pdf> (Section 3.2).
///
/// By reconstructing `f` the prover and verifier are able to cut their amount of communication
/// roughly in half vs a naive implementation. These optimizations come at the expense of slightly
/// more calculations that need to be made by the verifier.
fn eval_round_polynomial(
    x: SecureField,
    claim: SecureField,
    z: SecureField,
    univariate_sum_evals: UnivariateEvals,
) -> SecureField {
    let x0 = SecureField::zero();
    let x1 = SecureField::one();
    let x2 = BaseField::from(2).into();

    let y0 = univariate_sum_evals.e0;
    let y1 = (claim - eq(&[x0], &[z]) * univariate_sum_evals.e0) / eq(&[x1], &[z]);
    let y2 = univariate_sum_evals.e2;

    let t = Polynomial::interpolate_lagrange(&[x0, x1, x2], &[y0, y1, y2]);

    // Add in the missing factor `eq(x_1, z_1)`.
    eq(&[x], &[z]) * t.eval(x)
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

    for (round, round_polynomial) in proof.round_polynomials.iter().enumerate() {
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
/// Round 0 corresponds to the first round.
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
    pub round_evals: Vec<UnivariateEvals>,
}

// TODO: Re-integrate and document.
// /// Evaluations of the univariate round polynomial at `0` and `2`.
// ///
// /// Created as a generic function to handle all [`LogupTrace`] variants with a single
// /// implementation. [`None`] is passed to `numerators` for [`LogupTrace::Singles`] - the
// /// idea is that the compiler will inline the function and flatten the `numerator` match
// /// blocks that occur in the inner loop.

/// Stores the evaluations of a univariate polynomial `p(x)` at `x=0` and `x=2`.
#[derive(Debug, Clone, Copy)]
pub struct UnivariateEvals {
    /// Evaluation `p(0)`
    pub e0: SecureField,
    /// Evaluation `p(2)`.
    pub e2: SecureField,
}

#[cfg(test)]
mod tests {
    use std::ops::{AddAssign, MulAssign};
    use std::time::Instant;

    // use criterion::black_box;
    use num_traits::One;

    use super::{partially_verify, prove};
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec, K_BLOCK_SIZE};
    // use crate::core::backend::avx512::AVX512Backend;
    use crate::core::backend::cpu::CpuMle;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::mle::Mle;

    #[test]
    fn cpu_sumcheck_works() {
        const SIZE: usize = 1 << 26;
        let values = test_channel().draw_felts(SIZE);
        let claim = values.iter().copied().sum::<SecureField>();
        let mle = CpuMle::<SecureField>::new(values.into_iter().collect());
        let cloned_mle = mle.clone();
        let now = Instant::now();
        let (proof, ..) = prove(claim, cloned_mle, &mut test_channel());
        println!("CPU gen: {:?}", now.elapsed());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, mle.eval_at_point(&assignment));
    }

    #[test]
    fn avx_sumcheck_works() {
        const SIZE: usize = 1 << 26;
        let values = test_channel().draw_felts(SIZE);
        let claim = values.iter().copied().sum::<SecureField>();
        let mle = Mle::<AVX512Backend, SecureField>::new(values.into_iter().collect());
        let cloned_mle = mle.clone();
        let now = Instant::now();
        let (proof, ..) = prove(claim, cloned_mle, &mut test_channel());
        println!("AVX gen: {:?}", now.elapsed());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, mle.eval_at_point(&assignment));
    }

    #[test]
    fn invalid_sumcheck_proof_fails() {
        let values = [1, 2, 3, 4, 5, 6, 7, 8].map(|v| BaseField::from(v).into());
        let claim = values.iter().sum::<SecureField>();
        // Compromise the first value.
        let mut invalid_values = values.to_vec();
        invalid_values[0] += SecureField::one();
        let invalid_claim = invalid_values.iter().sum::<SecureField>();
        let invalid_mle = CpuMle::new(invalid_values.into_iter().collect());
        let (invalid_proof, ..) = prove(invalid_claim, invalid_mle, &mut test_channel());

        assert!(partially_verify(claim, &invalid_proof, &mut test_channel()).is_err());
    }

    #[test]
    fn avx_sum_vs_cpu_sum() {
        fn _product<T: MulAssign + Copy>(values: &[T]) -> T {
            let mut acc = values[0];
            values[1..].iter().for_each(|&v| acc *= v);
            acc
        }

        fn _sum<T: AddAssign + Copy>(values: &[T]) -> T {
            let mut acc = values[0];
            values[1..].iter().for_each(|&v| acc += v);
            acc
        }

        let data = std::iter::repeat(BaseField::one()).take(K_BLOCK_SIZE << 20);
        let column = data.collect::<BaseFieldVec>();

        let now = Instant::now();
        let p = _sum(&column.data);
        println!("AVX sum took {:?} {p:?}", now.elapsed());

        let now = Instant::now();
        let p = _sum(column.as_slice());
        println!("CPU sum took {:?} {p:?}", now.elapsed());
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
