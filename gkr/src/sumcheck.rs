use std::iter::{zip, Sum};
use std::marker::PhantomData;
use std::mem::{self, swap};
use std::ops::{Deref, DerefMut};

use num_traits::{One, Zero};
use prover_research::core::channel::Channel;
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::qm31::SecureField;
use prover_research::core::fields::{ExtensionOf, Field};
use prover_research::core::poly::NaturalOrder;

use crate::multivariate::{hypercube_sum, MultivariatePolynomial};
use crate::utils::Polynomial;

/// Multi-Linear Extension.
// TODO: "Values are assumed to be in lagrange basis" - Is that correct wording?
#[derive(Debug, Clone)]
pub struct MultiLinearExtension<F> {
    num_variables: usize,
    evals: Vec<F>,
}

impl<F: Field> MultiLinearExtension<F> {
    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        let num_variables = evals.len().ilog2() as usize;
        Self {
            num_variables,
            evals,
        }
    }

    pub fn eval(&self, point: &[F]) -> F {
        fn eval_part<F: Field>(evals: &[F], point: &[F]) -> F {
            if evals.len() == 1 {
                // TODO: Remove.
                assert!(point.is_empty());
                return evals[0];
            }

            let (lhs, rhs) = evals.split_at(evals.len() / 2);
            let lhs_eval = eval_part(lhs, &point[1..]);
            let rhs_eval = eval_part(rhs, &point[1..]);
            let coordinate = point[0];
            lhs_eval * (F::one() - coordinate) + rhs_eval * coordinate
        }

        assert_eq!(point.len(), self.num_variables);
        eval_part(&self.evals, point)
    }
}

// impl<F: Field> HypercubeEvaluation<F> {
//     /// Stores the evaluations in [`EvenOddOrder`].
//     fn partition_even_odd(mut self) -> HypercubeEvaluation<F, EvenOddOrder> {
//         let parity_pairs = self.evals.as_chunks_mut().0;
//         let (lhs, rhs) = parity_pairs.split_at_mut(parity_pairs.len() / 2);
//         zip(lhs, rhs).for_each(|([_, lhs_odd], [rhs_even, _])| swap(lhs_odd, rhs_even));
//         HypercubeEvaluation {
//             evals: self.evals,
//             dimension: self.dimension,
//             _eval_order: PhantomData,
//         }
//     }
// }

impl<F> Deref for MultiLinearExtension<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

impl<F> DerefMut for MultiLinearExtension<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evals
    }
}

// /// All even point evaluations are in the LHS and all odd point evaluations are in the RHS.
// #[derive(Copy, Clone, Debug)]
// pub struct EvenOddOrder;

pub struct SumcheckProof {
    pub round_polynomials: Vec<Polynomial<SecureField>>,
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
    proof.round_polynomials.iter().try_fold(
        (Vec::new(), claim),
        |(mut challenges, round_claim), round_poly| {
            let r0 = round_poly.eval(SecureField::zero());
            let r1 = round_poly.eval(SecureField::one());

            if round_claim != r0 + r1 {
                return None;
            }

            channel.mix_felts(round_poly);
            let challenge = channel.draw_felt();
            challenges.push(challenge);
            Some((challenges, round_poly.eval(challenge)))
        },
    )
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

        println!("eval at 0: {y0}");
        println!("eval at 1: {y1}");

        let round_polynomial =
            Polynomial::interpolate_lagrange(&[x0, x1, x2, x3, x4], &[y0, y1, y2, y3, y4]);
        // println!("yo:d:{} ", round_polynomial.degree());
        channel.mix_felts(&round_polynomial);
        round_polynomials.push(round_polynomial);

        challenges.push(channel.draw_felt());
    }

    (SumcheckProof { round_polynomials }, challenges)
}

/// The sum-check protocol enables proving claims about the sum of a multivariate polynomial
/// `g(x_1, ..., x_n)` over the boolean hypercube `{0, 1}^n`. This trait enables evaluating sums of
/// `g` in the context of the protocol and is used in conjunction with [`sumcheck::prove()`] to
/// generate proofs.
///
/// [`sumcheck::prove()`]: prove
pub trait SumcheckOracle {
    /// Oracle for the next round of the protocol.
    // TODO: Consider removing. Currently here so the first round can operate on smaller fields.
    type NextRoundOracle: SumcheckOracle<NextRoundOracle = Self::NextRoundOracle>;

    /// Returns the number of variables in `g` (determines the number of rounds in the protocol).
    fn num_variables(&self) -> u32;

    /// Computes the sum of `g(x_1, x_2, ..., x_n)` over all possible values `(x_2, ..., x_n)` in
    /// `{0, 1}^{n-1}`, effectively reducing `g` to a univariate polynomial.
    fn univariate_sum(&self) -> Polynomial<SecureField>;

    /// Returns a new oracle where the first variable of `g` is fixed to `challenge`.
    fn fix_first(self, challenge: SecureField) -> Self::NextRoundOracle;
}

// TODO: docs
pub fn prove3<O: SumcheckOracle>(
    oracle: O,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>, O::NextRoundOracle) {
    let num_rounds = oracle.num_variables();
    let mut round_polynomials = Vec::new();
    let mut challenges = Vec::new();

    // Handle first round.
    let mut oracle = {
        let round_polynomial = oracle.univariate_sum();
        channel.mix_felts(&round_polynomial);
        let challenge = channel.draw_felt();
        round_polynomials.push(round_polynomial);
        challenges.push(challenge);
        oracle.fix_first(challenge)
    };

    // Handle remaining rounds.
    for round in 1..num_rounds {
        let round_polynomial = oracle.univariate_sum();
        channel.mix_felts(&round_polynomial);
        let challenge = channel.draw_felt();
        round_polynomials.push(round_polynomial);
        challenges.push(challenge);
        oracle = oracle.fix_first(challenge);
    }

    (SumcheckProof { round_polynomials }, challenges, oracle)
}

/// Returns proof and the evaluation point.
pub fn prove2<F>(
    g: &MultiLinearExtension<F>,
    channel: &mut impl Channel,
) -> (SumcheckProof, Vec<SecureField>)
where
    F: ExtensionOf<BaseField>,
    SecureField: ExtensionOf<F> + Field,
{
    let mut round_polynomials = Vec::new();
    let mut challenges = Vec::new();

    // Handle first round.
    let mut t = {
        let (t, round_polynomial, challenge) = collapse(g, channel);
        channel.mix_felts(&round_polynomial);
        round_polynomials.push(round_polynomial);
        challenges.push(challenge);
        t
    };

    for round in 1..g.num_variables {
        let (t_next, round_polynomial, challenge) = collapse::<SecureField>(&t, channel);
        channel.mix_felts(&round_polynomial);
        round_polynomials.push(round_polynomial);
        challenges.push(challenge);
        t = t_next;
    }

    (SumcheckProof { round_polynomials }, challenges)
}

// /// Source: <https://github.com/ingonyama-zk/papers/blob/main/sumcheck_201_chapter_1.pdf> (Algorithm 1)
// fn prove_round<F>(t: &MultiLinearExtension<F>, channel: &mut impl Channel)
// where
//     F: ExtensionOf<BaseField>,
//     SecureField: ExtensionOf<F> + Field,
// {
//     collapse(t)
// }

/// Returns output of the form: `(collapsed_evaluation, r, challenge)`
///
/// Source: <https://github.com/ingonyama-zk/papers/blob/main/sumcheck_201_chapter_1.pdf> (Algorithm 1)
fn collapse<F>(
    mle: &MultiLinearExtension<F>,
    channel: &mut impl Channel,
) -> (
    MultiLinearExtension<SecureField>,
    Polynomial<SecureField>,
    SecureField,
)
where
    F: ExtensionOf<BaseField>,
    SecureField: ExtensionOf<F> + Field,
{
    let [r0, r1] = mle.array_chunks().copied().fold([F::zero(); 2], sum_pairs);

    // TODO: Send evaluations to verifier not coefficients
    let r = Polynomial::interpolate_lagrange(
        &[SecureField::zero(), SecureField::one()],
        &[r0.into(), r1.into()],
    );
    channel.mix_felts(&r);
    let challenge = channel.draw_felt();

    let collapsed_mle = MultiLinearExtension::new(
        mle.array_chunks()
            // Computes `(1 - challenge) * e + challenge * o` with one less multiplication.
            .map(|&[e, o]| challenge * (o - e) + e)
            .collect(),
    );

    (collapsed_mle, r, challenge)
}

fn sum_pairs<F: Field>([a0, a1]: [F; 2], [b0, b1]: [F; 2]) -> [F; 2] {
    [a0 + b0, a1 + b1]
}

#[cfg(test)]
mod tests {

    use std::hint::black_box;
    use std::time::Instant;

    use num_traits::{One, Zero};
    use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
    use prover_research::commitment_scheme::hasher::Hasher;
    use prover_research::core::channel::{Blake2sChannel, Channel};
    use prover_research::core::fields::m31::BaseField;
    use prover_research::core::fields::qm31::SecureField;
    use prover_research::core::fields::Field;
    use prover_research::core::poly::NaturalOrder;
    use prover_research::fibonacci::verify_proof;

    use super::{collapse, prove};
    use crate::gkr::eq;
    use crate::multivariate::{self, hypercube_sum, MultivariatePolynomial};
    use crate::sumcheck::{partially_verify, prove2, MultiLinearExtension};

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

    #[test]
    fn collapse_test() {
        let one = SecureField::one();
        let zero = SecureField::zero();
        let random = test_channel().draw_felt();
        let mle = MultiLinearExtension::new(vec![random; 1 << 26]);

        let g = multivariate::from_const_fn(|[random]| {
            eq(&[zero, zero, random], &[zero, zero, zero]) * mle[0]
                + eq(&[zero, zero, random], &[zero, zero, one]) * mle[1]
                + eq(&[zero, one, random], &[zero, one, zero]) * mle[2]
                + eq(&[zero, one, random], &[zero, one, one]) * mle[3]
                + eq(&[one, zero, random], &[one, zero, zero]) * mle[4]
                + eq(&[one, zero, random], &[one, zero, one]) * mle[5]
                + eq(&[one, one, random], &[one, one, zero]) * mle[6]
                + eq(&[one, one, random], &[one, one, one]) * mle[7]
        });

        let random = BaseField::from(123).into();
        println!("yo: {}", g.eval(&[random]));

        // while res.len() > 8 {
        //     res = collapse(&res, &mut test_channel());
        // }

        let now = Instant::now();
        let claim = mle.iter().copied().sum();
        println!("claim time: {:?}", now.elapsed());

        let now = Instant::now();
        let (proof, eval_point) = prove2(&mle, &mut test_channel());
        println!("prove time: {:?}", now.elapsed());

        let now = Instant::now();
        let result = partially_verify(claim, &proof, &mut test_channel());
        println!("verify time: {:?}", now.elapsed());
        // println!("YO: {:?}", result);

        // println!("")
    }

    // #[test]
    // fn benchmark_sumcheck() {
    //     let now = Instant::now();
    //     let eval = black_box(HypercubeEvaluation::<BaseField, NaturalOrder>::new(
    //         (0..1 << 24).map(BaseField::from_u32_unchecked).collect(),
    //     ));
    //     println!("allocation in: {:?}", now.elapsed());

    //     println!("eval: {}", eval[10000]);

    //     let now = Instant::now();
    //     let next_evals = black_box(eval);
    //     println!("reorder in: {:?}", now.elapsed());

    //     println!("eval: {}", next_evals[10001]);

    //     let _ = eval;

    //     todo!()
    // }
}
