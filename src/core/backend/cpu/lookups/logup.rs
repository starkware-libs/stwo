use num_traits::{One, Zero};

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::logup::{Fraction, LogupOps, LogupOracle, LogupTrace};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::{eq, Polynomial};

impl LogupOps for CPUBackend {
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self> {
        let mut next_numerators = SecureColumn::default();
        let mut next_denominators = SecureColumn::default();

        let half_layer_len = layer.len() / 2;
        let one = BaseField::one();

        match layer {
            LogupTrace::Singles { denominators } => {
                for i in 0..half_layer_len {
                    let d0 = denominators.at(i * 2);
                    let d1 = denominators.at(i * 2 + 1);

                    let res = Fraction::new(one, d0) + Fraction::new(one, d1);

                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => {
                for i in 0..half_layer_len {
                    let n0 = numerators[i * 2];
                    let d0 = denominators.at(i * 2);

                    let n1 = numerators[i * 2 + 1];
                    let d1 = denominators.at(i * 2 + 1);

                    let res = Fraction::new(n0, d0) + Fraction::new(n1, d1);

                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
            LogupTrace::Generic {
                numerators,
                denominators,
            } => {
                for i in 0..half_layer_len {
                    let n0 = numerators.at(i * 2);
                    let d0 = denominators.at(i * 2);

                    let n1 = numerators.at(i * 2 + 1);
                    let d1 = denominators.at(i * 2 + 1);

                    let res = Fraction::new(n0, d0) + Fraction::new(n1, d1);

                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
        }

        LogupTrace::Generic {
            numerators: Mle::new(next_numerators),
            denominators: Mle::new(next_denominators),
        }
    }

    fn univariate_sum(
        oracle: &LogupOracle<'_, Self>,
        claim: SecureField,
    ) -> Polynomial<SecureField> {
        let num_terms = 1 << (oracle.num_variables() - 1);
        let lambda = oracle.lambda();
        let eq_evals = oracle.eq_evals();
        let trace = oracle.trace();
        let z = oracle.z();
        let r = oracle.r();

        // Obtain the evaluations at `0` and `2`.
        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();

        for i in 0..num_terms {
            let (fraction0, fraction2) = match trace {
                LogupTrace::Singles { denominators } => {
                    let d0_lhs = denominators.at(i * 2);
                    let d1_lhs = denominators.at(i * 2 + 1);

                    let fraction0 = Fraction::new(d0_lhs + d1_lhs, d0_lhs * d1_lhs);

                    let d0_rhs = denominators.at((num_terms + i) * 2);
                    let d1_rhs = denominators.at((num_terms + i) * 2 + 1);

                    let fraction2 = {
                        let d0 = d0_rhs.double() - d0_lhs;
                        let d1 = d1_rhs.double() - d1_lhs;
                        Fraction::new(d0 + d1, d0 * d1)
                    };

                    (fraction0, fraction2)
                }
                LogupTrace::Multiplicities {
                    numerators,
                    denominators,
                } => {
                    let n0_lhs = numerators[i * 2];
                    let n1_lhs = numerators[i * 2 + 1];

                    let d0_lhs = denominators.at(i * 2);
                    let d1_lhs = denominators.at(i * 2 + 1);

                    let fraction0 = {
                        let a = Fraction::new(n0_lhs, d0_lhs);
                        let b = Fraction::new(n1_lhs, d1_lhs);
                        a + b
                    };

                    let n0_rhs = numerators[(num_terms + i) * 2];
                    let n1_rhs = numerators[(num_terms + i) * 2 + 1];

                    let d0_rhs = denominators.at((num_terms + i) * 2);
                    let d1_rhs = denominators.at((num_terms + i) * 2 + 1);

                    let fraction2 = {
                        let n0 = n0_rhs.double() - n0_lhs;
                        let n1 = n1_rhs.double() - n1_lhs;

                        let d0 = d0_rhs.double() - d0_lhs;
                        let d1 = d1_rhs.double() - d1_lhs;

                        let a = Fraction::new(n0, d0);
                        let b = Fraction::new(n1, d1);
                        a + b
                    };

                    (fraction0, fraction2)
                }
                LogupTrace::Generic {
                    numerators,
                    denominators,
                } => {
                    let n0_lhs = numerators.at(i * 2);
                    let n1_lhs = numerators.at(i * 2 + 1);

                    let d0_lhs = denominators.at(i * 2);
                    let d1_lhs = denominators.at(i * 2 + 1);

                    let fraction0 = {
                        let a = Fraction::new(n0_lhs, d0_lhs);
                        let b = Fraction::new(n1_lhs, d1_lhs);
                        a + b
                    };

                    let n0_rhs = numerators.at((num_terms + i) * 2);
                    let n1_rhs = numerators.at((num_terms + i) * 2 + 1);

                    let d0_rhs = denominators.at((num_terms + i) * 2);
                    let d1_rhs = denominators.at((num_terms + i) * 2 + 1);

                    let fraction2 = {
                        let n0 = n0_rhs.double() - n0_lhs;
                        let n1 = n1_rhs.double() - n1_lhs;

                        let d0 = d0_rhs.double() - d0_lhs;
                        let d1 = d1_rhs.double() - d1_lhs;

                        let a = Fraction::new(n0, d0);
                        let b = Fraction::new(n1, d1);
                        a + b
                    };

                    (fraction0, fraction2)
                }
            };

            let eq_eval = eq_evals[i];
            eval_at_0 += eq_eval * (fraction0.numerator + lambda * fraction0.denominator);
            eval_at_2 += eq_eval * (fraction2.numerator + lambda * fraction2.denominator);
        }

        // We wanted to compute a sum of a multivariate polynomial
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n)) * (..)` over
        // all `(x_k, ..., x_n)` in `{0, 1}^(n-k)`. Instead we computes a sum over
        // `eq((0^(k-2), x_k, .., x_n), (z_2, .., z_n)) * (..)`. The two multivariate sums differs
        // by a constant factor `eq((0), (z_1))` which is added back in here.
        //
        // The reason the factor is left out originally is for performance reasons. In the naive
        // version we want to precompute the evaluations of `eq((x_1, .., x_n), (z_1, .., z_n))`
        // ahead of time for all `(x_1, .., x_n)` in `{0, 1}^n`. Notice we only use half of
        // these evaluations (specifically those where `x_1` is zero). Each the term of the sum gets
        // multiplied by one of these evaluations. Notice all the terms of the sum contain a
        // constant factor `eq((x_1), (z_1))` (since x_1 equals zero). In the optimized
        // version we precompute the evaluations of `eq((x_2, .., x_n), (z_2, .., z_n))` which is
        // half the size (and takes half the work) of the original precomputation. We then add the
        // missing `eq((x_1), (z_1))` factor back here.
        //
        // TODO: Doc is a bit wordy it's not great have to explain all this but the optimization
        // is worthwhile. Consider modifying `gen_eq_evals()` so that it only returns the first
        // half. Would be just as optimized but prevent having to explain things here.
        eval_at_0 *= eq(&[SecureField::zero()], &[z[0]]);
        eval_at_2 *= eq(&[SecureField::zero()], &[z[0]]);

        // The evaluations on `0` and `2` are invalid. They were obtained by summing over the poly
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n)) * (..)` but we require the sum to be taken
        // on `eq((r_1, ..., r_{k-1}, x_k, .., x_n), (z_1, .., z_n)) * (..)`. Conveniently
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n))` and `eq((r_1, ..., r_{k-1}, x_k, .., x_n),
        // (z_1, .., z_n))` differ only by a constant factor `eq((r_1, ..., r_{k-1}), (z_1, ..,
        // z_{k-1})) / eq((0^(k-1)), (z_1, .., z_{k-1}))` for all values of `x`.
        // TODO: explain
        let k = r.len();
        let eq_correction_factor = eq(r, &z[0..k]) / eq(&vec![SecureField::zero(); k], &z[0..k]);

        // Our goal is to compute the sum of `eq((x_k, .., x_n), (z_k, .., z_n)) * h(x_k, .., x_n)`
        // over all possible values `(x_{k+1}, .., x_n)` in `{0, 1}^{n-1}`, effectively reducing the
        // sum to a univariate polynomial in `x_k`. Let this univariate polynomial be `f`. Our
        // method to is to evaluate `f` in `deg(f) + 1` points (which can be done efficiently) to
        // obtain the coefficient representation of `f` via interpolation.
        //
        // Although evaluating `f` is efficient, the runtime of the sumcheck prover is proportional
        // to how many points `f` needs to be evaluated on. To reduce the number of evaluations the
        // prover must perform we can reduce the degree of of the polynomial we need to interpolate.
        // This can be done by instead computing the sum over `eq((0, .., x_n), (z_k, .., z_n)) *
        // h(x_k, .., x_n)` denoted `simplified_sum` which has degree `deg(f) - 1`. We interpolate,
        // our lower degree polynomial, `simplified_sum` with one less evaluation and multiply it
        // afterwards by `eq((x_k), (z_k)) / eq((0), (z_k))` to obtain the original `f`. This idea
        // and algorithm is from <https://eprint.iacr.org/2024/108.pdf> (Section 3.2).
        let correction_factor_at = |x| eq(&[x], &[z[k]]) / eq(&[SecureField::zero()], &[z[k]]);

        let x0: SecureField = BaseField::zero().into();
        let x1 = BaseField::one().into();
        let x2 = BaseField::from(2).into();

        let mut y0 = eq_correction_factor * eval_at_0;
        let mut y1 = (claim - y0) / correction_factor_at(x1);
        let mut y2 = eq_correction_factor * eval_at_2;

        // We are interpolating a degree 2 function so need three evaluations.
        let simplified_univariate_sum =
            Polynomial::interpolate_lagrange(&[x0, x1, x2], &[y0, y1, y2]);

        let x3 = BaseField::from(3).into();
        let mut y3 = simplified_univariate_sum.eval(x3);

        // Correct all the evaluations (see comment above).
        y0 *= correction_factor_at(x0); // `y0 *= 1`
        y1 *= correction_factor_at(x1);
        y2 *= correction_factor_at(x2);
        y3 *= correction_factor_at(x3);

        Polynomial::interpolate_lagrange(&[x0, x1, x2, x3], &[y0, y1, y2, y3])
    }
}
