use num_traits::{One, Zero};

use crate::core::backend::cpu::CpuMle;
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, Field};
use crate::core::lookups::logup::{Fraction, LogupOps, LogupOracle, LogupTrace};
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::{eq, Polynomial};

impl LogupOps for CPUBackend {
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self> {
        /// Computes the values in the next layer of the circuit.
        ///
        /// Created as a generic function to handle all [`LogupTrace`] variants with a single
        /// implementation. [`None`] is passed to `numerators` for [`LogupTrace::Singles`] - the
        /// idea is that the compiler will inline the function and flatten the `numerator` match
        /// blocks that occur in the inner loop.
        fn _next_layer<F>(
            numerators: Option<&CpuMle<F>>,
            denominators: &CpuMle<SecureField>,
        ) -> LogupTrace<CPUBackend>
        where
            F: Field,
            SecureField: ExtensionOf<F> + Field,
            CPUBackend: MleOps<F, Column = Vec<F>>,
        {
            let mut next_numerators = Vec::new();
            let mut next_denominators = Vec::new();

            let mut numerator_pairs = numerators.map(|n| n.array_chunks()).into_iter().flatten();
            let denominator_pairs = denominators.array_chunks();

            let one = BaseField::one();

            for &[d0, d1] in denominator_pairs {
                let res = match numerator_pairs.next() {
                    Some(&[n0, n1]) => Fraction::new(n0, d0) + Fraction::new(n1, d1),
                    None => Fraction::new(one, d0) + Fraction::new(one, d1),
                };

                next_numerators.push(res.numerator);
                next_denominators.push(res.denominator);
            }

            LogupTrace::Generic {
                numerators: Mle::new(next_numerators),
                denominators: Mle::new(next_denominators),
            }
        }

        match layer {
            LogupTrace::Singles { denominators } => _next_layer::<SecureField>(None, denominators),
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => _next_layer(Some(numerators), denominators),
            LogupTrace::Generic {
                numerators: n,
                denominators: d,
            } => _next_layer(Some(n), d),
        }
    }

    fn univariate_sum(
        oracle: &LogupOracle<'_, Self>,
        claim: SecureField,
    ) -> Polynomial<SecureField> {
        /// Evaluates the univariate sum at `0` and `2`. (TODO improve doc)
        ///
        /// Created as a generic function to handle all [`LogupTrace`] variants with a single
        /// implementation. [`None`] is passed to `numerators` for [`LogupTrace::Singles`] - the
        /// idea is that the compiler will inline the function and flatten the `numerator` match
        /// blocks that occur in the inner loop.
        fn univariate_sum_evals<F>(
            numerators: Option<&CpuMle<F>>,
            denominators: &CpuMle<SecureField>,
            eq_evals: &[SecureField],
            lambda: SecureField,
            num_terms: usize,
        ) -> (SecureField, SecureField)
        where
            F: Field,
            SecureField: ExtensionOf<F> + Field,
            CPUBackend: MleOps<F, Column = Vec<F>>,
        {
            let mut eval_at_0 = SecureField::zero();
            let mut eval_at_2 = SecureField::zero();

            let numerator_pairs = numerators.map(|n| n.as_chunks().0);
            let denominator_pairs = denominators.as_chunks().0;

            for i in 0..num_terms {
                let [denominator_lhs0, denominator_lhs1] = denominator_pairs[i];

                let fraction0 = match numerator_pairs.map(|v| v[i]) {
                    Some([numerator_lhs0, numerator_lhs1]) => {
                        let a = Fraction::new(numerator_lhs0, denominator_lhs0);
                        let b = Fraction::new(numerator_lhs1, denominator_lhs1);
                        a + b
                    }
                    None => Fraction::new(
                        denominator_lhs0 + denominator_lhs1,
                        denominator_lhs0 * denominator_lhs1,
                    ),
                };

                let [denominator_rhs0, denominator_rhs1] = denominator_pairs[num_terms + i];

                let fraction2 = {
                    let d0 = denominator_rhs0.double() - denominator_lhs0;
                    let d1 = denominator_rhs1.double() - denominator_lhs1;

                    match numerator_pairs.map(|v| (v[i], v[num_terms + i])) {
                        Some((
                            [numerator_lhs0, numerator_lhs1],
                            [numerator_rhs0, numerator_rhs1],
                        )) => {
                            let n0 = numerator_rhs0.double() - numerator_lhs0;
                            let n1 = numerator_rhs1.double() - numerator_lhs1;

                            let a = Fraction::new(n0, d0);
                            let b = Fraction::new(n1, d1);
                            a + b
                        }
                        None => Fraction::new(d0 + d1, d0 * d1),
                    }
                };

                let eq_eval = eq_evals[i];
                eval_at_0 += eq_eval * (fraction0.numerator + lambda * fraction0.denominator);
                eval_at_2 += eq_eval * (fraction2.numerator + lambda * fraction2.denominator);
            }

            (eval_at_0, eval_at_2)
        }

        let num_terms = (1 << oracle.num_variables()) / 2;
        let lambda = oracle.lambda();
        let eq_evals = oracle.eq_evals();
        let z = oracle.z();
        let r = oracle.r();

        // Obtain the evaluations at `0` and `2`.
        let (mut eval_at_0, mut eval_at_2) = match oracle.trace() {
            LogupTrace::Generic {
                numerators,
                denominators,
            } => univariate_sum_evals(Some(numerators), denominators, eq_evals, lambda, num_terms),
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => univariate_sum_evals(Some(numerators), denominators, eq_evals, lambda, num_terms),
            LogupTrace::Singles { denominators } => {
                univariate_sum_evals::<SecureField>(None, denominators, eq_evals, lambda, num_terms)
            }
        };

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
