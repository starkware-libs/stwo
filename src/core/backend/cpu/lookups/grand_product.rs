use num_traits::{One, Zero};

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::grand_product::{GrandProductOps, GrandProductOracle, GrandProductTrace};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::{eq, Polynomial};

impl GrandProductOps for CPUBackend {
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self> {
        let half_n = layer.len() / 2;
        let mut next_layer = SecureColumn::default();

        for i in 0..half_n {
            let a = layer.at(i * 2);
            let b = layer.at(i * 2 + 1);
            next_layer.push(a * b);

            // let c = a * b;

            // let [c0, c1, c2, c3] = c.to_m31_array();
            // next_layer.cols[0].push(c0);
            // next_layer.cols[1].push(c1);
            // next_layer.cols[2].push(c2);
            // next_layer.cols[3].push(c3);
        }

        GrandProductTrace::new(Mle::new(next_layer))
    }

    fn univariate_sum(
        oracle: &GrandProductOracle<'_, Self>,
        claim: SecureField,
    ) -> Polynomial<SecureField> {
        let num_terms = 1 << (oracle.num_variables() - 1);
        let eq_evals = oracle.eq_evals();
        let trace = oracle.trace();
        let z = oracle.z();
        let r = oracle.r();

        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();

        #[allow(clippy::needless_range_loop)]
        for i in 0..num_terms {
            let lhs0 = trace.at(i * 2);
            let lhs1 = trace.at(i * 2 + 1);

            let product0 = lhs0 * lhs1;

            let rhs0 = trace.at((num_terms + i) * 2);
            let rhs1 = trace.at((num_terms + i) * 2 + 1);

            let product2 = (rhs0.double() - lhs0) * (rhs1.double() - lhs1);

            let eq_eval = eq_evals[i];
            eval_at_0 += eq_eval * product0;
            eval_at_2 += eq_eval * product2;
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
