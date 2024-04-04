use std::iter::zip;

use num_traits::{One, Zero};

use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::UnivariatePolynomial;

impl MleOps<BaseField> for CPUBackend {
    fn eval_at_point(mle: &Mle<Self, BaseField>, point: &[BaseField]) -> BaseField {
        eval_mle_at_point(mle, point)
    }

    fn fix_first(mle: Mle<Self, BaseField>, assignment: SecureField) -> Mle<Self, SecureField> {
        let midpoint = mle.len() / 2;
        let (lhs_evals, rhs_evals) = mle.split_at(midpoint);

        let res = zip(lhs_evals, rhs_evals)
            // Equivalent to `eq(0, assignment) * lhs_eval + eq(1, assignment) * rhs_eval`.
            .map(|(&lhs_eval, &rhs_eval)| assignment * (rhs_eval - lhs_eval) + lhs_eval)
            .collect();

        Mle::new(res)
    }
}

impl MleOps<SecureField> for CPUBackend {
    fn eval_at_point(mle: &Mle<Self, SecureField>, point: &[SecureField]) -> SecureField {
        eval_mle_at_point(mle, point)
    }

    fn fix_first(mle: Mle<Self, SecureField>, assignment: SecureField) -> Mle<Self, SecureField> {
        let midpoint = mle.len() / 2;
        let mut evals = mle.into_evals();

        for i in 0..midpoint {
            let lhs_eval = evals[i];
            let rhs_eval = evals[i + midpoint];
            // Equivalent to `eq(0, assignment) * lhs_eval + eq(1, assignment) * rhs_eval`.
            evals[i] = lhs_eval + assignment * (rhs_eval - lhs_eval);
        }

        evals.truncate(midpoint);

        Mle::new(evals)
    }
}

impl SumcheckOracle for Mle<CPUBackend, SecureField> {
    fn num_variables(&self) -> usize {
        self.num_variables()
    }

    fn univariate_sum(&self, claim: SecureField) -> UnivariatePolynomial<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let y0 = self[0..self.len() / 2].iter().sum();
        let y1 = claim - y0;

        UnivariatePolynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        self.fix_first(challenge)
    }
}

fn eval_mle_at_point<F: Field>(mle_evals: &[F], p: &[F]) -> F {
    match p {
        [] => mle_evals[0],
        [p_i, p @ ..] => {
            let (lhs, rhs) = mle_evals.split_at(mle_evals.len() / 2);
            let lhs_eval = eval_mle_at_point(lhs, p);
            let rhs_eval = eval_mle_at_point(rhs, p);
            // Equivalent to `eq(0, p_i) * lhs_eval + eq(1, p_i) * rhs_eval`.
            *p_i * (rhs_eval - lhs_eval) + lhs_eval
        }
    }
}
