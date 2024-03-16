use std::iter::zip;

use num_traits::{One, Zero};

use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::Polynomial;

/// Evaluates the multi-linear extension `mle` at point `p`.
fn eval_mle_at_point<F: Field>(mle: &[F], p: &[F]) -> F {
    match p {
        [] => mle[0],
        [p_i, p @ ..] => {
            let (lhs, rhs) = mle.split_at(mle.len() / 2);
            let lhs_eval = eval_mle_at_point(lhs, p);
            let rhs_eval = eval_mle_at_point(rhs, p);
            // `= eq(0, p_i) * lhs + eq(1, p_i) * rhs`
            *p_i * (rhs_eval - lhs_eval) + lhs_eval
        }
    }
}

impl MleOps<BaseField> for CPUBackend {
    fn eval_at_point(mle: &Mle<Self, BaseField>, point: &[BaseField]) -> BaseField {
        assert_eq!(point.len(), mle.num_variables());
        eval_mle_at_point(mle, point)
    }

    fn fix_first(mle: Mle<Self, BaseField>, assignment: SecureField) -> Mle<Self, SecureField> {
        let (lhs_evals, rhs_evals) = mle.split_at(mle.len() / 2);
        let mut fixed_evals = Vec::new();

        for (&lhs_eval, &rhs_eval) in zip(lhs_evals, rhs_evals) {
            // `eval = eq(0, assignment) * lhs + eq(1, assignment) * rhs`
            let eval = assignment * (rhs_eval - lhs_eval) + lhs_eval;
            fixed_evals.push(eval);
        }

        Mle::new(fixed_evals)
    }
}

impl MleOps<SecureField> for CPUBackend {
    fn eval_at_point(mle: &Mle<Self, SecureField>, point: &[SecureField]) -> SecureField {
        assert_eq!(point.len(), mle.num_variables());
        eval_mle_at_point(mle, point)
    }

    fn fix_first(mle: Mle<Self, SecureField>, assignment: SecureField) -> Mle<Self, SecureField> {
        let n_fixed_evals = mle.len() / 2;
        let mut evals = mle.into_evals();

        for i in 0..n_fixed_evals {
            let lhs = evals[i];
            let rhs = evals[i + n_fixed_evals];
            // `evals[i] = eq(0, assignment) * lhs + eq(1, assignment) * rhs`
            evals[i] += assignment * (rhs - lhs);
        }

        evals.truncate(n_fixed_evals);

        Mle::new(evals)
    }
}

impl SumcheckOracle for Mle<CPUBackend, SecureField> {
    fn num_variables(&self) -> usize {
        self.num_variables()
    }

    fn univariate_sum(&self, claim: SecureField) -> Polynomial<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let (lhs_evals, _rhs_evals) = self.split_at(self.len() / 2);

        let y0 = lhs_evals.iter().sum();
        let y1 = claim - y0;

        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        self.fix_first(challenge)
    }
}
