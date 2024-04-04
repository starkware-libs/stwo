use std::iter::zip;

use num_traits::{One, Zero};

use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::UnivariatePoly;

impl MleOps<BaseField> for CPUBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
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
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
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

impl MultivariatePolyOracle for Mle<CPUBackend, SecureField> {
    fn n_variables(&self) -> usize {
        self.n_variables()
    }

    fn sum_as_poly_in_first_variable(&self, claim: SecureField) -> UnivariatePoly<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let y0 = self[0..self.len() / 2].iter().sum();
        let y1 = claim - y0;

        UnivariatePoly::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first_variable(self, challenge: SecureField) -> Self {
        self.fix_first_variable(challenge)
    }
}
