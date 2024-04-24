use std::iter::zip;

use stwo_verifier::core::fields::m31::BaseField;
use stwo_verifier::core::fields::qm31::SecureField;

use crate::core::backend::CPUBackend;
use crate::core::lookups::mle::{Mle, MleOps};

impl MleOps<BaseField> for CPUBackend {
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
