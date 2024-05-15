use std::iter::zip;

use num_traits::{One, Zero};

use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::UnivariatePoly;

impl MleOps<BaseField> for SimdBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let midpoint = mle.len() / 2;

        // Use CPU backend to to prevent dealing with instances smaller than `PackedSecureField`.
        if midpoint < N_LANES {
            let cpu_mle = Mle::<CpuBackend, BaseField>::new(mle.to_cpu());
            let cpu_res = cpu_mle.fix_first_variable(assignment);
            return Mle::new(cpu_res.into_evals().into_iter().collect());
        }

        let assignment = PackedSecureField::broadcast(assignment);
        let packed_midpoint = midpoint / N_LANES;
        let (lhs_evals, rhs_evals) = mle.data.split_at(packed_midpoint);

        let res = zip(lhs_evals, rhs_evals)
            // Equivalent to `eq(0, assignment) * lhs_eval + eq(1, assignment) * rhs_eval`.
            .map(|(&lhs_eval, &rhs_eval)| assignment * (rhs_eval - lhs_eval) + lhs_eval)
            .collect();

        Mle::new(res)
    }
}

impl MleOps<SecureField> for SimdBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let midpoint = mle.len() / 2;

        // Use CPU backend to to prevent dealing with instances smaller than `PackedSecureField`.
        if midpoint < N_LANES {
            let cpu_mle = Mle::<CpuBackend, SecureField>::new(mle.to_cpu());
            let cpu_res = cpu_mle.fix_first_variable(assignment);
            return Mle::new(cpu_res.into_evals().into_iter().collect());
        }

        let packed_midpoint = midpoint / N_LANES;
        let assignment = PackedSecureField::broadcast(assignment);
        let mut packed_evals = mle.into_evals().data;

        for i in 0..packed_midpoint {
            let lhs_eval = packed_evals[i];
            let rhs_eval = packed_evals[i + packed_midpoint];
            // Equivalent to `eq(0, assignment) * lhs_eval + eq(1, assignment) * rhs_eval`.
            packed_evals[i] += assignment * (rhs_eval - lhs_eval);
        }

        packed_evals.truncate(packed_midpoint);

        let length = packed_evals.len() * N_LANES;
        let data = packed_evals;

        Mle::new(SecureFieldVec { data, length })
    }
}

impl MultivariatePolyOracle for Mle<SimdBackend, SecureField> {
    fn n_variables(&self) -> usize {
        self.n_variables()
    }

    fn sum_as_poly_in_first_variable(&self, claim: SecureField) -> UnivariatePoly<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let midpoint = self.len() / 2;

        // Use CPU backend to to prevent dealing with instances smaller than `PackedSecureField`.
        if midpoint < N_LANES {
            let cpu_mle = Mle::<CpuBackend, SecureField>::new(self.to_cpu());
            return cpu_mle.sum_as_poly_in_first_variable(claim);
        }

        let packed_midpoint = midpoint / N_LANES;

        let y0 = self.data[..packed_midpoint]
            .iter()
            .sum::<PackedSecureField>()
            .pointwise_sum();
        let y1 = claim - y0;

        UnivariatePoly::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first_variable(self, challenge: SecureField) -> Self {
        self.fix_first_variable(challenge)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use crate::core::backend::simd::SimdBackend;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::sumcheck::{partially_verify, prove_batch};
    use crate::core::test_utils::test_channel;

    #[test]
    fn sumcheck_works() {
        const N_VARIABLES: u32 = 6;
        let values = test_channel().draw_felts(1 << N_VARIABLES);
        let claim = values.iter().sum();
        let mle = Mle::<SimdBackend, SecureField>::new(values.into_iter().collect());
        let lambda = SecureField::one();
        let (proof, ..) = prove_batch(vec![claim], vec![mle.clone()], lambda, &mut test_channel());

        let (assignment, eval) = partially_verify(claim, &proof, &mut test_channel()).unwrap();

        assert_eq!(eval, mle.eval_at_point(&assignment));
    }
}
