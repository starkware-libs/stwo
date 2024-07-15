use std::iter::zip;

use num_traits::{One, Zero};

use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::{PackedBaseField, N_LANES};
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

        let packed_assignment = PackedSecureField::broadcast(assignment);
        let packed_midpoint = midpoint / N_LANES;
        let (lhs_evals, rhs_evals) = mle.data.split_at(packed_midpoint);

        let res = zip(lhs_evals, rhs_evals)
            .map(|(&packed_lhs_eval, &packed_rhs_eval)| {
                fold_packed_mle_evals(packed_assignment, packed_lhs_eval, packed_rhs_eval)
            })
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
        let packed_assignment = PackedSecureField::broadcast(assignment);
        let mut packed_evals = mle.into_evals().data;

        for i in 0..packed_midpoint {
            let packed_lhs_eval = packed_evals[i];
            let packed_rhs_eval = packed_evals[i + packed_midpoint];
            packed_evals[i] =
                fold_packed_mle_secure_evals(packed_assignment, packed_lhs_eval, packed_rhs_eval);
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

/// Computes all `eq(0, assignment_i) * lhs_eval_i + eq(1, assignment_i) * rhs_eval_i`.
fn fold_packed_mle_evals(
    packed_assignment: PackedSecureField,
    packed_eval0: PackedBaseField,
    packed_eval1: PackedBaseField,
) -> PackedSecureField {
    packed_assignment * (packed_eval1 - packed_eval0) + packed_eval0
}

/// Computes all `eq(0, assignment_i) * lhs_eval_i + eq(1, assignment_i) * rhs_eval_i`.
// TODO: Consider unifying fold_packed_mle_* functions once we have something like
// AbstractField/AbstractExtensionField traits implemented on packed types.
fn fold_packed_mle_secure_evals(
    packed_assignment: PackedSecureField,
    packed_eval0: PackedSecureField,
    packed_eval1: PackedSecureField,
) -> PackedSecureField {
    packed_assignment * (packed_eval1 - packed_eval0) + packed_eval0
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::channel::Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::mle::Mle;
    use crate::core::test_utils::test_channel;

    #[test]
    fn fix_first_variable_with_secure_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;
        let values = test_channel().draw_felts(1 << N_VARIABLES);
        let mle_simd = Mle::<SimdBackend, SecureField>::new(values.iter().copied().collect());
        let mle_cpu = Mle::<CpuBackend, SecureField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = mle_cpu.fix_first_variable(random_assignment);

        let mle_fixed_simd = mle_simd.fix_first_variable(random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }

    #[test]
    fn fix_first_variable_with_base_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;
        let values = (0..1 << N_VARIABLES).map(BaseField::from).collect_vec();
        let mle_simd = Mle::<SimdBackend, BaseField>::new(values.iter().copied().collect());
        let mle_cpu = Mle::<CpuBackend, BaseField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = mle_cpu.fix_first_variable(random_assignment);

        let mle_fixed_simd = mle_simd.fix_first_variable(random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }
}
