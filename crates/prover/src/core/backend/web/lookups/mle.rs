use crate::core::backend::simd::SimdBackend;
use crate::core::backend::web::WebBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::{Mle, MleOps};

impl MleOps<BaseField> for WebBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        SimdBackend::fix_first_variable(mle.into(), assignment).into()
    }
}

impl MleOps<SecureField> for WebBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        SimdBackend::fix_first_variable(mle.into(), assignment).into()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::web::WebBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::channel::Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::mle::Mle;
    use crate::core::test_utils::test_channel;

    #[test]
    fn fix_first_variable_with_secure_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;
        let values = test_channel().draw_secure_felts(1 << N_VARIABLES);
        let mle_simd = Mle::<WebBackend, SecureField>::new(values.iter().copied().collect());
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
        let mle_simd = Mle::<WebBackend, BaseField>::new(values.iter().copied().collect());
        let mle_cpu = Mle::<CpuBackend, BaseField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = mle_cpu.fix_first_variable(random_assignment);

        let mle_fixed_simd = mle_simd.fix_first_variable(random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }
}
