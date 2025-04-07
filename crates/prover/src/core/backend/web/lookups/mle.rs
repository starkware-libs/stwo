use crate::core::backend::simd::SimdBackend;
use crate::core::backend::web::WebBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::lookups::mle::{Mle, MleOps};

// WARNING: This works because they are literally the same object layout.
//
// The only difference is the backend methods.
// When we implement all methods for WebGPU,
// we will no longer need this to convert back/forth.
impl Into<Mle<SimdBackend, BaseField>> for Mle<WebBackend, BaseField> {
    fn into(self) -> Mle<SimdBackend, BaseField> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<Mle<SimdBackend, QM31>> for Mle<WebBackend, QM31> {
    fn into(self) -> Mle<SimdBackend, QM31> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<Mle<WebBackend, BaseField>> for Mle<SimdBackend, BaseField> {
    fn into(self) -> Mle<WebBackend, BaseField> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<Mle<WebBackend, QM31>> for Mle<SimdBackend, QM31> {
    fn into(self) -> Mle<WebBackend, QM31> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

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
        let values = test_channel().draw_felts(1 << N_VARIABLES);
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
