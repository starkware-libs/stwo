use crate::prover::backend::Column;
use crate::prover::lookups::mle::{Mle, MleOps};
use crate::core::{
    fields::{m31::BaseField, qm31::SecureField},
};

use crate::{
    stwo_cuda::secure_field_vec::SecureFieldVec,
    prover::backend::cuda::CudaBackend,
};

use crate::stwo_cuda as interface;

impl MleOps<BaseField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>,
    {
        let evals_size = mle.len();
        let result_evals = SecureFieldVec::new_uninitialized(evals_size >> 1);
        unsafe {
            interface::bindings::fix_first_variable_base_field(
                mle.into_evals().device_ptr,
                evals_size,
                assignment.into(),
                result_evals.device_ptr,
            )
        }
        Mle::new(result_evals)
    }
}

impl MleOps<SecureField> for CudaBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>,
    {
        let evals_size = mle.len();
        let result_evals = SecureFieldVec::new_uninitialized(evals_size >> 1);
        unsafe {
            interface::bindings::fix_first_variable_secure_field(
                mle.into_evals().device_ptr,
                evals_size,
                assignment.into(),
                result_evals.device_ptr,
            )
        }
        Mle::new(result_evals)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::lookups::mle::{Mle, MleOps};
    use crate::core::{
        fields::{m31::BaseField, qm31::SecureField},
    };

    use crate::stwo_cuda::{
        base_field_vec::BaseFieldVec, secure_field_vec::SecureFieldVec
    };
    use crate::prover::backend::cuda::CudaBackend;

    #[test]
    fn fix_first_variable_with_base_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;

        let values = (0..1 << N_VARIABLES).map(BaseField::from).collect_vec();

        let mle_cuda = Mle::<CudaBackend, BaseField>::new(BaseFieldVec::from_vec(values.clone()));
        let mle_cpu = Mle::<CpuBackend, BaseField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = MleOps::<BaseField>::fix_first_variable(mle_cpu, random_assignment);

        let mle_fixed_simd = MleOps::<BaseField>::fix_first_variable(mle_cuda, random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }

    #[test]
    fn fix_first_variable_with_secure_field_mle_matches_cpu() {
        const N_VARIABLES: u32 = 8;

        let values = (0..(1 << N_VARIABLES) * 4)
            .collect_vec()
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();

        let mle_cuda =
            Mle::<CudaBackend, SecureField>::new(SecureFieldVec::from_vec(values.clone()));
        let mle_cpu = Mle::<CpuBackend, SecureField>::new(values);
        let random_assignment = SecureField::from_u32_unchecked(7, 12, 3, 2);
        let mle_fixed_cpu = MleOps::<SecureField>::fix_first_variable(mle_cpu, random_assignment);

        let mle_fixed_simd = MleOps::<SecureField>::fix_first_variable(mle_cuda, random_assignment);

        assert_eq!(mle_fixed_simd.into_evals().to_cpu(), *mle_fixed_cpu)
    }
}
