use crate::core::fields::qm31::SecureField;
use crate::prover::backend::cuda::CudaBackend;
use crate::prover::lookups::gkr_prover::GkrOps;
use crate::prover::lookups::mle::Mle;
use crate::stwo_cuda::bindings;
use crate::stwo_cuda::bindings::CudaSecureField;
use crate::stwo_cuda::secure_field_vec::SecureFieldVec;

#[allow(unused_variables)]
impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let y_size = y.len();
        let result_evals = SecureFieldVec::new_uninitialized(1 << y_size);

        unsafe {
            bindings::gen_eq_evals(
                v.into(),
                y.as_ptr() as *const CudaSecureField,
                y_size as u32,
                result_evals.device_ptr as *const CudaSecureField,
                result_evals.size as u32,
            );
        }

        Mle::new(result_evals)
    }

    fn next_layer(
        layer: &crate::prover::lookups::gkr_prover::Layer<Self>,
    ) -> crate::prover::lookups::gkr_prover::Layer<Self> {
        todo!()
    }

    fn sum_as_poly_in_first_variable(
        h: &crate::prover::lookups::gkr_prover::GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> crate::prover::lookups::utils::UnivariatePoly<SecureField> {
        todo!()
    }
}

mod tests {

    #[test]
    fn gen_eq_evals_matches_cpu() {
        use itertools::Itertools;

        use crate::core::fields::m31::BaseField;
        use crate::core::fields::qm31::SecureField;
        use crate::prover::backend::cuda::CudaBackend;
        use crate::prover::backend::{Column, CpuBackend};
        use crate::prover::lookups::gkr_prover::GkrOps;

        let two = BaseField::from(2).into();

        let from_raw = [7, 3, 5, 6, 1, 1, 9].repeat(4);
        let y = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();

        let cpu_eq_evals = CpuBackend::gen_eq_evals(&y, two);
        let gpu_eq_evals = CudaBackend::gen_eq_evals(&y, two);

        assert_eq!(gpu_eq_evals.to_cpu(), *cpu_eq_evals);
    }
}
