use std::iter::zip;

use crate::core::backend::cpu::lookups::gkr::gen_eq_evals as cpu_gen_eq_evals;
use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::{LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::{GkrMultivariatePolyOracle, GkrOps, Layer};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::utils::UnivariatePoly;

impl GkrOps for SimdBackend {
    #[allow(clippy::uninit_vec)]
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        if y.len() < LOG_N_LANES as usize {
            return Mle::new(cpu_gen_eq_evals(y, v).into_iter().collect());
        }

        // Start DP with CPU backend to prevent dealing with instances smaller than a SIMD vector.
        let (y_last_chunk, y_rem) = y.split_last_chunk::<{ LOG_N_LANES as usize }>().unwrap();
        let initial = SecureFieldVec::from_iter(cpu_gen_eq_evals(y_last_chunk, v));
        assert_eq!(initial.len(), N_LANES);

        let packed_len = 1 << y_rem.len();
        let mut data = initial.data;

        data.reserve(packed_len - data.len());
        unsafe { data.set_len(packed_len) };

        for (i, &y_j) in y_rem.iter().rev().enumerate() {
            let packed_y_j = PackedSecureField::broadcast(y_j);

            let (lhs_evals, rhs_evals) = data.split_at_mut(1 << i);

            for (lhs, rhs) in zip(lhs_evals, rhs_evals) {
                // Equivalent to:
                // `rhs = eq(1, y_j) * lhs`,
                // `lhs = eq(0, y_j) * lhs`
                *rhs = *lhs * packed_y_j;
                *lhs -= *rhs;
            }
        }

        let length = packed_len * N_LANES;
        Mle::new(SecureFieldVec { data, length })
    }

    fn next_layer(_layer: &Layer<Self>) -> Layer<Self> {
        todo!()
    }

    fn sum_as_poly_in_first_variable(
        _h: &GkrMultivariatePolyOracle<'_, Self>,
        _claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::m31::BaseField;
    use crate::core::lookups::gkr_prover::GkrOps;

    #[test]
    fn gen_eq_evals_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5, 6, 1, 1, 9].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_simd = SimdBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_simd.to_cpu(), *eq_evals_cpu);
    }

    #[test]
    fn gen_eq_evals_with_small_assignment_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_simd = SimdBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_simd.to_cpu(), *eq_evals_cpu);
    }
}
