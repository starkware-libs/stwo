use crate::core::backend::cpu::lookups::gkr::gen_eq_evals as cpu_gen_eq_evals;
use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::{LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::GkrOps;
use crate::core::lookups::mle::Mle;

impl GkrOps for SimdBackend {
    #[allow(clippy::uninit_vec)]
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        if y.len() < LOG_N_LANES as usize {
            return Mle::new(cpu_gen_eq_evals(y, v).into_iter().collect());
        }

        // Start DP with CPU backend to prevent dealing with instances smaller than a SIMD vector.
        let (y_initial, y_rem) = y.split_last_chunk::<{ LOG_N_LANES as usize }>().unwrap();
        let initial = SecureFieldVec::from_iter(cpu_gen_eq_evals(y_initial, v));
        assert_eq!(initial.len(), N_LANES);

        let packed_len = 1 << y_rem.len();
        let mut data = initial.data;

        data.reserve(packed_len - data.len());
        unsafe { data.set_len(packed_len) };

        for (j, &y_j) in y_rem.iter().rev().enumerate() {
            let packed_y_j = PackedSecureField::broadcast(y_j);

            let (lhs, rhs) = data.split_at_mut(1 << j);

            for i in 0..1 << j {
                // `lhs[i] = eq(0, y_j) * lhs[i]`
                // `rhs[i] = eq(1, y_j) * lhs[i]`
                let tmp = lhs[i] * packed_y_j;
                lhs[i] -= tmp;
                rhs[i] = tmp;
            }
        }

        let length = packed_len * N_LANES;
        Mle::new(SecureFieldVec { data, length })
    }
}

#[cfg(test)]
mod tests {
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::m31::BaseField;
    use crate::core::lookups::gkr::GkrOps;

    #[test]
    fn gen_eq_evals_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5, 6, 1, 1, 9].map(|v| BaseField::from(v).into());
        let cpu_eq_evals = CpuBackend::gen_eq_evals(&y, two);

        let simd_eq_evals = SimdBackend::gen_eq_evals(&y, two);

        assert_eq!(*cpu_eq_evals, simd_eq_evals.to_cpu());
    }
}
