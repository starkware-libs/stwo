use itertools::Itertools;

use crate::core::air::accumulation::AccumulationOps;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::CpuBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;

impl AccumulationOps for SimdBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        for i in 0..column.packed_len() {
            let res_coeff = unsafe { column.packed_at(i) + other.packed_at(i) };
            unsafe { column.set_packed(i, res_coeff) };
        }
    }

    /// Generates the first `n_powers` powers of `felt` using SIMD.
    /// Refer to `CpuBackend::generate_secure_powers` for the scalar CPU implementation.
    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        let base_arr = <CpuBackend as AccumulationOps>::generate_secure_powers(felt, N_LANES)
            .try_into()
            .unwrap();
        let base = PackedSecureField::from_array(base_arr);
        let step = PackedSecureField::broadcast(base_arr[N_LANES - 1] * felt);
        let size = n_powers.div_ceil(N_LANES);

        // Collects the next N_LANES powers of `felt` in each iteration.
        (0..size)
            .scan(base, |acc, _| {
                let res = *acc;
                *acc *= step;
                Some(res)
            })
            .flat_map(|x| x.to_array())
            .take(n_powers)
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::core::air::accumulation::AccumulationOps;
    use crate::core::backend::cpu::CpuBackend;
    use crate::core::backend::simd::SimdBackend;
    use crate::qm31;

    #[test]
    fn test_generate_secure_powers_simd() {
        let felt = qm31!(1, 2, 3, 4);
        let n_powers_vec = [0, 16, 100];

        n_powers_vec.iter().for_each(|&n_powers| {
            let expected = <CpuBackend as AccumulationOps>::generate_secure_powers(felt, n_powers);
            let actual = <SimdBackend as AccumulationOps>::generate_secure_powers(felt, n_powers);
            assert_eq!(
                expected, actual,
                "Error generating secure powers in n_powers = {}.",
                n_powers
            );
        });
    }
}
