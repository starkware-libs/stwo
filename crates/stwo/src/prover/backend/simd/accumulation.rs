use itertools::Itertools;

use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::m31::{PackedM31, N_LANES};
use crate::prover::backend::simd::qm31::PackedSecureField;
use crate::prover::backend::simd::utils::to_lifted_simd;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

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

    fn lift_and_accumulate(
        column: &mut SecureColumnByCoords<Self>,
        other: &SecureColumnByCoords<Self>,
    ) {
        let log_ratio = column.packed_len().ilog2() - other.packed_len().ilog2();
        unsafe {
            for i in 0..column.packed_len() {
                let packed_before_lift: [PackedM31; 4] =
                    other.packed_at(i >> log_ratio).into_packed_m31s();
                let packed_after_lift: [PackedM31; 4] = std::array::from_fn(|j| {
                    PackedM31::from_simd_unchecked(to_lifted_simd(
                        packed_before_lift[j].into_simd(),
                        log_ratio,
                        i,
                    ))
                });
                let res_coeff =
                    column.packed_at(i) + PackedSecureField::from_packed_m31s(packed_after_lift);

                column.set_packed(i, res_coeff);
            }
        }
    }

    #[allow(unused)]
    fn lift_and_accumulate_v2(
        cols: Vec<SecureColumnByCoords<Self>>,
    ) -> Option<SecureColumnByCoords<Self>> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::fields::m31::M31;
    use crate::prover::backend::cpu::CpuBackend;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::prover::AccumulationOps;
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
                "Error generating secure powers in n_powers = {n_powers}."
            );
        });
    }

    #[test]
    fn test_lift_accumulate_simd() {
        const LOG_SIZE_SHORT: u32 = 6;
        const LOG_SIZE_LONG: u32 = 10;
        let mut rng = SmallRng::seed_from_u64(0);
        let col_short = (0..1 << LOG_SIZE_SHORT)
            .map(|_| M31::from(rng.gen::<u32>()))
            .collect_vec();
        let col_long = (0..1 << LOG_SIZE_LONG)
            .map(|_| M31::from(rng.gen::<u32>()))
            .collect_vec();

        let secure_col_short = SecureColumnByCoords {
            columns: std::array::from_fn(|_| col_short.clone()),
        };
        let mut secure_col_long = SecureColumnByCoords {
            columns: std::array::from_fn(|_| col_long.clone()),
        };
        <CpuBackend as AccumulationOps>::lift_and_accumulate(
            &mut secure_col_long,
            &secure_col_short,
        );

        let secure_col_short_simd = SecureColumnByCoords::<SimdBackend> {
            columns: std::array::from_fn(|_| BaseColumn::from_cpu(col_short.clone())),
        };

        let mut secure_col_long_simd = SecureColumnByCoords::<SimdBackend> {
            columns: std::array::from_fn(|_| BaseColumn::from_cpu(col_long.clone())),
        };

        <SimdBackend as AccumulationOps>::lift_and_accumulate(
            &mut secure_col_long_simd,
            &secure_col_short_simd,
        );

        assert_eq!(
            secure_col_long.columns,
            secure_col_long_simd.to_cpu().columns
        );
    }
}
