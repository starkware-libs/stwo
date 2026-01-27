use itertools::{zip_eq, Itertools};

use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
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

    /// Receives a collections of columns sorted in strictly ascending order by size and returns
    /// a column which is the coordinate-wise sum of the lifts of the columns. For more information
    /// see the docs in [`crate::prover::backend::simd::accumulation::AccumulationOps`].
    fn lift_and_accumulate(
        cols: Vec<SecureColumnByCoords<Self>>,
    ) -> Option<SecureColumnByCoords<Self>> {
        let mut cols_iter = cols.into_iter();
        let first = cols_iter.next()?;
        assert!(!first.is_empty(), "Columns should be non-empty");

        let mut prev = first;
        for mut col in cols_iter {
            // Perform the lift on the previous accumulation (which is of smaller size) and add it
            // to the current accumulation.
            let log_ratio = col.len().ilog2() - prev.len().ilog2();
            for i in 0..col.len() >> LOG_N_LANES {
                unsafe {
                    let packed_before_lift: [PackedM31; 4] =
                        prev.packed_at(i >> log_ratio).into_packed_m31s();
                    let packed_after_lift: [PackedM31; 4] = std::array::from_fn(|j| {
                        PackedM31::from_simd_unchecked(to_lifted_simd(
                            packed_before_lift[j].into_simd(),
                            log_ratio,
                            i,
                        ))
                    });
                    for (base_column, lift_value) in
                        zip_eq(col.columns.iter_mut(), packed_after_lift)
                    {
                        base_column.data[i] += lift_value;
                    }
                }
            }
            prev = col;
        }
        Some(prev)
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
        const LOG_SIZE_SHORT: u32 = 4;
        const LOG_SIZE_LONG: u32 = 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let col_short = (0..1 << LOG_SIZE_SHORT)
            .map(|_| M31::from(rng.gen::<u32>()))
            .collect_vec();
        let col_long = (0..1 << LOG_SIZE_LONG)
            .map(|_| M31::from(rng.gen::<u32>()))
            .collect_vec();

        // Prepare CPU inputs.
        let secure_col_short = SecureColumnByCoords {
            columns: std::array::from_fn(|_| col_short.clone()),
        };
        let secure_col_long = SecureColumnByCoords {
            columns: std::array::from_fn(|_| col_long.clone()),
        };
        let res_cpu = <CpuBackend as AccumulationOps>::lift_and_accumulate(vec![
            secure_col_short,
            secure_col_long,
        ])
        .unwrap();

        // Prepare SIMD inputs.
        let secure_col_short_simd = SecureColumnByCoords::<SimdBackend> {
            columns: std::array::from_fn(|_| BaseColumn::from_cpu(&col_short)),
        };
        let secure_col_long_simd = SecureColumnByCoords::<SimdBackend> {
            columns: std::array::from_fn(|_| BaseColumn::from_cpu(&col_long)),
        };
        let res_simd = <SimdBackend as AccumulationOps>::lift_and_accumulate(vec![
            secure_col_short_simd,
            secure_col_long_simd,
        ])
        .unwrap();

        assert_eq!(res_cpu.columns, res_simd.to_cpu().columns);
    }
}
