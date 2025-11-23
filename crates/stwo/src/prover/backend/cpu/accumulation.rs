use num_traits::One;

use crate::core::fields::qm31::SecureField;
use crate::prover::backend::cpu::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

impl AccumulationOps for CpuBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        for i in 0..column.len() {
            let res_coeff = column.at(i) + other.at(i);
            column.set(i, res_coeff);
        }
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        (0..n_powers)
            .scan(SecureField::one(), |acc, _| {
                let res = *acc;
                *acc *= felt;
                Some(res)
            })
            .collect()
    }

    fn lift_and_accumulate(
        cols: Vec<SecureColumnByCoords<Self>>,
    ) -> Option<SecureColumnByCoords<Self>> {
        if cols.is_empty() {
            return None;
        };
        const INITIAL_SIZE: usize = 2;
        assert!(
            cols[0].len() >= INITIAL_SIZE,
            "A column must be of length at least {INITIAL_SIZE}.",
        );
        let mut curr = SecureColumnByCoords::zeros(INITIAL_SIZE);
        for mut col in cols.into_iter() {
            let log_ratio = col.len().ilog2() - curr.len().ilog2();
            for i in 0..col.len() {
                let res_coeff = col.at(i) + curr.at((i >> (log_ratio + 1) << 1) + (i & 1));
                col.set(i, res_coeff);
            }
            curr = col;
        }
        Some(curr)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
    use crate::core::fields::FieldExpOps;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs_lifted::test_utils::lift_poly;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::poly::circle::CircleEvaluation;
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::prover::AccumulationOps;
    use crate::qm31;
    #[test]
    fn generate_secure_powers_works() {
        let felt = qm31!(1, 2, 3, 4);
        let n_powers = 10;

        let powers = <CpuBackend as AccumulationOps>::generate_secure_powers(felt, n_powers);

        assert_eq!(powers.len(), n_powers);
        assert_eq!(powers[0], SecureField::one());
        assert_eq!(powers[1], felt);
        assert_eq!(powers[7], felt.pow(7));
    }

    #[test]
    fn generate_empty_secure_powers_works() {
        let felt = qm31!(1, 2, 3, 4);
        let max_log_size = 0;

        let powers = <CpuBackend as AccumulationOps>::generate_secure_powers(felt, max_log_size);

        assert_eq!(powers, vec![]);
    }

    #[test]
    fn test_lift_and_accumulate() {
        const LOG_SIZE_MIN: u32 = 3;
        const N_SECURE_COLS: usize = 4;
        let mut rng = SmallRng::seed_from_u64(0);

        let polys: Vec<CpuCirclePoly> = (0..N_SECURE_COLS * SECURE_EXTENSION_DEGREE)
            .map(|i| {
                CpuCirclePoly::new(
                    (0..1 << (LOG_SIZE_MIN as usize + (i / SECURE_EXTENSION_DEGREE)))
                        .map(|_| M31::from(rng.gen::<u32>()))
                        .collect(),
                )
            })
            .collect();
        // Compute the lifted evaluations and accumulate them by hand.
        let lifted_log_size = polys.iter().map(|p| p.log_size()).max().unwrap();
        let lifted_evals: Vec<CircleEvaluation<_, M31, BitReversedOrder>> = polys
            .iter()
            .map(|p| lift_poly(p, lifted_log_size))
            .collect();
        let mut expected = SecureColumnByCoords::<CpuBackend>::zeros(1 << lifted_log_size);
        for idx in 0..expected.len() {
            let res = lifted_evals
                .iter()
                .map(|eval| eval.values.at(idx))
                .chunks(SECURE_EXTENSION_DEGREE)
                .into_iter()
                .fold(SecureField::zero(), |acc, x| {
                    acc + SecureField::from_m31_array(x.collect_vec().try_into().unwrap())
                });
            expected.set(idx, res);
        }
        // Prepare the inputs to `lift_and_accumulate`.
        let evals: Vec<CpuCircleEvaluation<BaseField, BitReversedOrder>> = polys
            .iter()
            .map(|p| p.evaluate(CanonicCoset::new(p.log_size()).circle_domain()))
            .collect();
        let secure_cols: Vec<SecureColumnByCoords<CpuBackend>> = evals
            .into_iter()
            .map(|eval| eval.values)
            .chunks(SECURE_EXTENSION_DEGREE)
            .into_iter()
            .map(|mut x| SecureColumnByCoords {
                columns: std::array::from_fn(|_| x.next().unwrap()),
            })
            .collect();
        let actual = CpuBackend::lift_and_accumulate(secure_cols).unwrap();

        assert_eq!(actual.columns, expected.columns);
    }
}
