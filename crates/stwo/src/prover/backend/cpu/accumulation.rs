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
        column: &mut SecureColumnByCoords<Self>,
        other: &SecureColumnByCoords<Self>,
    ) {
        assert!(column.len() >= 2);
        let log_ratio = column.len().ilog2() - other.len().ilog2();
        for i in 0..column.len() {
            let res_coeff = column.at(i) + other.at((i >> (log_ratio + 1) << 1) + (i & 1));
            column.set(i, res_coeff);
        }
    }

    #[allow(unused_variables)]
    fn lift_and_accumulate_v2(
        cols: Vec<SecureColumnByCoords<Self>>,
    ) -> Option<SecureColumnByCoords<Self>> {
        if cols.is_empty() {
            return None;
        };
        let size = cols.last().as_ref().unwrap().len();
        let mut curr = SecureColumnByCoords::zeros(2);
        for mut col in cols.into_iter() {
            CpuBackend::lift_and_accumulate(&mut col, &curr);
            curr = col;
        }
        Some(curr)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::FieldExpOps;
    use crate::prover::backend::CpuBackend;
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
}
