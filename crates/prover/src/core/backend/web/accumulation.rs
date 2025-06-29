use super::WebBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::backend::simd::SimdBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::secure_column::SecureColumnByCoords;

impl AccumulationOps for WebBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        SimdBackend::accumulate(column.as_mut(), other.as_ref());
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        SimdBackend::generate_secure_powers(felt, n_powers)
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
