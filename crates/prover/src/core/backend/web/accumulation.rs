use super::WebBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::backend::simd::SimdBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;

// WARNING: This works because they are literally the same object layout.
//
// The only difference is the backend methods.
// When we implement all methods for WebGPU,
// we will no longer need this to convert back/forth.
impl AsMut<SecureColumnByCoords<SimdBackend>> for SecureColumnByCoords<WebBackend> {
    fn as_mut(&mut self) -> &mut SecureColumnByCoords<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsRef<SecureColumnByCoords<SimdBackend>> for SecureColumnByCoords<WebBackend> {
    fn as_ref(&self) -> &SecureColumnByCoords<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

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
