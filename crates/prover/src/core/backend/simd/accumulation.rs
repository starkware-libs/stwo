use super::SimdBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumnByCoords;

impl AccumulationOps for SimdBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        for i in 0..column.packed_len() {
            let res_coeff = unsafe { column.packed_at(i) + other.packed_at(i) };
            unsafe { column.set_packed(i, res_coeff) };
        }
    }
}
