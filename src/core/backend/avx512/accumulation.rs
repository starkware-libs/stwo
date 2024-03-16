use super::qm31::PackedQM31;
use super::AVX512Backend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;

impl AccumulationOps for AVX512Backend {
    fn accumulate(column: &mut SecureColumn<Self>, alpha: SecureField, other: &SecureColumn<Self>) {
        let alpha = PackedQM31::broadcast(alpha);
        unsafe {
            for i in 0..column.n_packs() {
                let res_coeff = column.get_packed(i) * alpha + other.get_packed(i);
                column.set_packed(i, res_coeff);
            }
        }
    }
}
