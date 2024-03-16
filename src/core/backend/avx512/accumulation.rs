use super::qm31::PackedQM31;
use super::AVX512Backend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;

impl AccumulationOps for AVX512Backend {
    fn accumulate(column: &mut SecureColumn<Self>, alpha: SecureField, other: &SecureColumn<Self>) {
        let alpha = PackedQM31::broadcast(alpha);
        for i in 0..column.len() {
            let res_coeff = column.packed_at(i) * alpha + other.packed_at(i);
            column.set_packed(i, res_coeff);
        }
    }
}
