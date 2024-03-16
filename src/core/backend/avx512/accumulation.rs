use super::qm31::PackedQM31;
use super::AVX512Backend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure::SecureColumn;

impl AccumulationOps for AVX512Backend {
    fn accumulate(column: &mut SecureColumn<Self>, alpha: SecureField, other: &SecureColumn<Self>) {
        let alpha = PackedQM31::broadcast(alpha);
        for i in 0..column.vec_len() {
            let res_coeff = column.get_vec(i) * alpha + other.get_vec(i);
            column.set_vec(i, res_coeff);
        }
    }
}
