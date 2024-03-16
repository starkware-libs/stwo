use super::AVX512Backend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumn;

impl AccumulationOps for AVX512Backend {
    fn accumulate(column: &mut SecureColumn<Self>, other: &SecureColumn<Self>) {
        for i in 0..column.n_packs() {
            let res_coeff = column.packed_at(i) + other.packed_at(i);
            column.set_packed(i, res_coeff);
        }
    }
}
