use super::CPUBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure::SecureColumn;

impl AccumulationOps for CPUBackend {
    fn accumulate(column: &mut SecureColumn<Self>, alpha: SecureField, other: &SecureColumn<Self>) {
        for i in 0..column.len() {
            let res_coeff = column.at(i) * alpha + other.at(i);
            column.set(i, res_coeff);
        }
    }
}
