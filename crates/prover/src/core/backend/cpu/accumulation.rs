use super::CPUBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::poly::circle::SecureEvaluation;

impl AccumulationOps for CPUBackend {
    fn accumulate(column: &mut SecureColumn<Self>, other: &SecureColumn<Self>) {
        for i in 0..column.len() {
            let res_coeff = column.at(i) + other.at(i);
            column.set(i, res_coeff);
        }
    }

    fn mul_and_accumulate(
        column: &mut SecureEvaluation<Self>,
        other: &SecureEvaluation<Self>,
        factor: crate::core::fields::qm31::SecureField,
    ) {
        for i in 0..column.len() {
            let res_coeff = column.at(i) * factor + other.at(i);
            column.set(i, res_coeff);
        }
    }
}
