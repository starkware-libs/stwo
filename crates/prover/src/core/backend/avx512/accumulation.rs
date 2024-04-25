use super::qm31::PackedSecureField;
use super::AVX512Backend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::poly::circle::SecureEvaluation;

impl AccumulationOps for AVX512Backend {
    fn accumulate(column: &mut SecureColumn<Self>, other: &SecureColumn<Self>) {
        for i in 0..column.n_packs() {
            let res_coeff = column.packed_at(i) + other.packed_at(i);
            unsafe { column.set_packed(i, res_coeff) };
        }
    }

    fn mul_and_accumulate(
        column: &mut SecureEvaluation<Self>,
        other: &SecureEvaluation<Self>,
        factor: SecureField,
    ) {
        let broadcasted_factor = PackedSecureField::broadcast(factor);
        for i in 0..column.n_packs() {
            let res_coeff = column.packed_at(i) * broadcasted_factor + other.packed_at(i);
            unsafe { column.set_packed(i, res_coeff) };
        }
    }
}
