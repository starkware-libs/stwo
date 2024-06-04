use super::GpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation};
use crate::core::poly::BitReversedOrder;

impl QuotientOps for GpuBackend {
    fn accumulate_quotients(
        _domain: CircleDomain,
        _columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        _random_coeff: SecureField,
        _sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        todo!()
    }
}
