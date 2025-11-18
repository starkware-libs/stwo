use super::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::QuotientOps;

impl QuotientOps for CpuBackend {
    #[allow(unused_variables)]
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        curr_coeff_power: &mut SecureField,
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn accumulate_denominators(
        accs: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        unimplemented!()
    }
}
