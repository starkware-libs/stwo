use super::column::CM31Column;
use super::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::QuotientOps;

pub struct QuotientConstants {
    pub line_coeffs: Vec<Vec<(SecureField, SecureField, SecureField)>>,
    pub denominator_inverses: Vec<CM31Column>,
}

impl QuotientOps for SimdBackend {
    #[allow(unused_variables)]
    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        unimplemented!()
    }

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
}
