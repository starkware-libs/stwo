use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::FieldOps;

use crate::CudaBackend;

impl FieldOps<BaseField> for CudaBackend {
    fn batch_inverse(_column: &Self::Column, _dst: &mut Self::Column) {
        todo!()
    }
}
