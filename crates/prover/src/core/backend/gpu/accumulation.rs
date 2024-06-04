use super::GpuBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumn;

impl AccumulationOps for GpuBackend {
    fn accumulate(_column: &mut SecureColumn<Self>, _other: &SecureColumn<Self>) {
        todo!()
    }
}
