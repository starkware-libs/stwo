use crate::core::{air::accumulation::AccumulationOps, fields::secure_column::SecureColumn};

use super::GpuBackend;

impl AccumulationOps for GpuBackend {
    fn accumulate(_column: &mut SecureColumn<Self>, _other: &SecureColumn<Self>) {
        todo!()
    }
}
