use cudarc::driver::{CudaSlice, DeviceSlice};

use super::GpuBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldOps;

impl FieldOps<BaseField> for GpuBackend {
    fn batch_inverse(_column: &Self::Column, _dst: &mut Self::Column) {
        todo!()
    }
}

impl FieldOps<SecureField> for GpuBackend {
    fn batch_inverse(_column: &Self::Column, _dst: &mut Self::Column) {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct BaseFieldCudaColumn(CudaSlice<u32>);

#[allow(unused)]
impl BaseFieldCudaColumn {
    pub fn new(slice: CudaSlice<u32>) -> Self {
        Self(slice)
    }

    pub fn as_slice(&self) -> &CudaSlice<u32> {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut CudaSlice<u32> {
        &mut self.0
    }
}

impl FromIterator<BaseField> for BaseFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = BaseField>>(_iter: T) -> Self {
        todo!()
    }
}

impl Column<BaseField> for BaseFieldCudaColumn {
    fn zeros(_len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        todo!()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn at(&self, _index: usize) -> BaseField {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct SecureFieldCudaColumn([CudaSlice<u32>; 4]);

impl FromIterator<SecureField> for SecureFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = SecureField>>(_iter: T) -> Self {
        todo!()
    }
}

impl Column<SecureField> for SecureFieldCudaColumn {
    fn zeros(_len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        todo!()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn at(&self, _index: usize) -> SecureField {
        todo!()
    }
}
