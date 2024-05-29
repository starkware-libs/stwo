use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DeviceRepr, DriverError};

use super::{GpuBackend, DEVICE};
use crate::core::backend::Column;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldOps;

unsafe impl DeviceRepr for M31 {
    fn as_kernel_param(&self) -> *mut c_void {
        self.0 as *const Self as *mut c_void
    }
}

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
pub struct BaseFieldCudaColumn(Vec<M31>);

#[allow(unused)]
impl BaseFieldCudaColumn {
    pub fn new(column: Vec<M31>) -> Self {
        Self(column)
    }

    pub fn inplace_copy_from_slice(&mut self, cuda_slice: &CudaSlice<M31>) {
        DEVICE.dtoh_sync_copy_into(cuda_slice, &mut self.0);
    }

    pub fn to_vec(self) -> Vec<M31> {
        self.0
    }

    pub fn to_device(&self) -> Result<CudaSlice<M31>, DriverError> {
        DEVICE.htod_sync_copy(&self.0)
    }
}


impl FromIterator<BaseField> for BaseFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        BaseFieldCudaColumn(iter.into_iter().collect::<Vec<_>>())
    }
}

impl Column<BaseField> for BaseFieldCudaColumn {
    fn zeros(len: usize) -> Self {
        Self(vec![M31::default(); len])
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.0.clone()
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
