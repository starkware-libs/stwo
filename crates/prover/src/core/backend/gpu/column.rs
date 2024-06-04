use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DeviceRepr, DeviceSlice};

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
    fn batch_inverse(_from: &Self::Column, _dst: &mut Self::Column) {
        todo!()
    }
}

impl FieldOps<SecureField> for GpuBackend {
    fn batch_inverse(_column: &Self::Column, _dst: &mut Self::Column) {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct BaseFieldCudaColumn(CudaSlice<M31>);

#[allow(unused)]
impl BaseFieldCudaColumn {
    pub fn new(column: CudaSlice<M31>) -> Self {
        Self(column)
    }

    pub fn from_vec(column: Vec<M31>) -> Self {
        Self(DEVICE.htod_sync_copy(&column).unwrap())
    }

    pub fn as_mut_slice(&mut self) -> &mut CudaSlice<M31> {
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
        DEVICE.dtoh_sync_copy(&self.0).unwrap()
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
