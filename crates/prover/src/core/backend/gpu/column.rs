use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

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
    fn batch_inverse(from: &Self::Column, dst: &mut Self::Column) {
        let size = from.len();
        let log_size = u32::BITS - (size as u32).leading_zeros() - 1;

        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        let mut a_column = from.to_device().unwrap();
        let mut b_column = from.to_device().unwrap();
        let mut c_column = from.to_device().unwrap();

        let config = LaunchConfig::for_num_elems(size as u32);
        let kernel = DEVICE.get_func("column", "batch_inverse").unwrap();
        unsafe { kernel.launch(config, (&mut a_column, &mut b_column, &mut c_column, size, log_size)) }.unwrap();

        dst.inplace_copy_from_slice(&c_column);    
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

    pub fn into_vec(self) -> Vec<M31> {
        self.0
    }

    pub fn to_device(&self) -> Result<CudaSlice<M31>, DriverError> {
        DEVICE.htod_sync_copy(&self.0)
    }
}

impl FromIterator<BaseField> for BaseFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        BaseFieldCudaColumn(iter.into_iter().collect())
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

pub fn load_column_ptx(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("column.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device.load_ptx(ptx, "column", &["batch_inverse"]).unwrap();
}

#[cfg(test)]
mod tests {
    use crate::core::backend::gpu::column::BaseFieldCudaColumn;
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::CpuBackend;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::FieldOps;

    #[test]
    fn test_batch_inverse() {
        //let size: usize = 2048;
        let column = BaseFieldCudaColumn::new(vec![1, 2, 3, 4, 5, 6, 7, 8].into_iter().map(|x| M31(x)).collect::<Vec<_>>());
        let mut expected_result = column.clone().into_vec();

        let mut res = BaseFieldCudaColumn::new(vec![M31(1); 8]);
        CpuBackend::batch_inverse(&column.clone().into_vec(), &mut expected_result);

        <GpuBackend as FieldOps<BaseField>>::batch_inverse(&column, &mut res);
        assert_eq!(res.into_vec(), expected_result);
    }
}
