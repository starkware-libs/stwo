use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
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

        let config = LaunchConfig::for_num_elems(size as u32 >> 1);
        let batch_inverse = DEVICE.get_func("column", "batch_inverse").unwrap();
        unsafe {
            let mut inner_tree: CudaSlice<M31> = DEVICE.alloc(size).unwrap();
            let res = batch_inverse.launch(
                config,
                (
                    from.as_slice(),
                    dst.as_mut_slice(),
                    &mut inner_tree,
                    size,
                    log_size,
                ),
            );
            res
        }
        .unwrap();
        DEVICE.synchronize().unwrap();
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

    pub fn as_slice(&self) -> &CudaSlice<M31> {
        &self.0
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

pub fn load_batch_inverse_ptx(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("column.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device.load_ptx(ptx, "column", &["batch_inverse"]).unwrap();
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::gpu::column::BaseFieldCudaColumn;
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::FieldOps;

    #[test]
    fn test_batch_inverse() {
        let size: usize = 1 << 12;
        let from = (1..(size + 1) as u32).map(|x| M31(x)).collect_vec();
        let dst = from.clone();
        let mut dst_expected = dst.clone();
        CpuBackend::batch_inverse(&from, &mut dst_expected);

        let from_device = BaseFieldCudaColumn::from_vec(from);
        let mut dst_device = BaseFieldCudaColumn::from_vec(dst);
        <GpuBackend as FieldOps<BaseField>>::batch_inverse(&from_device, &mut dst_device);

        assert_eq!(dst_device.to_cpu(), dst_expected.to_cpu());
    }
}
