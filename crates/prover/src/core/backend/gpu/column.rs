use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync};
use cudarc::nvrtc::compile_ptx;
use itertools::Itertools;

use super::{GpuBackend, DEVICE};
use crate::core::backend::Column;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::FieldOps;

impl FieldOps<BaseField> for GpuBackend {
    fn batch_inverse(from: &Self::Column, dst: &mut Self::Column) {
        let size = from.len();
        let log_size = u32::BITS - (size as u32).leading_zeros() - 1;

        let config = Self::launch_config_for_num_elems(size as u32 >> 1, 256, 512 * 4 * 2);
        let batch_inverse = DEVICE
            .get_func("column", "batch_inverse_basefield")
            .unwrap();
        unsafe {
            let res = batch_inverse.launch(
                config,
                (
                    from.as_slice(),
                    dst.as_mut_slice(),
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
    fn batch_inverse(from: &Self::Column, dst: &mut Self::Column) {
        let size = from.len();
        let log_size = u32::BITS - (size as u32).leading_zeros() - 1;

        let config = Self::launch_config_for_num_elems(size as u32 >> 1, 512, 1024 * 4 * 4 * 2);
        let batch_inverse = DEVICE
            .get_func("column", "batch_inverse_secure_field")
            .unwrap();
        unsafe {
            let res = batch_inverse.launch(
                config,
                (
                    from.as_slice(),
                    dst.as_mut_slice(),
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
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        BaseFieldCudaColumn::new(DEVICE.htod_copy(iter.into_iter().collect_vec()).unwrap())
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
pub struct SecureFieldCudaColumn(CudaSlice<QM31>);

#[allow(unused)]
impl SecureFieldCudaColumn {
    pub fn new(column: CudaSlice<QM31>) -> Self {
        Self(column)
    }

    pub fn from_vec(column: Vec<QM31>) -> Self {
        Self(DEVICE.htod_sync_copy(&column).unwrap())
    }

    pub fn as_mut_slice(&mut self) -> &mut CudaSlice<QM31> {
        &mut self.0
    }

    pub fn as_slice(&self) -> &CudaSlice<QM31> {
        &self.0
    }
}

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
        DEVICE.dtoh_sync_copy(&self.0).unwrap()
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
    device
        .load_ptx(
            ptx,
            "column",
            &["batch_inverse_basefield", "batch_inverse_secure_field"],
        )
        .unwrap();
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::gpu::column::{BaseFieldCudaColumn, SecureFieldCudaColumn};
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::{SecureField, QM31};
    use crate::core::fields::FieldOps;

    #[test]
    fn test_batch_inverse_basefield() {
        let size: usize = 1 << 24;
        let from = (1..(size + 1) as u32).map(|x| M31(x)).collect_vec();
        let dst = from.clone();
        let mut dst_expected = dst.clone();
        CpuBackend::batch_inverse(&from, &mut dst_expected);

        let from_device = BaseFieldCudaColumn::from_vec(from);
        let mut dst_device = BaseFieldCudaColumn::from_vec(dst);
        <GpuBackend as FieldOps<BaseField>>::batch_inverse(&from_device, &mut dst_device);

        assert_eq!(dst_device.to_cpu(), dst_expected.to_cpu());
    }

    #[test]
    fn test_batch_inverse_secure_field() {
        let size: usize = 1 << 25;

        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();

        let from_cpu = from_raw
            .chunks(4)
            .map(|a| QM31::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();
        let mut dst_expected_cpu = from_cpu.clone();

        CpuBackend::batch_inverse(&from_cpu, &mut dst_expected_cpu);

        let from_device = SecureFieldCudaColumn::from_vec(from_cpu.clone());
        let mut dst_device = SecureFieldCudaColumn::from_vec(from_cpu.clone());

        <GpuBackend as FieldOps<SecureField>>::batch_inverse(&from_device, &mut dst_device);

        assert_eq!(dst_device.to_cpu(), dst_expected_cpu);
    }
}
