use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use super::column::{BaseFieldCudaColumn, SecureFieldCudaColumn};
use super::GpuBackend;
use crate::core::backend::gpu::DEVICE;
use crate::core::backend::{Column, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;

impl ColumnOps<BaseField> for GpuBackend {
    type Column = BaseFieldCudaColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two());
        let bits = usize::BITS - size.leading_zeros() - 1;
        let config = LaunchConfig::for_num_elems(size as u32);
        let kernel = DEVICE.get_func("bit_reverse", "kernel").unwrap();
        unsafe { kernel.launch(config, (column.as_mut_slice(), size, bits)) }.unwrap();
    }
}

impl ColumnOps<SecureField> for GpuBackend {
    type Column = SecureFieldCudaColumn;

    fn bit_reverse_column(_column: &mut Self::Column) {
        todo!()
    }
}

pub fn load_bit_reverse_ptx(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("bit_reverse.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device.load_ptx(ptx, "bit_reverse", &["kernel"]).unwrap();
}

#[cfg(test)]
mod tests {
    use super::DEVICE;
    use crate::core::backend::gpu::column::BaseFieldCudaColumn;
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::{ColumnOps, CpuBackend};
    use crate::core::fields::m31::BaseField;

    #[test]
    fn test_bit_reverse() {
        let size: usize = 128;
        let mut h_column: Vec<u32> = (0..size as u32).collect();
        let mut column: BaseFieldCudaColumn =
            BaseFieldCudaColumn::new(DEVICE.htod_sync_copy(&h_column).unwrap());

        <GpuBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut column);
        DEVICE
            .dtoh_sync_copy_into(column.as_slice(), &mut h_column)
            .unwrap();

        let mut expected_result: Vec<u32> = (0..size as u32).collect();
        CpuBackend::bit_reverse_column(&mut expected_result);

        assert_eq!(h_column, expected_result);
    }
}
