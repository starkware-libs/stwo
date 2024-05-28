use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::core::{backend::{gpu::DEVICE, Column, ColumnOps}, fields::{m31::BaseField, qm31::SecureField}};

use super::{column::{BaseFieldCudaColumn, SecureFieldCudaColumn}, GpuBackend};



impl ColumnOps<BaseField> for GpuBackend {
    type Column = BaseFieldCudaColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two());
        let bits = usize::BITS - size.leading_zeros() - 1;
        let config = LaunchConfig::for_num_elems(16);
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



#[cfg(test)]
mod tests {
    use super::DEVICE;
    use crate::core::backend::gpu::column::BaseFieldCudaColumn;
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::ColumnOps;
    use crate::core::fields::m31::BaseField;

    #[test]
    fn test_bit_reverse() {
        let size: usize = 16;
        let mut h_column: Vec<u32> = (0..size as u32).collect();
        let mut column: BaseFieldCudaColumn = BaseFieldCudaColumn::new(DEVICE.htod_sync_copy(&h_column).unwrap());

        <GpuBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut column);

        DEVICE
            .dtoh_sync_copy_into(column.as_slice(), &mut h_column)
            .unwrap();
        assert_eq!(
            h_column,
            vec![0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
        );
    }
}