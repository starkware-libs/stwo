use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use super::column::{BaseFieldCudaColumn, SecureFieldCudaColumn};
use super::GpuBackend;
use crate::core::backend::gpu::DEVICE;
use crate::core::backend::{Column, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;

// TODO: Do we need to handle columns of sizes larger than 2^32?
impl ColumnOps<BaseField> for GpuBackend {
    type Column = BaseFieldCudaColumn;

    /// Permutes `column` in place according to bit-reversed order.
    /// This method assumes that the length of `column` is 2^k for some k < 32.
    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);
        let bits = u32::BITS - (size as u32).leading_zeros() - 1;

        let config = LaunchConfig::for_num_elems(size as u32);
        let kernel = DEVICE
            .get_func("bit_reverse", "bit_reverse_basefield")
            .unwrap();
        unsafe { kernel.launch(config, (column.as_mut_slice(), size, bits)) }.unwrap();
        DEVICE.synchronize().unwrap();
    }
}

impl ColumnOps<SecureField> for GpuBackend {
    type Column = SecureFieldCudaColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);
        let bits = u32::BITS - (size as u32).leading_zeros() - 1;

        let config = LaunchConfig::for_num_elems(size as u32);
        let kernel = DEVICE
            .get_func("bit_reverse", "bit_reverse_secure_field")
            .unwrap();
        unsafe { kernel.launch(config, (column.as_mut_slice(), size, bits)) }.unwrap();
        DEVICE.synchronize().unwrap();
    }
}

pub fn load_bit_reverse_ptx(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("bit_reverse.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device
        .load_ptx(
            ptx,
            "bit_reverse",
            &["bit_reverse_basefield", "bit_reverse_secure_field"],
        )
        .unwrap();
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::gpu::column::{BaseFieldCudaColumn, SecureFieldCudaColumn};
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::{Column, ColumnOps, CpuBackend};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::{SecureField, QM31};

    #[test]
    fn test_bit_reverse_basefield() {
        let size: usize = 1 << 12;
        let column_data = (0..size as u32).map(|x| M31(x)).collect_vec();
        let mut expected_result = column_data.clone();
        CpuBackend::bit_reverse_column(&mut expected_result);

        let mut column = BaseFieldCudaColumn::from_vec(column_data);
        <GpuBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut column);

        assert_eq!(column.to_cpu(), expected_result);
    }

    #[test]
    fn test_bit_reverse_secure_field() {
        let size: usize = 1 << 12;

        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();
        let from_cpu = from_raw
            .chunks(4)
            .map(|a| QM31::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();
        let mut array_expected = from_cpu.clone();

        CpuBackend::bit_reverse_column(&mut array_expected);

        let mut array = SecureFieldCudaColumn::from_vec(from_cpu.clone());
        <GpuBackend as ColumnOps<SecureField>>::bit_reverse_column(&mut array);

        assert_eq!(array.to_cpu(), array_expected);
    }
}
