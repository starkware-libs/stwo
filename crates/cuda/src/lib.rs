#![feature(lazy_cell)]
mod accumulation;
mod m31;

use std::fmt::Debug;
use std::sync::{Arc, LazyLock};

use bytemuck::{cast_vec, Pod, Zeroable};
use cudarc::driver::{
    CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use stwo_prover::core::backend::{Column, ColumnOps, CpuBackend};
use stwo_prover::core::fields::m31::BaseField;

pub struct CudaBackend;
static CUDA_CTX: LazyLock<Arc<CudaDevice>> = LazyLock::new(|| {
    let device = CudaDevice::new(0).unwrap();

    let ptx = cudarc::nvrtc::compile_ptx(include_str!("kernels/bit_rev.cu")).unwrap();
    device
        .load_ptx(ptx, "bit_rev", &["bit_rev_kernel"])
        .unwrap();

    let ptx = cudarc::nvrtc::compile_ptx(include_str!("kernels/accumulate.cu")).unwrap();
    device
        .load_ptx(ptx, "accumulate", &["accumulate_kernel"])
        .unwrap();

    device
});

#[derive(Clone, Debug)]
pub struct CudaColumn<T> {
    buffer: CudaSlice<CudaWrappedValue<T>>,
}
// impl Clone for CudaColumn<CudaBaseField> {
//     fn clone(&self) -> Self {}
// }
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(transparent)]
pub struct CudaWrappedValue<T>(pub T);
unsafe impl<T> DeviceRepr for CudaWrappedValue<T> {}
unsafe impl<T> ValidAsZeroBits for CudaWrappedValue<T> {}

impl ColumnOps<BaseField> for CudaBackend {
    type Column = CudaColumn<BaseField>;

    fn bit_reverse_column(column: &mut Self::Column) {
        let log_size = column.len().trailing_zeros();
        assert_eq!(column.len(), 1 << log_size);
        assert!(log_size >= 10);
        let m_bits = log_size - 10;

        let kernel = CUDA_CTX.get_func("bit_rev", "bit_rev_kernel").unwrap();
        let cfg = LaunchConfig {
            grid_dim: (1 << m_bits, 1, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        };
        unsafe { kernel.launch(cfg, (&mut column.buffer, m_bits)) }.unwrap();
    }
}

impl<T: Clone + Debug + Pod + std::marker::Unpin> Column<T> for CudaColumn<T> {
    fn zeros(len: usize) -> Self {
        Self {
            buffer: CUDA_CTX.alloc_zeros(len).unwrap(),
        }
    }

    fn to_cpu(&self) -> Vec<T> {
        cast_vec(CUDA_CTX.dtoh_sync_copy(&self.buffer).unwrap())
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn at(&self, index: usize) -> T {
        CUDA_CTX
            .dtoh_sync_copy(&self.buffer.slice(index..(index + 1)))
            .unwrap()[0]
            .0
    }

    fn set(&mut self, index: usize, value: T) {
        CUDA_CTX
            .htod_sync_copy_into(
                &[CudaWrappedValue(value)],
                &mut self.buffer.slice_mut(index..(index + 1)),
            )
            .unwrap();
    }
}
impl<T: Clone + Debug + Pod + std::marker::Unpin> FromIterator<T> for CudaColumn<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<_> = iter.into_iter().map(CudaWrappedValue).collect();
        let buffer = CUDA_CTX.htod_copy(vec).unwrap();
        Self { buffer }
    }
}

#[test]
fn test_buffers() {
    use stwo_prover::core::fields::m31::BaseField;
    let src = vec![BaseField::from(1), BaseField::from(2), BaseField::from(3)];
    let dst = CudaColumn::from_iter(src.clone());
    assert_eq!(src, dst.to_cpu());
}

#[test]
fn test_bit_reverse() {
    use stwo_prover::core::fields::m31::BaseField;
    for log_size in 10..=16 {
        let mut data: CudaColumn<_> = (0..(1 << log_size)).map(BaseField::from).collect();
        CudaBackend::bit_reverse_column(&mut data);
        let actual = data.to_cpu();

        let mut data: Vec<_> = (0..(1 << log_size)).map(BaseField::from).collect();
        CpuBackend::bit_reverse_column(&mut data);

        assert_eq!(actual, data, "log_size = {}", log_size);
    }
}
