#![feature(lazy_cell)]
use std::fmt::Debug;
use std::sync::{Arc, LazyLock};

use bytemuck::{cast_vec, Pod, Zeroable};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use stwo_prover::core::backend::{Column, ColumnOps};

pub struct CudaBackend;
static CUDA_CTX: LazyLock<Arc<CudaDevice>> = LazyLock::new(|| CudaDevice::new(0).unwrap());

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

impl<T: Clone + Debug + Pod + std::marker::Unpin> ColumnOps<T> for CudaBackend {
    type Column = CudaColumn<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        todo!()
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
