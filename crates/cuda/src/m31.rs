use cudarc::driver::{LaunchAsync, LaunchConfig};
use stwo_prover::core::backend::Column;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::{Field, FieldOps};

use crate::{CudaBackend, CudaColumn, CUDA_CTX};

impl FieldOps<BaseField> for CudaBackend {
    // TODO(spapini): Use a static allocation for temporary src buffer.
    fn batch_inverse(src: &Self::Column, dst: &mut Self::Column) {
        batch_inverse(src, dst, "upsweep_m31_kernel", "downsweep_m31_kernel");
    }
}

impl FieldOps<SecureField> for CudaBackend {
    // TODO(spapini): Use a static allocation for temporary src buffer.
    fn batch_inverse(src: &Self::Column, dst: &mut Self::Column) {
        batch_inverse(src, dst, "upsweep_qm31_kernel", "downsweep_qm31_kernel");
    }
}

// TODO(spapini): Use a static allocation for temporary src buffer.
fn batch_inverse<F: Field>(
    src: &CudaColumn<F>,
    dst: &mut CudaColumn<F>,
    upsweep_kernel: &str,
    downsweep_kernel: &str,
) where
    CudaColumn<F>: Column<F>,
{
    let log_size = src.len().trailing_zeros();
    assert_eq!(src.len(), 1 << log_size);
    assert_eq!(src.len(), dst.len());

    // Upsweep.
    let upsweep = CUDA_CTX.get_func("batch_inv", upsweep_kernel).unwrap();
    for i in (0..log_size).rev() {
        // From i+1 to i.
        // Memory layout: 2^i other values, 2^i ith layer, 2^(i+1) (i+1)th layer, ... .
        let bot_layer = dst.buffer.slice(1 << i..2 << i);
        let top_layer = if i == log_size - 1 {
            src.buffer.slice(..)
        } else {
            dst.buffer.slice(2 << i..4 << i)
        };

        unsafe {
            upsweep.clone().launch(
                LaunchConfig::for_num_elems(1 << i),
                (&top_layer, &bot_layer, 1 << i),
            )
        }
        .unwrap();
    }

    // Inverse the root element.
    dst.set(1, dst.at(1).inverse());

    // Downsweep.
    let downsweep = CUDA_CTX.get_func("batch_inv", downsweep_kernel).unwrap();
    for i in 0..log_size {
        // Memory layout: 2^i other values, 2^i ith layer, 2^(i+1) (i+1)th layer, ... .
        let bot_layer = dst.buffer.slice(1 << i..2 << i);
        let top_layer = if i == log_size - 1 {
            src.buffer.slice(..)
        } else {
            dst.buffer.slice(2 << i..4 << i)
        };
        let dst_layer = if i == log_size - 1 {
            dst.buffer.slice(..)
        } else {
            dst.buffer.slice(2 << i..4 << i)
        };

        unsafe {
            downsweep.clone().launch(
                LaunchConfig::for_num_elems(1 << i),
                (
                    &top_layer.slice(..),
                    &bot_layer.slice(..),
                    &dst_layer.slice(..),
                    1 << i,
                ),
            )
        }
        .unwrap();
    }
}

#[test]
fn test_batch_inv_m31() {
    use stwo_prover::core::fields::FieldExpOps;

    use crate::CudaColumn;

    const LOG_SIZE: u32 = 10;
    let src: CudaColumn<_> = (1..(1 + (1 << LOG_SIZE))).map(BaseField::from).collect();
    let mut dst: CudaColumn<_> = Column::zeros(1 << LOG_SIZE);
    <CudaBackend as FieldOps<BaseField>>::batch_inverse(&src, &mut dst);
    let actual = dst.to_cpu();

    let expected: Vec<_> = (1..(1 + (1 << LOG_SIZE)))
        .map(|x| BaseField::from(x).inverse())
        .collect();

    assert_eq!(actual, expected);
}

#[test]
fn test_batch_inv_qm31() {
    use stwo_prover::core::fields::FieldExpOps;

    use crate::CudaColumn;

    const LOG_SIZE: u32 = 10;
    let src: CudaColumn<_> = (0..(1 << LOG_SIZE))
        .map(|i| SecureField::from_u32_unchecked(i, i + 1, i + 2, i + 3))
        .collect();
    let mut dst: CudaColumn<_> = Column::zeros(1 << LOG_SIZE);
    <CudaBackend as FieldOps<SecureField>>::batch_inverse(&src, &mut dst);
    let actual = dst.to_cpu();

    let expected: Vec<_> = (0..(1 << LOG_SIZE))
        .map(|i| SecureField::from_u32_unchecked(i, i + 1, i + 2, i + 3).inverse())
        .collect();

    assert_eq!(actual, expected);
}
