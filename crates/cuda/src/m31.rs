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
    let mut src = src.clone();
    let log_size = src.len().trailing_zeros();
    assert_eq!(src.len(), 1 << log_size);
    assert_eq!(src.len(), dst.len());

    // Copy src to dst.
    CUDA_CTX
        .dtod_copy(&mut src.buffer, &mut dst.buffer)
        .unwrap();

    // Upsweep.
    let upsweep = CUDA_CTX.get_func("batch_inv", upsweep_kernel).unwrap();
    for i in (0..log_size).rev() {
        // From i+1 to i.
        unsafe {
            upsweep.clone().launch(
                LaunchConfig::for_num_elems(1 << i),
                (&mut src.buffer, &mut dst.buffer, 1 << i),
            )
        }
        .unwrap();
    }

    // Inverse the first element.
    dst.set(0, dst.at(0).inverse());

    // Downsweep.
    let downsweep = CUDA_CTX.get_func("batch_inv", downsweep_kernel).unwrap();
    for i in 0..log_size {
        // From i to i+1.
        unsafe {
            downsweep.clone().launch(
                LaunchConfig::for_num_elems(1 << i),
                (&src.buffer, &mut dst.buffer, 1 << i),
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
