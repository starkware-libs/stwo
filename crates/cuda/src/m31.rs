use cudarc::driver::{LaunchAsync, LaunchConfig};
use stwo_prover::core::backend::Column;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::{FieldExpOps, FieldOps};

use crate::{CudaBackend, CUDA_CTX};

impl FieldOps<BaseField> for CudaBackend {
    // TODO(spapini): Use a static allocation for temporary src buffer.
    fn batch_inverse(src: &Self::Column, dst: &mut Self::Column) {
        let mut src = src.clone();
        let log_size = src.len().trailing_zeros();
        assert_eq!(src.len(), 1 << log_size);
        assert_eq!(src.len(), dst.len());

        // Copy src to dst.
        CUDA_CTX
            .dtod_copy(&mut src.buffer, &mut dst.buffer)
            .unwrap();

        // Upsweep.
        let upsweep = CUDA_CTX.get_func("batch_inv", "upsweep_kernel").unwrap();
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
        let downsweep = CUDA_CTX.get_func("batch_inv", "downsweep_kernel").unwrap();
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
}

#[test]
fn test_batch_inv() {
    use crate::CudaColumn;
    const LOG_SIZE: u32 = 10;
    let src: CudaColumn<_> = (1..(1 + (1 << LOG_SIZE))).map(BaseField::from).collect();
    let mut dst: CudaColumn<_> = Column::zeros(1 << LOG_SIZE);
    CudaBackend::batch_inverse(&src, &mut dst);
    let actual = dst.to_cpu();

    let expected: Vec<_> = (1..(1 + (1 << LOG_SIZE)))
        .map(|x| BaseField::from(x).inverse())
        .collect();

    assert_eq!(actual, expected);
}
