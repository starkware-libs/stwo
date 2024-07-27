use cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use stwo_prover::core::backend::{Col, Column, ColumnOps};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::MerkleOps;

use crate::{CudaBackend, CudaColumn, CUDA_CTX};

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = CudaColumn<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        if prev_layer.is_some() {
            unimplemented!();
        }

        let kernel = CUDA_CTX
            .get_func("blake2s", "commit_layer_no_parent")
            .unwrap();
        let cfg = LaunchConfig::for_num_elems(1 << log_size);

        // Allocate pointer buf.
        let col_ptrs = CUDA_CTX
            .htod_copy(
                columns
                    .iter()
                    .map(|column| *column.buffer.device_ptr())
                    .collect(),
            )
            .unwrap();

        let mut res = Col::<Self, Blake2sHash>::zeros(1 << log_size);
        unsafe {
            kernel.clone().launch(
                cfg,
                (&mut res.buffer, &col_ptrs, 1 << log_size, columns.len()),
            )
        }
        .unwrap();
        res
    }
}

#[test]
fn test_blake2s_no_parent() {
    use stwo_prover::core::backend::CpuBackend;

    const LOG_SIZE: u32 = 10;
    let cols: Vec<Col<CudaBackend, BaseField>> = (0..35)
        .map(|i| {
            (0..(1 << LOG_SIZE))
                .map(|j| BaseField::from(i * j))
                .collect()
        })
        .collect();

    let actual = CudaBackend::commit_on_layer(LOG_SIZE, None, &cols.iter().collect::<Vec<_>>());

    let cols: Vec<_> = cols.iter().map(|c| c.to_cpu()).collect();
    let expected = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
        LOG_SIZE,
        None,
        &cols.iter().collect::<Vec<_>>(),
    );

    assert_eq!(actual.to_cpu(), expected);
}
