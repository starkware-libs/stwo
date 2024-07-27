use cudarc::driver::{LaunchAsync, LaunchConfig};
use stwo_prover::core::air::accumulation::AccumulationOps;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;

use crate::{CudaBackend, CUDA_CTX};

impl AccumulationOps for CudaBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        assert_eq!(column.len(), other.len());
        let kernel = CUDA_CTX
            .get_func("accumulate", "accumulate_kernel")
            .unwrap();
        let n = column.len() as u32;
        let cfg = LaunchConfig::for_num_elems(n);
        for i in 0..4 {
            unsafe {
                kernel.clone().launch(
                    cfg,
                    (&mut column.columns[i].buffer, &other.columns[i].buffer, n),
                )
            }
            .unwrap();
        }
    }
}

#[test]
fn test_accumulate() {
    use stwo_prover::core::backend::CpuBackend;
    use stwo_prover::core::fields::m31::BaseField;
    fn add_for_test<B: AccumulationOps>(log_size: u32) -> SecureColumnByCoords<B> {
        let mut secure_col0 = SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                (0..(1 << log_size))
                    .map(|j| BaseField::from(i + j))
                    .collect()
            }),
        };
        let secure_col1 = SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                (0..(1 << log_size))
                    .map(|j| BaseField::from(i + j + 100))
                    .collect()
            }),
        };

        B::accumulate(&mut secure_col0, &secure_col1);
        secure_col0
    }

    for log_size in 10..=16 {
        let res_cuda = add_for_test::<CudaBackend>(log_size);
        let res_cpu = add_for_test::<CpuBackend>(log_size);

        assert_eq!(
            res_cuda.to_cpu().into_iter().collect::<Vec<_>>(),
            res_cpu.into_iter().collect::<Vec<_>>(),
            "log_size = {}",
            log_size
        );
    }
}
