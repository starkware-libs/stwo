use crate::core::fields::qm31::SecureField;
use crate::prover::backend::cuda::secure_column::CudaSecureColumn;
use crate::prover::backend::cuda::CudaBackend;
use crate::prover::backend::CpuBackend;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;
use crate::stwo_cuda::bindings;

impl AccumulationOps for CudaBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        unsafe {
            bindings::accumulate(
                column.len() as u32,
                CudaSecureColumn::from(column).device_ptr(),
                CudaSecureColumn::from(other).device_ptr(),
            )
        }
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        CpuBackend::generate_secure_powers(felt, n_powers)
    }

    fn lift_and_accumulate(
        cols: Vec<SecureColumnByCoords<Self>>,
    ) -> Option<SecureColumnByCoords<Self>> {
        if cols.is_empty() {
            return None;
        }

        const INITIAL_SIZE: usize = 2;
        assert!(
            cols[0].len() >= INITIAL_SIZE,
            "A column must be of length at least {INITIAL_SIZE}.",
        );
        let mut curr: SecureColumnByCoords<CudaBackend> = SecureColumnByCoords::zeros(INITIAL_SIZE);
        for col in cols.into_iter() {
            let log_ratio = col.len().ilog2() - curr.len().ilog2();
            unsafe {
                bindings::lift_and_accumulate(
                    col.len() as u32,
                    col.columns[0].device_ptr,
                    col.columns[1].device_ptr,
                    col.columns[2].device_ptr,
                    col.columns[3].device_ptr,
                    curr.columns[0].device_ptr,
                    curr.columns[1].device_ptr,
                    curr.columns[2].device_ptr,
                    curr.columns[3].device_ptr,
                    log_ratio,
                );
            }
            curr = col;
        }
        Some(curr)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::{M31, P};
    use crate::prover::backend::cuda::CudaBackend;
    use crate::prover::backend::Column;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::prover::AccumulationOps;
    use crate::stwo_cuda::base_field_vec::BaseFieldVec;

    #[test]
    fn test_accumulation() {
        let size = 2 << 20;
        let left_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec([M31::from(1)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(2)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(3)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(4)].repeat(size)),
        ];
        let right_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec([M31::from(5)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(6)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(7)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(8)].repeat(size)),
        ];

        let mut left_secure_column = SecureColumnByCoords {
            columns: left_summand,
        };
        let right_secure_column = SecureColumnByCoords {
            columns: right_summand,
        };
        CudaBackend::accumulate(&mut left_secure_column, &right_secure_column);

        let expected_result = [
            [M31::from(6)].repeat(size),
            [M31::from(8)].repeat(size),
            [M31::from(10)].repeat(size),
            [M31::from(12)].repeat(size),
        ];
        assert_eq!(expected_result[0], left_secure_column.columns[0].to_cpu());
        assert_eq!(expected_result[1], left_secure_column.columns[1].to_cpu());
        assert_eq!(expected_result[2], left_secure_column.columns[2].to_cpu());
        assert_eq!(expected_result[3], left_secure_column.columns[3].to_cpu());
    }

    #[test]
    fn test_accumulation_log24() {
        // Test at log_size=24 to check for size-related issues
        let size = 1 << 24;
        let left_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec((0..size).map(|i| M31::from(i as u32)).collect()),
            BaseFieldVec::from_vec((0..size).map(|i| M31::from((i * 2) as u32)).collect()),
            BaseFieldVec::from_vec((0..size).map(|i| M31::from((i * 3) as u32)).collect()),
            BaseFieldVec::from_vec((0..size).map(|i| M31::from((i * 4) as u32)).collect()),
        ];
        let right_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec((0..size).map(|i| M31::from((i * 5) as u32)).collect()),
            BaseFieldVec::from_vec((0..size).map(|i| M31::from((i * 6) as u32)).collect()),
            BaseFieldVec::from_vec((0..size).map(|i| M31::from((i * 7) as u32)).collect()),
            BaseFieldVec::from_vec((0..size).map(|i| M31::from((i * 8) as u32)).collect()),
        ];

        let mut left_secure_column = SecureColumnByCoords {
            columns: left_summand,
        };
        let right_secure_column = SecureColumnByCoords {
            columns: right_summand,
        };
        CudaBackend::accumulate(&mut left_secure_column, &right_secure_column);

        // Check first 1000 elements
        let result0 = left_secure_column.columns[0].to_cpu();
        let result1 = left_secure_column.columns[1].to_cpu();
        let result2 = left_secure_column.columns[2].to_cpu();
        let result3 = left_secure_column.columns[3].to_cpu();

        for i in 0..1000 {
            assert_eq!(
                result0[i],
                M31::from((i + i * 5) as u32),
                "Mismatch at index {} for column 0",
                i
            );
            assert_eq!(
                result1[i],
                M31::from((i * 2 + i * 6) as u32),
                "Mismatch at index {} for column 1",
                i
            );
            assert_eq!(
                result2[i],
                M31::from((i * 3 + i * 7) as u32),
                "Mismatch at index {} for column 2",
                i
            );
            assert_eq!(
                result3[i],
                M31::from((i * 4 + i * 8) as u32),
                "Mismatch at index {} for column 3",
                i
            );
        }

        // Check last 1000 elements
        for i in (size - 1000)..size {
            let expected0 = M31::from(i as u32) + M31::from((i * 5) as u32);
            let expected1 = M31::from((i * 2) as u32) + M31::from((i * 6) as u32);
            let expected2 = M31::from((i * 3) as u32) + M31::from((i * 7) as u32);
            let expected3 = M31::from((i * 4) as u32) + M31::from((i * 8) as u32);
            assert_eq!(
                result0[i], expected0,
                "Mismatch at index {} for column 0",
                i
            );
            assert_eq!(
                result1[i], expected1,
                "Mismatch at index {} for column 1",
                i
            );
            assert_eq!(
                result2[i], expected2,
                "Mismatch at index {} for column 2",
                i
            );
            assert_eq!(
                result3[i], expected3,
                "Mismatch at index {} for column 3",
                i
            );
        }
    }

    /// Test the accumulator finalize sequence when jumping from log_size=20 to log_size=24.
    /// This is the exact scenario that fails in test_prove_verify_all_builtins_cuda_v0.
    #[test]
    fn test_finalize_sequence_20_to_24() {
        use crate::core::fields::m31::BaseField;
        use crate::core::poly::circle::CanonicCoset;
        use crate::prover::backend::CpuBackend;
        use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
        use crate::prover::poly::BitReversedOrder;
        use crate::stwo_cuda::base_field_vec::BaseFieldVec;

        const SMALL_LOG_SIZE: u32 = 20;
        const LARGE_LOG_SIZE: u32 = 24;

        let small_size = 1usize << SMALL_LOG_SIZE;
        let large_size = 1usize << LARGE_LOG_SIZE;

        // Create a polynomial of log_size=20 (like prev_poly in finalize)
        let small_coset = CanonicCoset::new(SMALL_LOG_SIZE);
        let small_domain = small_coset.circle_domain();
        let large_coset = CanonicCoset::new(LARGE_LOG_SIZE);
        let large_domain = large_coset.circle_domain();

        // Create 4 BaseField polynomials for SecureCirclePoly
        let cpu_values: [Vec<BaseField>; 4] = std::array::from_fn(|component| {
            (0..small_size)
                .map(|i| BaseField::from((i * (component + 1)) as u32))
                .collect()
        });

        let gpu_values: [BaseFieldVec; 4] = cpu_values.clone().map(BaseFieldVec::from_vec);

        // Interpolate to get polynomials
        let cpu_small_twiddles = CpuBackend::precompute_twiddles(small_coset.half_coset());
        let gpu_small_twiddles = CudaBackend::precompute_twiddles(small_coset.half_coset());

        let cpu_polys: [CircleCoefficients<CpuBackend>; 4] = cpu_values.map(|v| {
            let eval = CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(small_domain, v);
            CpuBackend::interpolate(eval, &cpu_small_twiddles)
        });

        let gpu_polys: [CircleCoefficients<CudaBackend>; 4] = gpu_values.map(|v| {
            let eval = CircleEvaluation::<CudaBackend, _, BitReversedOrder>::new(small_domain, v);
            CudaBackend::interpolate(eval, &gpu_small_twiddles)
        });

        // Precompute twiddles for large domain
        let cpu_large_twiddles = CpuBackend::precompute_twiddles(large_coset.half_coset());
        let gpu_large_twiddles = CudaBackend::precompute_twiddles(large_coset.half_coset());

        // Evaluate polynomials on large domain (this is what finalize does)
        let cpu_evals: [Vec<BaseField>; 4] =
            cpu_polys.map(|p| CpuBackend::evaluate(&p, large_domain, &cpu_large_twiddles).values);

        let gpu_evals: [Vec<BaseField>; 4] = gpu_polys.map(|p| {
            CudaBackend::evaluate(&p, large_domain, &gpu_large_twiddles)
                .values
                .to_cpu()
        });

        // Compare evaluations
        for i in 0..4 {
            assert_eq!(
                cpu_evals[i].len(),
                large_size,
                "CPU eval {} length mismatch",
                i
            );
            assert_eq!(
                gpu_evals[i].len(),
                large_size,
                "GPU eval {} length mismatch",
                i
            );
            assert_eq!(
                cpu_evals[i][..1000],
                gpu_evals[i][..1000],
                "First 1000 of eval {} mismatch",
                i
            );
            assert_eq!(
                cpu_evals[i][large_size - 1000..],
                gpu_evals[i][large_size - 1000..],
                "Last 1000 of eval {} mismatch",
                i
            );
        }

        // Create values at large_size (like values in finalize)
        let cpu_values_large: [Vec<BaseField>; 4] = std::array::from_fn(|component| {
            (0..large_size)
                .map(|i| BaseField::from((i * (component + 10)) as u32))
                .collect()
        });
        let gpu_values_large: [BaseFieldVec; 4] =
            cpu_values_large.clone().map(BaseFieldVec::from_vec);

        // Accumulate (eval + values)
        let mut cpu_accumulated: [Vec<BaseField>; 4] = cpu_values_large.clone();
        for i in 0..4 {
            for j in 0..large_size {
                cpu_accumulated[i][j] += cpu_evals[i][j];
            }
        }

        let mut gpu_secure_col = SecureColumnByCoords {
            columns: gpu_values_large,
        };
        let gpu_eval_secure_col = SecureColumnByCoords {
            columns: gpu_evals.map(BaseFieldVec::from_vec),
        };
        CudaBackend::accumulate(&mut gpu_secure_col, &gpu_eval_secure_col);

        // Compare accumulated values by checking the first and last elements (without consuming)
        for (i, cpu_acc_col) in cpu_accumulated.iter().enumerate() {
            let gpu_col = gpu_secure_col.columns[i].to_cpu();
            assert_eq!(
                cpu_acc_col[..1000],
                gpu_col[..1000],
                "First 1000 of accumulated {} mismatch",
                i
            );
            assert_eq!(
                cpu_acc_col[large_size - 1000..],
                gpu_col[large_size - 1000..],
                "Last 1000 of accumulated {} mismatch",
                i
            );
        }

        // Now interpolate the accumulated values to get final polynomial
        let cpu_final_polys: [CircleCoefficients<CpuBackend>; 4] = cpu_accumulated.map(|v| {
            let eval = CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(large_domain, v);
            CpuBackend::interpolate(eval, &cpu_large_twiddles)
        });

        let gpu_accumulated_cols: [BaseFieldVec; 4] = gpu_secure_col.columns;
        let gpu_final_polys: [CircleCoefficients<CudaBackend>; 4] = gpu_accumulated_cols.map(|v| {
            let eval = CircleEvaluation::<CudaBackend, _, BitReversedOrder>::new(large_domain, v);
            CudaBackend::interpolate(eval, &gpu_large_twiddles)
        });

        // Compare final polynomial coefficients
        for i in 0..4 {
            let cpu_coeffs = &cpu_final_polys[i].coeffs;
            let gpu_coeffs = gpu_final_polys[i].coeffs.to_cpu();
            assert_eq!(
                cpu_coeffs.len(),
                gpu_coeffs.len(),
                "Coeffs length mismatch for poly {}",
                i
            );
            assert_eq!(
                cpu_coeffs[..1000],
                gpu_coeffs[..1000],
                "First 1000 coeffs of poly {} mismatch",
                i
            );
            assert_eq!(
                cpu_coeffs[large_size - 1000..],
                gpu_coeffs[large_size - 1000..],
                "Last 1000 coeffs of poly {} mismatch",
                i
            );
        }

        println!("test_finalize_sequence_20_to_24: All checks passed!");
    }

    #[test]
    fn test_accumulation_m31_arithmetic() {
        let size = 2 << 20;
        let left_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec([M31::from(1)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(1)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(1)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(1)].repeat(size)),
        ];
        let right_summand: [BaseFieldVec; 4] = [
            BaseFieldVec::from_vec([M31::from(P - 1)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(P - 1)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(P - 1)].repeat(size)),
            BaseFieldVec::from_vec([M31::from(P - 1)].repeat(size)),
        ];

        let mut left_secure_column = SecureColumnByCoords {
            columns: left_summand,
        };
        let right_secure_column = SecureColumnByCoords {
            columns: right_summand,
        };
        CudaBackend::accumulate(&mut left_secure_column, &right_secure_column);

        let expected_result = [
            [M31::from(0)].repeat(size),
            [M31::from(0)].repeat(size),
            [M31::from(0)].repeat(size),
            [M31::from(0)].repeat(size),
        ];
        assert_eq!(expected_result[0], left_secure_column.columns[0].to_cpu());
        assert_eq!(expected_result[1], left_secure_column.columns[1].to_cpu());
        assert_eq!(expected_result[2], left_secure_column.columns[2].to_cpu());
        assert_eq!(expected_result[3], left_secure_column.columns[3].to_cpu());
    }

    /// Test polynomial extend + evaluate at log_size 19 (two-stage NTT kernels).
    /// This tests the hypothesis that the bug is in CUDA NTT for log_n >= 13.
    #[test]
    fn test_poly_extend_evaluate_log19() {
        use crate::core::fields::m31::BaseField;
        use crate::core::poly::circle::CanonicCoset;
        use crate::prover::backend::CpuBackend;
        use crate::prover::poly::circle::{CircleCoefficients, PolyOps};
        use crate::stwo_cuda::base_field_vec::BaseFieldVec;

        const SMALL_LOG_SIZE: u32 = 7; // Small polynomial
        const LARGE_LOG_SIZE: u32 = 19; // Evaluate on large domain (uses two-stage NTT)

        let small_size = 1usize << SMALL_LOG_SIZE;
        let large_size = 1usize << LARGE_LOG_SIZE;

        println!(
            "Testing poly extend+evaluate: {} -> {} ({}x expansion)",
            SMALL_LOG_SIZE,
            LARGE_LOG_SIZE,
            large_size / small_size
        );

        // Create a small polynomial with some non-trivial coefficients
        let cpu_coeffs: Vec<BaseField> = (0..small_size)
            .map(|i| BaseField::from((i * 7 + 13) as u32))
            .collect();
        let gpu_coeffs = BaseFieldVec::from_vec(cpu_coeffs.clone());

        let cpu_poly = CircleCoefficients::<CpuBackend>::new(cpu_coeffs);
        let gpu_poly = CircleCoefficients::<CudaBackend>::new(gpu_coeffs);

        println!(
            "CPU poly log_size: {}, coeffs.len: {}",
            cpu_poly.log_size(),
            cpu_poly.coeffs.len()
        );
        println!(
            "GPU poly log_size: {}, coeffs.len: {}",
            gpu_poly.log_size(),
            gpu_poly.coeffs.len()
        );

        // Compute twiddles for the large domain
        let large_coset = CanonicCoset::new(LARGE_LOG_SIZE);
        let large_domain = large_coset.circle_domain();

        let cpu_twiddles = CpuBackend::precompute_twiddles(large_coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(large_coset.half_coset());

        // Evaluate both polynomials on the large domain
        println!(
            "Evaluating CPU poly on domain log_size={}...",
            LARGE_LOG_SIZE
        );
        let cpu_eval = CpuBackend::evaluate(&cpu_poly, large_domain, &cpu_twiddles);

        println!(
            "Evaluating GPU poly on domain log_size={}...",
            LARGE_LOG_SIZE
        );
        let gpu_eval = CudaBackend::evaluate(&gpu_poly, large_domain, &gpu_twiddles);

        // Compare results
        let cpu_values = &cpu_eval.values;
        let gpu_values = gpu_eval.values.to_cpu();

        assert_eq!(cpu_values.len(), gpu_values.len(), "Eval length mismatch");
        println!("Comparing {} evaluation points...", cpu_values.len());

        let mut diff_count = 0;
        let mut first_diff_idx = None;
        for i in 0..cpu_values.len() {
            if cpu_values[i] != gpu_values[i] {
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
                diff_count += 1;
            }
        }

        if diff_count > 0 {
            println!(
                "MISMATCH: {} differences out of {} (first at index {})",
                diff_count,
                cpu_values.len(),
                first_diff_idx.unwrap()
            );
            let idx = first_diff_idx.unwrap();
            println!("  cpu_eval[{}] = {:?}", idx, cpu_values[idx]);
            println!("  gpu_eval[{}] = {:?}", idx, gpu_values[idx]);

            // Also check a few more indices
            for check_idx in [0, 1, 16, 128, 256, large_size / 2, large_size - 1] {
                if check_idx < large_size {
                    println!(
                        "  [{}]: CPU={:?}, GPU={:?}, match={}",
                        check_idx,
                        cpu_values[check_idx],
                        gpu_values[check_idx],
                        cpu_values[check_idx] == gpu_values[check_idx]
                    );
                }
            }
        } else {
            println!("SUCCESS: All {} evaluation points match!", cpu_values.len());
        }

        assert_eq!(diff_count, 0, "CPU and GPU evaluations should match!");
    }

    /// Test polynomial extend + evaluate at log_size 12 (single-stage NTT).
    /// This should PASS since single-stage NTT works.
    #[test]
    fn test_poly_extend_evaluate_log12() {
        use crate::core::fields::m31::BaseField;
        use crate::core::poly::circle::CanonicCoset;
        use crate::prover::backend::CpuBackend;
        use crate::prover::poly::circle::{CircleCoefficients, PolyOps};
        use crate::stwo_cuda::base_field_vec::BaseFieldVec;

        const SMALL_LOG_SIZE: u32 = 7;
        const LARGE_LOG_SIZE: u32 = 12; // Uses single-stage NTT

        let small_size = 1usize << SMALL_LOG_SIZE;
        let large_size = 1usize << LARGE_LOG_SIZE;

        println!(
            "Testing poly extend+evaluate: {} -> {} ({}x expansion)",
            SMALL_LOG_SIZE,
            LARGE_LOG_SIZE,
            large_size / small_size
        );

        let cpu_coeffs: Vec<BaseField> = (0..small_size)
            .map(|i| BaseField::from((i * 7 + 13) as u32))
            .collect();
        let gpu_coeffs = BaseFieldVec::from_vec(cpu_coeffs.clone());

        let cpu_poly = CircleCoefficients::<CpuBackend>::new(cpu_coeffs);
        let gpu_poly = CircleCoefficients::<CudaBackend>::new(gpu_coeffs);

        let large_coset = CanonicCoset::new(LARGE_LOG_SIZE);
        let large_domain = large_coset.circle_domain();

        let cpu_twiddles = CpuBackend::precompute_twiddles(large_coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(large_coset.half_coset());

        let cpu_eval = CpuBackend::evaluate(&cpu_poly, large_domain, &cpu_twiddles);
        let gpu_eval = CudaBackend::evaluate(&gpu_poly, large_domain, &gpu_twiddles);

        let cpu_values = &cpu_eval.values;
        let gpu_values = gpu_eval.values.to_cpu();

        assert_eq!(cpu_values.len(), gpu_values.len(), "Eval length mismatch");

        let mut diff_count = 0;
        for i in 0..cpu_values.len() {
            if cpu_values[i] != gpu_values[i] {
                diff_count += 1;
            }
        }

        println!(
            "Result: {} differences out of {}",
            diff_count,
            cpu_values.len()
        );
        assert_eq!(
            diff_count, 0,
            "CPU and GPU evaluations should match for log_size 12!"
        );
    }

    /// Test the exact finalize sequence with two components of very different sizes.
    /// This mimics what happens with rc_6 (log_size=7) + rc_9_9 (log_size=19).
    #[test]
    fn test_finalize_sequence_7_to_19() {
        use crate::core::fields::m31::BaseField;
        use crate::core::poly::circle::CanonicCoset;
        use crate::prover::backend::CpuBackend;
        use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
        use crate::prover::poly::BitReversedOrder;
        use crate::stwo_cuda::base_field_vec::BaseFieldVec;

        const SMALL_LOG_SIZE: u32 = 7;
        const LARGE_LOG_SIZE: u32 = 19;

        let small_size = 1usize << SMALL_LOG_SIZE;
        let large_size = 1usize << LARGE_LOG_SIZE;

        println!(
            "Testing finalize sequence: {} -> {}",
            SMALL_LOG_SIZE, LARGE_LOG_SIZE
        );

        // Precompute twiddles for the large domain (like finalize does)
        let large_coset = CanonicCoset::new(LARGE_LOG_SIZE);
        let cpu_twiddles = CpuBackend::precompute_twiddles(large_coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(large_coset.half_coset());

        // Step 1: Create values at small log_size (simulates sub_accumulations[7])
        let small_domain = CanonicCoset::new(SMALL_LOG_SIZE).circle_domain();
        let cpu_small_values: [Vec<BaseField>; 4] = std::array::from_fn(|c| {
            (0..small_size)
                .map(|i| BaseField::from(((i * (c + 1) * 7) % 2147483647) as u32))
                .collect()
        });
        let gpu_small_values: [BaseFieldVec; 4] =
            cpu_small_values.clone().map(BaseFieldVec::from_vec);

        // Step 2: Interpolate to get prev_poly (simulates creating cur_poly at log_size=7)
        println!(
            "Step 2: Interpolating small values at log_size={}...",
            SMALL_LOG_SIZE
        );
        let cpu_prev_poly: [CircleCoefficients<CpuBackend>; 4] = cpu_small_values.map(|v| {
            let eval = CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(small_domain, v);
            eval.interpolate_with_twiddles(&cpu_twiddles)
        });
        let gpu_prev_poly: [CircleCoefficients<CudaBackend>; 4] = gpu_small_values.map(|v| {
            let eval =
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(small_domain, v);
            CudaBackend::interpolate(eval, &gpu_twiddles)
        });

        // Verify prev_poly matches
        for i in 0..4 {
            let cpu_coeffs = &cpu_prev_poly[i].coeffs;
            let gpu_coeffs = gpu_prev_poly[i].coeffs.to_cpu();
            assert_eq!(
                cpu_coeffs.len(),
                gpu_coeffs.len(),
                "prev_poly[{}] size mismatch",
                i
            );
            let mismatches: Vec<_> = (0..cpu_coeffs.len())
                .filter(|&j| cpu_coeffs[j] != gpu_coeffs[j])
                .collect();
            if !mismatches.is_empty() {
                println!(
                    "prev_poly[{}] has {} mismatches: {:?}",
                    i,
                    mismatches.len(),
                    &mismatches[..mismatches.len().min(10)]
                );
            }
            assert!(
                mismatches.is_empty(),
                "prev_poly[{}] should match after interpolation",
                i
            );
        }
        println!("Step 2: prev_poly interpolation matches!");

        // Step 3: Create values at large log_size (simulates sub_accumulations[19])
        let large_domain = CanonicCoset::new(LARGE_LOG_SIZE).circle_domain();
        let cpu_large_values: [Vec<BaseField>; 4] = std::array::from_fn(|c| {
            (0..large_size)
                .map(|i| BaseField::from(((i * (c + 10) * 13) % 2147483647) as u32))
                .collect()
        });
        let gpu_large_values: [BaseFieldVec; 4] =
            cpu_large_values.clone().map(BaseFieldVec::from_vec);

        // Step 4: Evaluate prev_poly on large domain
        println!(
            "Step 4: Evaluating prev_poly on domain log_size={}...",
            LARGE_LOG_SIZE
        );
        let cpu_eval: [Vec<BaseField>; 4] =
            cpu_prev_poly.map(|p| CpuBackend::evaluate(&p, large_domain, &cpu_twiddles).values);
        let gpu_eval: [Vec<BaseField>; 4] = gpu_prev_poly.map(|p| {
            CudaBackend::evaluate(&p, large_domain, &gpu_twiddles)
                .values
                .to_cpu()
        });

        // Verify evaluation matches
        for i in 0..4 {
            let mismatches: Vec<_> = (0..large_size)
                .filter(|&j| cpu_eval[i][j] != gpu_eval[i][j])
                .collect();
            if !mismatches.is_empty() {
                println!(
                    "eval[{}] has {} mismatches (first 10): {:?}",
                    i,
                    mismatches.len(),
                    &mismatches[..mismatches.len().min(10)]
                );
                println!(
                    "  first mismatch at {}: CPU={:?}, GPU={:?}",
                    mismatches[0], cpu_eval[i][mismatches[0]], gpu_eval[i][mismatches[0]]
                );
            }
            assert!(
                mismatches.is_empty(),
                "eval[{}] should match after evaluation",
                i
            );
        }
        println!("Step 4: prev_poly evaluation matches!");

        // Step 5: Accumulate (values = values + eval)
        println!("Step 5: Accumulating...");
        let mut cpu_accumulated: [Vec<BaseField>; 4] = cpu_large_values.clone();
        for i in 0..4 {
            for j in 0..large_size {
                cpu_accumulated[i][j] += cpu_eval[i][j];
            }
        }

        let mut gpu_values_col = SecureColumnByCoords {
            columns: gpu_large_values,
        };
        let gpu_eval_col = SecureColumnByCoords {
            columns: gpu_eval.map(BaseFieldVec::from_vec),
        };
        CudaBackend::accumulate(&mut gpu_values_col, &gpu_eval_col);

        // Verify accumulation matches
        for (i, cpu_acc_col) in cpu_accumulated.iter().enumerate() {
            let gpu_acc = gpu_values_col.columns[i].to_cpu();
            let mismatches: Vec<_> = (0..large_size)
                .filter(|&j| cpu_acc_col[j] != gpu_acc[j])
                .collect();
            if !mismatches.is_empty() {
                println!(
                    "accumulated[{}] has {} mismatches (first 10): {:?}",
                    i,
                    mismatches.len(),
                    &mismatches[..mismatches.len().min(10)]
                );
            }
            assert!(mismatches.is_empty(), "accumulated[{}] should match", i);
        }
        println!("Step 5: Accumulation matches!");

        // Step 6: Final interpolation
        println!("Step 6: Final interpolation...");
        let cpu_final: [CircleCoefficients<CpuBackend>; 4] = cpu_accumulated.map(|v| {
            let eval = CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(large_domain, v);
            eval.interpolate_with_twiddles(&cpu_twiddles)
        });
        let gpu_accumulated_cols: [BaseFieldVec; 4] = gpu_values_col.columns;
        let gpu_final: [CircleCoefficients<CudaBackend>; 4] = gpu_accumulated_cols.map(|v| {
            let eval =
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(large_domain, v);
            CudaBackend::interpolate(eval, &gpu_twiddles)
        });

        // Verify final polynomial matches
        for i in 0..4 {
            let cpu_coeffs = &cpu_final[i].coeffs;
            let gpu_coeffs = gpu_final[i].coeffs.to_cpu();
            let mismatches: Vec<_> = (0..large_size)
                .filter(|&j| cpu_coeffs[j] != gpu_coeffs[j])
                .collect();
            if !mismatches.is_empty() {
                println!(
                    "final_poly[{}] has {} mismatches (first 10): {:?}",
                    i,
                    mismatches.len(),
                    &mismatches[..mismatches.len().min(10)]
                );
                println!(
                    "  first mismatch at {}: CPU={:?}, GPU={:?}",
                    mismatches[0], cpu_coeffs[mismatches[0]], gpu_coeffs[mismatches[0]]
                );
            }
            assert!(mismatches.is_empty(), "final_poly[{}] should match", i);
        }
        println!("Step 6: Final interpolation matches!");

        println!("test_finalize_sequence_7_to_19: All steps passed!");
    }

    /// Test interpolation at log_size=19 with actual constraint evaluation values.
    /// This tests if CUDA interpolation produces different results than SIMD
    /// when given the same input values.
    #[test]
    fn test_interpolation_log19_with_real_values() {
        use crate::core::fields::m31::BaseField;
        use crate::core::poly::circle::CanonicCoset;
        use crate::prover::backend::CpuBackend;
        use crate::prover::poly::circle::{CircleEvaluation, PolyOps};
        use crate::prover::poly::BitReversedOrder;
        use crate::stwo_cuda::base_field_vec::BaseFieldVec;

        const LOG_SIZE: u32 = 19;
        let size = 1usize << LOG_SIZE;

        println!("Testing interpolation at log_size={}", LOG_SIZE);

        // Create values that mimic actual constraint evaluation results
        // Use a pattern that might trigger bugs (not just sequential numbers)
        let values: Vec<BaseField> = (0..size)
            .map(|i| {
                // Mix of different patterns to simulate real constraint values
                let val = ((i as u64 * 1863506683u64) % 2147483647) as u32;
                BaseField::from(val)
            })
            .collect();

        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();

        // CPU interpolation
        let cpu_eval = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            domain,
            values.clone().into_iter().collect(),
        );
        let cpu_twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let cpu_poly = cpu_eval.interpolate_with_twiddles(&cpu_twiddles);
        let cpu_coeffs = cpu_poly.coeffs.to_vec();

        // GPU interpolation
        let gpu_eval = CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
            domain,
            BaseFieldVec::from_vec(values),
        );
        let gpu_twiddles = CudaBackend::precompute_twiddles(domain.half_coset);
        let gpu_poly = gpu_eval.interpolate_with_twiddles(&gpu_twiddles);
        let gpu_coeffs = gpu_poly.coeffs.to_cpu();

        // Compare
        assert_eq!(
            cpu_coeffs.len(),
            gpu_coeffs.len(),
            "Coefficient count mismatch"
        );

        let mut diff_count = 0;
        let mut first_diff_idx = None;
        for i in 0..cpu_coeffs.len() {
            if cpu_coeffs[i] != gpu_coeffs[i] {
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
                diff_count += 1;
            }
        }

        if diff_count > 0 {
            println!(
                "MISMATCH: {} differences out of {}",
                diff_count,
                cpu_coeffs.len()
            );
            if let Some(idx) = first_diff_idx {
                println!(
                    "First mismatch at index {}: CPU={:?}, GPU={:?}",
                    idx, cpu_coeffs[idx], gpu_coeffs[idx]
                );
                // Print surrounding values
                let start = idx.saturating_sub(4);
                let end = (idx + 5).min(cpu_coeffs.len());
                println!("Context [{}..{}]:", start, end);
                for j in start..end {
                    println!(
                        "  [{}]: CPU={:?}, GPU={:?} {}",
                        j,
                        cpu_coeffs[j],
                        gpu_coeffs[j],
                        if j == idx { "<-- FIRST DIFF" } else { "" }
                    );
                }
            }
        } else {
            println!("SUCCESS: All {} coefficients match!", cpu_coeffs.len());
        }

        assert_eq!(
            diff_count, 0,
            "CPU and GPU interpolation should match at log_size=19!"
        );
    }

    #[test]
    fn test_lift_and_accumulate_compared_with_cpu() {
        use crate::prover::backend::CpuBackend;

        // Test with multiple columns of increasing sizes (like real accumulator usage)
        let sizes: Vec<usize> = vec![1 << 4, 1 << 5, 1 << 6, 1 << 7];

        let cpu_cols: Vec<SecureColumnByCoords<CpuBackend>> = sizes
            .iter()
            .map(|&size| {
                let columns: [Vec<crate::core::fields::m31::BaseField>; 4] =
                    std::array::from_fn(|c| {
                        (0..size)
                            .map(|i| M31::from(((i * (c + 1) * 7 + c * 13) % 2147483647) as u32))
                            .collect()
                    });
                SecureColumnByCoords { columns }
            })
            .collect();

        let gpu_cols: Vec<SecureColumnByCoords<CudaBackend>> = cpu_cols
            .iter()
            .map(|col| SecureColumnByCoords {
                columns: col.columns.clone().map(BaseFieldVec::from_vec),
            })
            .collect();

        let cpu_result = CpuBackend::lift_and_accumulate(cpu_cols).unwrap();
        let gpu_result = CudaBackend::lift_and_accumulate(gpu_cols).unwrap();

        let cpu_vals = cpu_result.to_vec();
        let gpu_vals_on_cpu = gpu_result.to_cpu().to_vec();
        assert_eq!(
            cpu_vals, gpu_vals_on_cpu,
            "lift_and_accumulate GPU vs CPU mismatch"
        );
    }

    /// Test if SIMD and CUDA twiddles are identical at log_size=19
    #[test]
    fn test_twiddles_match_log19() {
        use crate::core::poly::circle::CanonicCoset;
        use crate::prover::backend::CpuBackend;
        use crate::prover::poly::circle::PolyOps;

        const LOG_SIZE: u32 = 19;

        println!("Testing twiddles at log_size={}", LOG_SIZE);

        let coset = CanonicCoset::new(LOG_SIZE).circle_domain().half_coset;

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset);
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset);

        // Compare twiddles
        let cpu_tw = cpu_twiddles.twiddles.to_vec();
        let gpu_tw = gpu_twiddles.twiddles.to_cpu();

        assert_eq!(cpu_tw.len(), gpu_tw.len(), "Twiddles length mismatch");

        let mut diff_count = 0;
        let mut first_diff_idx = None;
        for i in 0..cpu_tw.len() {
            if cpu_tw[i] != gpu_tw[i] {
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
                diff_count += 1;
            }
        }

        if diff_count > 0 {
            println!(
                "TWIDDLES MISMATCH: {} differences out of {}",
                diff_count,
                cpu_tw.len()
            );
            if let Some(idx) = first_diff_idx {
                println!(
                    "First mismatch at index {}: CPU={:?}, GPU={:?}",
                    idx, cpu_tw[idx], gpu_tw[idx]
                );
            }
        } else {
            println!("SUCCESS: All {} twiddles match!", cpu_tw.len());
        }

        assert_eq!(
            diff_count, 0,
            "CPU and GPU twiddles should match at log_size=19!"
        );
    }
}
