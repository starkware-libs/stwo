use crate::core::fields::m31::BaseField;
use crate::core::pcs::quotients::{ColumnSampleBatch, quotient_constants};
use crate::core::poly::circle::CanonicCoset;
use crate::prover::QuotientOps;
use crate::prover::backend::Column;
use crate::prover::backend::cuda::CudaBackend;
use crate::prover::backend::cuda::secure_column::CudaSecureColumn;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::simd::column::BaseColumn as SimdBaseColumn;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::BitReversedOrder;
use crate::prover::poly::circle::{
    CircleCoefficients, CircleEvaluation, PolyOps, SecureEvaluation,
};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::stwo_cuda as interface;
use crate::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::stwo_cuda::bindings::{CirclePointSecureField, CudaSecureField};

// NATIVE (device-resident) LIFTED QuotientOps for CudaBackend, matching the lifted quotient scheme
// of SimdBackend (crates/stwo/src/prover/backend/simd/quotients.rs).
//
// The lifted scheme is expressed as Rust orchestration of EXISTING NitrooZK device kernels plus the
// device-resident NTT (CudaBackend::interpolate / CudaBackend::evaluate). No new CUDA was written.
//
//   accumulate_numerators: NitrooZK's `accumulate_numerators_batch` kernel run over the SUBDOMAIN
//   (first `size >> log_blowup_factor` rows, in bit-reversed order — a prefix of the committed
//   device column), mirroring simd `accumulate_numerators_on_subdomain`.
//
//   compute_quotients_and_combine: NitrooZK's `compute_quotients_and_combine` kernel run over the
//   SUBDOMAIN (giving the quotient on the subdomain, mirroring simd
// `compute_quotients_and_combine`),   then the on-device lift: for each of the 4 secure coords,
// interpolate the subdomain eval ->   coeffs and evaluate on the full domain.
//
// SAFETY OF THE PORT:
// * The combine kernel's lifting index `(row >> (log_ratio+1) << 1) + (row & 1)` reproduces simd
//   `to_lifted_simd`'s scalar source-index mapping for all log_ratios.
// * The kernel's `domain_at_index(half_coset.initial_index, half_coset.step_size,
//   bit_reverse(row))` convention matches simd's `CircleDomainBitRevIterator` (==
//   `domain.at(bit_reverse_index(row))`).
// * The denominator-inverse and numerator math in both kernels matches simd line-for-line (see the
//   per-function comments below).
//
// Small subdomains (`subdomain.log_size() < LOG_N_LANES == 4`) are handled by delegating to
// SimdBackend, which itself falls back to CPU for that case — this reproduces simd's exact small
// path instead of relying on CudaBackend NTT's own (separate) <=3 CPU fallback.

const LOG_N_LANES: u32 = 4;

fn eval_cuda_to_simd(
    c: &CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    CircleEvaluation::new(c.domain, c.values.to_cpu().into_iter().collect())
}

fn securecol_cuda_to_simd(
    c: &SecureColumnByCoords<CudaBackend>,
) -> SecureColumnByCoords<SimdBackend> {
    SecureColumnByCoords {
        columns: std::array::from_fn(|i| {
            c.columns[i].to_cpu().into_iter().collect::<SimdBaseColumn>()
        }),
    }
}

fn securecol_simd_to_cuda(
    c: SecureColumnByCoords<SimdBackend>,
) -> SecureColumnByCoords<CudaBackend> {
    SecureColumnByCoords { columns: c.columns.map(|col| BaseFieldVec::from_vec(col.to_cpu())) }
}

fn twiddles_cuda_to_simd(t: &TwiddleTree<CudaBackend>) -> TwiddleTree<SimdBackend> {
    // RECOMPUTE the SIMD twiddles from the root coset rather than converting the CudaBackend
    // buffers: NitrooZK's CUDA twiddle layout is NOT guaranteed to match SimdBackend's internal
    // (bit-reversed, layered) Vec<u32> layout, so a raw value-copy would feed the SIMD quotient
    // code mis-laid-out twiddles. Twiddles are a deterministic function of the coset.
    SimdBackend::precompute_twiddles(t.root_coset)
}

// === SIMD-delegated path (used for small subdomains, `< LOG_N_LANES`, where the native device
// kernels don't apply; mirrors simd's own CPU fallback for that case). ===

fn accumulate_numerators_simd_delegated(
    columns: &[&CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>],
    sample_batches: &[ColumnSampleBatch],
    accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<CudaBackend>>,
    log_blowup_factor: u32,
) {
    let simd_cols: Vec<_> = columns.iter().map(|c| eval_cuda_to_simd(c)).collect();
    let simd_refs: Vec<_> = simd_cols.iter().collect();
    let mut simd_acc: Vec<AccumulatedNumerators<SimdBackend>> = vec![];
    SimdBackend::accumulate_numerators(
        &simd_refs,
        sample_batches,
        &mut simd_acc,
        log_blowup_factor,
    );
    for a in simd_acc {
        accumulated_numerators_vec.push(AccumulatedNumerators {
            sample_point: a.sample_point,
            partial_numerators_acc: securecol_simd_to_cuda(a.partial_numerators_acc),
            first_linear_term_acc: a.first_linear_term_acc,
        });
    }
}

fn compute_quotients_and_combine_simd_delegated(
    accs: Vec<AccumulatedNumerators<CudaBackend>>,
    lifting_log_size: u32,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<CudaBackend>,
) -> SecureEvaluation<CudaBackend, BitReversedOrder> {
    let simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = accs
        .iter()
        .map(|a| AccumulatedNumerators {
            sample_point: a.sample_point,
            partial_numerators_acc: securecol_cuda_to_simd(&a.partial_numerators_acc),
            first_linear_term_acc: a.first_linear_term_acc,
        })
        .collect();
    let simd_tw = twiddles_cuda_to_simd(twiddles);
    let res = SimdBackend::compute_quotients_and_combine(
        simd_accs,
        lifting_log_size,
        log_blowup_factor,
        &simd_tw,
    );
    SecureEvaluation::new(res.domain, securecol_simd_to_cuda(res.values))
}

impl QuotientOps for CudaBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    ) {
        let domain = columns[0].domain;
        // simd quotients.rs L42: subdomain = first `size >> log_blowup_factor` rows.
        let (subdomain, _) = domain.split(log_blowup_factor);

        // Mirror simd's small-subdomain CPU path (simd quotients.rs L45-65) by delegating to
        // SimdBackend (which itself falls back to CPU for sub-LANE subdomains).
        if subdomain.log_size() < LOG_N_LANES {
            return accumulate_numerators_simd_delegated(
                columns,
                sample_batches,
                accumulated_numerators_vec,
                log_blowup_factor,
            );
        }

        let subdomain_size = subdomain.size();
        let quotient_constants = quotient_constants(sample_batches);

        // Resident whole-subdomain launch: every column is live on device.
        let host_col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.values.device_ptr).collect();
        let device_col_ptrs = unsafe {
            interface::bindings::copy_device_pointer_vec_from_host_to_device(
                host_col_ptrs.as_ptr(),
                host_col_ptrs.len(),
            )
        };
        for (batch, coeffs) in sample_batches.iter().zip(quotient_constants.line_coeffs) {
            let line_coeffs_b: Vec<CudaSecureField> =
                coeffs.iter().map(|(_, b, _)| CudaSecureField::from(*b)).collect();
            let line_coeffs_c: Vec<CudaSecureField> =
                coeffs.iter().map(|(_, _, c)| CudaSecureField::from(*c)).collect();
            let column_indices: Vec<u32> =
                batch.cols_vals_randpows.iter().map(|n| n.column_index as u32).collect();
            let result = unsafe { CudaSecureColumn::new_with_size(subdomain_size) };
            unsafe {
                interface::bindings::accumulate_numerators_batch(
                    subdomain_size as u32,
                    device_col_ptrs,
                    line_coeffs_b.as_ptr(),
                    line_coeffs_c.as_ptr(),
                    column_indices.as_ptr(),
                    coeffs.len() as u32,
                    result.columns[0].device_ptr,
                    result.columns[1].device_ptr,
                    result.columns[2].device_ptr,
                    result.columns[3].device_ptr,
                );
            }
            let first_linear_term_acc = coeffs.iter().map(|(a, ..)| *a).sum();
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc: result,
                first_linear_term_acc,
            });
        }
        unsafe {
            interface::bindings::cuda_free_memory(device_col_ptrs as *const std::ffi::c_void);
        }
    }

    fn compute_quotients_and_combine(
        accumulations: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let eval_domain = CanonicCoset::new(lifting_log_size).circle_domain();
        // simd quotients.rs L91-92.
        let (eval_subdomain, _) = eval_domain.split(log_blowup_factor);

        // Mirror simd's small-subdomain CPU path (simd quotients.rs L94-115).
        if eval_subdomain.log_size() < LOG_N_LANES {
            return compute_quotients_and_combine_simd_delegated(
                accumulations,
                lifting_log_size,
                log_blowup_factor,
                twiddles,
            );
        }

        let subdomain_size = eval_subdomain.size();
        let subdomain_log_size = eval_subdomain.log_size();
        let num_acc = accumulations.len();

        // === Step 1: compute the quotient on the SUBDOMAIN (simd quotients.rs L116-179). ===
        // The kernel computes, per subdomain row, for each accumulation:
        //   den_inv  = inv((prx - p.x)*piy - (pry - p.y)*pix)        (simd L255-283 / cu L259-267)
        //   lifted   = partial_acc[lift(row, log_ratio)]             (simd L157-166 / cu L270-278)
        //   full_num = lifted - first_linear_term_acc * p.y          (simd L168-169 / cu L280-281)
        //   quotient += full_num * den_inv                           (simd L170     / cu L283-284)
        // `acc_log_sizes[a]` is the log size of accumulation a's partial numerators; the kernel
        // derives `log_ratio = subdomain_log_size - acc_log_sizes[a]` (== simd L126-129).
        let acc_partial_ptrs: Vec<*const u32> = accumulations
            .iter()
            .flat_map(|acc| {
                // Order a.a, a.b, b.a, b.b — matches accumulate kernel result_0..3 and simd coords.
                acc.partial_numerators_acc
                    .columns
                    .iter()
                    .map(|col| col.device_ptr)
                    .collect::<Vec<_>>()
            })
            .collect();
        let acc_log_sizes: Vec<i32> = accumulations
            .iter()
            .map(|acc| acc.partial_numerators_acc.columns[0].len().ilog2() as i32)
            .collect();
        let first_linear_term_accs: Vec<CudaSecureField> = accumulations
            .iter()
            .map(|acc| CudaSecureField::from(acc.first_linear_term_acc))
            .collect();
        let sample_points: Vec<CirclePointSecureField> = accumulations
            .iter()
            .map(|acc| CirclePointSecureField::from(acc.sample_point))
            .collect();

        let subdomain_quotient = unsafe { CudaSecureColumn::new_with_size(subdomain_size) };
        unsafe {
            interface::bindings::compute_quotients_and_combine(
                subdomain_size as u32,
                subdomain_log_size,
                eval_subdomain.half_coset.initial_index.0 as u32,
                eval_subdomain.half_coset.step_size.0 as u32,
                num_acc as u32,
                acc_partial_ptrs.as_ptr(),
                acc_log_sizes.as_ptr(),
                first_linear_term_accs.as_ptr(),
                sample_points.as_ptr(),
                subdomain_quotient.columns[0].device_ptr,
                subdomain_quotient.columns[1].device_ptr,
                subdomain_quotient.columns[2].device_ptr,
                subdomain_quotient.columns[3].device_ptr,
            );
        }

        // === Step 2: lift subdomain quotient to the full domain (simd quotients.rs L180-197). ===
        // For each of the 4 secure coords: interpolate the subdomain eval -> coeffs, then evaluate
        // on the full eval_domain. CudaBackend::interpolate / ::evaluate are the device NTTs. The
        // interpolation uses the SUBDOMAIN twiddles (deterministic from eval_subdomain.half_coset,
        // identical values to simd's `extract_subdomain_twiddles`); evaluation uses the full
        // `twiddles` (root_coset == eval_domain.half_coset), exactly as simd does.
        let subdomain_twiddles = CudaBackend::precompute_twiddles(eval_subdomain.half_coset);
        let lifted_columns: [BaseFieldVec; 4] = subdomain_quotient.columns.map(|coord| {
            let coeffs: CircleCoefficients<CudaBackend> = CudaBackend::interpolate(
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                    eval_subdomain,
                    coord,
                ),
                &subdomain_twiddles,
            );
            CudaBackend::evaluate(&coeffs, eval_domain, twiddles).values
        });

        SecureEvaluation::new(eval_domain, SecureColumnByCoords { columns: lifted_columns })
    }
}
