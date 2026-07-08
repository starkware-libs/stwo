use std::ffi::c_void;

use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::ColumnVec;
use crate::prover::air::component_prover::Poly;
use crate::prover::backend::cuda::CudaBackend;
use crate::prover::backend::Column;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::stwo_cuda::bindings::CudaSecureField;

/// Default number of columns processed per `ntt_n2b_columns` launch in the batched
/// `evaluate_polynomials` extend+NTT. Chosen to cap the transient device-memory peak during the
/// tree-commit extend (so large shards fit in 40GB) while keeping batched launch efficiency.
/// Overridable via the `GATE_AIR_NTT_SUBBATCH` env var.
const DEFAULT_NTT_SUBBATCH: usize = 48;

/// Sub-batch width for the batched extend+NTT in `evaluate_polynomials`. Reads
/// `GATE_AIR_NTT_SUBBATCH` (a positive integer); falls back to `DEFAULT_NTT_SUBBATCH` when unset,
/// empty, unparseable, or zero. Purely an allocation-granularity knob — it does not affect outputs.
fn ntt_subbatch_size() -> usize {
    std::env::var("GATE_AIR_NTT_SUBBATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_NTT_SUBBATCH)
}

pub trait CudaVariable<T> {
    /// # Safety
    /// do not dereference if the memory is located on the device
    unsafe fn as_ref(&self) -> &T;

    fn as_ptr(&self) -> *const T {
        unsafe { self.as_ref() }
    }

    fn as_c_void_ptr(&self) -> *const c_void {
        self.as_ptr() as *const c_void
    }
}

pub trait CudaVariableMut<T>: CudaVariable<T> {
    /// # Safety
    /// do not dereference if the memory is located on the device
    unsafe fn as_mut(&mut self) -> &mut T;

    fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.as_mut() }
    }

    fn as_mut_c_void_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr() as *mut c_void
    }
}

impl<T> CudaVariable<T> for T {
    unsafe fn as_ref(&self) -> &T {
        self
    }
}

impl<T> CudaVariableMut<T> for T {
    unsafe fn as_mut(&mut self) -> &mut T {
        self
    }
}

use crate::prover::backend::cpu::CpuCirclePoly;
use crate::prover::backend::cuda::fused_commit;
use crate::stwo_cuda as interface;
use crate::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::stwo_cuda::SecureFieldVec;
pub(crate) type CudaCircleEvaluation<F, EvalOrder> = CircleEvaluation<CudaBackend, F, EvalOrder>;
// fn interpolate_native(
//     eval: CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>,
//     twiddle_tree: &TwiddleTree<CudaBackend>,
// ) -> CircleCoefficients<CudaBackend> {
//     let values = eval.values;
//     assert!(eval.domain.half_coset.is_doubling_of(twiddle_tree.root_coset));
//     unsafe {
//         interface::bindings::interpolate(
//             eval.domain.half_coset.size() as u32,
//             values.device_ptr,
//             twiddle_tree.itwiddles.device_ptr,
//             twiddle_tree.itwiddles.len() as u32,
//             values.len() as u32,
//         );
//     }

//     CircleCoefficients::new(values)
// }

// fn interpolate_columns_native(
//     columns: impl IntoIterator<Item = CircleEvaluation<CudaBackend, BaseField,
// BitReversedOrder>>,     twiddles: &TwiddleTree<CudaBackend>,
// ) -> Vec<CircleCoefficients<CudaBackend>> {
//     let columns = columns.into_iter().collect_vec();
//     let values = columns
//         .iter()
//         .map(|column| column.values.device_ptr)
//         .collect_vec();
//     let number_of_rows = columns[0].len();
//     unsafe {
//         interface::bindings::interpolate_columns(
//             columns[0].domain.half_coset.size() as u32,
//             values.as_ptr(),
//             twiddles.itwiddles.device_ptr,
//             twiddles.itwiddles.len() as u32,
//             columns.len() as u32,
//             number_of_rows as u32,
//         );
//     }

//     columns
//         .into_iter()
//         .map(|column| CircleCoefficients::new(column.values))
//         .collect_vec()
// }

use std::mem::transmute;

use crate::prover::backend::cpu::CpuCircleEvaluation;
use crate::prover::backend::CpuBackend;

/// Evaluate multiple same-size polynomials at the same point using a single batched CUDA call.
/// All polynomials must have the same coeffs_size.
/// Only the array of device pointers is copied to the GPU — the polynomial data stays in place.
/// Returns a Vec<SecureField> with one result per polynomial.
pub fn cuda_batch_eval_at_point(
    polys: &[&CircleCoefficients<CudaBackend>],
    point: CirclePoint<SecureField>,
) -> Vec<SecureField> {
    let num_polys = polys.len();
    if num_polys == 0 {
        return Vec::new();
    }

    // Collect device pointers from each polynomial (these are already GPU addresses)
    let host_ptrs: Vec<*const u32> = polys.iter().map(|p| p.coeffs.device_ptr).collect();
    let coeffs_size = polys[0].coeffs.len();

    // Upload only the pointer array to device (num_polys * 8 bytes, not the data)
    let device_ptrs = unsafe {
        interface::bindings::copy_device_pointer_vec_from_host_to_device(
            host_ptrs.as_ptr() as *const *const u32,
            num_polys,
        )
    };

    // Allocate host result buffer
    let mut results: Vec<CudaSecureField> =
        (0..num_polys).map(|_| CudaSecureField::zero()).collect();

    unsafe {
        interface::bindings::batch_eval_at_points(
            device_ptrs,
            coeffs_size as i32,
            num_polys as i32,
            CudaSecureField::from(point.x),
            CudaSecureField::from(point.y),
            results.as_mut_ptr(),
        );
    }

    // Free the device pointer array (not the polynomial data)
    unsafe {
        interface::bindings::cuda_free_memory(device_ptrs as *const std::ffi::c_void);
    }

    results.into_iter().map(SecureField::from).collect()
}

impl PolyOps for CudaBackend {
    type Twiddles = BaseFieldVec;

    // fn new_canonical_ordered(
    //     coset: CanonicCoset,
    //     values: Col<Self, BaseField>,
    // ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
    //     let size = values.len();
    //     let device_ptr = unsafe {
    //         interface::bindings::sort_values_and_permute_with_bit_reverse_order(values.
    // device_ptr, size)     };
    //     let result = BaseFieldVec::new(device_ptr, size);
    //     CircleEvaluation::new(coset.circle_domain(), result)
    // }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        assert!(eval
            .domain
            .half_coset
            .is_doubling_of(twiddle_tree.root_coset));

        if eval.domain.log_size() <= 3 {
            let cpu_eval = CpuCircleEvaluation::new(eval.domain, eval.values.to_cpu());

            let cpu_circle_poly =
                CpuBackend::interpolate(cpu_eval, unsafe { transmute(twiddle_tree) });

            let cuda_coeffs = BaseFieldVec::from_vec(cpu_circle_poly.coeffs.to_vec());

            return CircleCoefficients::<CudaBackend>::new(cuda_coeffs);
        }

        let values = eval.values;
        unsafe {
            interface::bindings::ntt_b2n_column(
                values.device_ptr.as_ptr() as *mut *mut u32,
                values.len().ilog2(),
                1 as u32,
                twiddle_tree.itwiddles.device_ptr,
                twiddle_tree.itwiddles.len() as u32,
                eval.domain.half_coset.size() as u32,
            );
        }

        CircleCoefficients::new(values)
    }
    fn interpolate_columns(
        columns: Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        // Collect columns with their original indices, then group by log_size for batch NTT.
        let mut indexed: Vec<(usize, u32, BaseFieldVec, CircleDomain)> = columns
            .into_iter()
            .enumerate()
            .map(|(i, eval)| {
                let log_size = eval.domain.log_size();
                (i, log_size, eval.values, eval.domain)
            })
            .collect();

        if indexed.is_empty() {
            return Vec::new();
        }

        // Sort by log_size to group same-size columns together.
        indexed.sort_by_key(|(_, ls, ..)| *ls);

        let mut results: Vec<(usize, CircleCoefficients<Self>)> = Vec::with_capacity(indexed.len());
        let mut group_start = 0;

        while group_start < indexed.len() {
            let log_size = indexed[group_start].1;

            // Find end of this size group.
            let mut group_end = group_start + 1;
            while group_end < indexed.len() && indexed[group_end].1 == log_size {
                group_end += 1;
            }

            let group = &mut indexed[group_start..group_end];

            if log_size <= 3 {
                // Small columns: CPU interpolation.
                for item in group.iter_mut() {
                    let values = std::mem::replace(&mut item.2, BaseFieldVec::new_uninitialized(0));
                    let cpu_eval = CpuCircleEvaluation::new(item.3, values.to_cpu());
                    let cpu_poly =
                        CpuBackend::interpolate(cpu_eval, unsafe { transmute(twiddles) });
                    let cuda_coeffs = BaseFieldVec::from_vec(cpu_poly.coeffs.to_vec());
                    results.push((item.0, CircleCoefficients::<Self>::new(cuda_coeffs)));
                }
            } else {
                // Batch NTT: single kernel call for all columns in this size group.
                let num_poly = group.len();
                let eval_domain_size = group[0].3.half_coset.size() as u32;

                let mut ptrs: Vec<*mut u32> = group
                    .iter()
                    .map(|item| item.2.device_ptr as *mut u32)
                    .collect();

                unsafe {
                    interface::bindings::ntt_b2n_column(
                        ptrs.as_mut_ptr() as *mut *mut u32,
                        log_size,
                        num_poly as u32,
                        twiddles.itwiddles.device_ptr,
                        twiddles.itwiddles.len() as u32,
                        eval_domain_size,
                    );
                }

                for item in group.iter_mut() {
                    let values = std::mem::replace(&mut item.2, BaseFieldVec::new_uninitialized(0));
                    results.push((item.0, CircleCoefficients::new(values)));
                }
            }

            group_start = group_end;
        }

        // Restore original column order.
        results.sort_by_key(|(idx, _)| *idx);
        results.into_iter().map(|(_, poly)| poly).collect()
    }

    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        unsafe {
            interface::bindings::eval_at_point(
                poly.coeffs.device_ptr,
                poly.coeffs.len() as u32,
                CudaSecureField::from(point.x),
                CudaSecureField::from(point.y),
            )
            .into()
        }
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        use crate::core::poly::utils::get_folding_alphas;
        use crate::prover::fri::FriOps;
        use crate::prover::poly::circle::SecureEvaluation;
        use crate::prover::secure_column::SecureColumnByCoords;

        let log_size = evals.domain.log_size();
        let mut folding_alphas = get_folding_alphas(point, log_size as usize);

        // Convert BaseField evals to SecureField evals on GPU
        let secure_col = SecureColumnByCoords::from_base_field_col(&evals.values);
        let secure_eval = SecureEvaluation::new(evals.domain, secure_col);

        // 74951f79: fold_circle_into_line now RETURNS a fresh LineEvaluation (no dst arg);
        // fold_line takes a SLICE of alphas. Here each step folds with a single alpha.
        let mut layer_evaluation = CudaBackend::fold_circle_into_line(
            &secure_eval,
            folding_alphas.pop().unwrap(),
            twiddles,
        );

        while layer_evaluation.len() > 1 {
            layer_evaluation = CudaBackend::fold_line(
                &layer_evaluation,
                &[folding_alphas.pop().unwrap()],
                twiddles,
            );
        }

        layer_evaluation.values.at(0) / SecureField::from(2_u32.pow(log_size))
    }

    fn barycentric_weights(coset: CanonicCoset, p: CirclePoint<SecureField>) -> SecureFieldVec {
        use crate::core::constraints::coset_vanishing;

        let domain = coset.circle_domain();
        let log_size = domain.log_size();
        let domain_size = domain.size();

        // Compute vn_p on CPU (single scalar, O(log_size) ops)
        let vn_p: SecureField = coset_vanishing(
            CanonicCoset::new(log_size).coset,
            p.into_ef::<SecureField>(),
        );

        // Compute exp_val = 4^(log_size-1) mod P on CPU
        let mut exp_val = BaseField::from(1u32);
        for _ in 1..log_size {
            exp_val = exp_val * BaseField::from(4u32);
        }

        // Allocate result on GPU
        let result = SecureFieldVec::new_uninitialized(domain_size);

        unsafe {
            interface::bindings::barycentric_weights_cuda(
                domain.half_coset.initial_index.0 as u32,
                domain.half_coset.step_size.0 as u32,
                domain_size as i32,
                log_size as i32,
                CudaSecureField::from(vn_p),
                CudaSecureField::from(p.x),
                CudaSecureField::from(p.y),
                exp_val.0,
                result.device_ptr,
            );
        }

        result
    }

    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &SecureFieldVec,
    ) -> SecureField {
        // STEP 2 (GATE_AIR_STREAM_COMMIT): the streamed commit freed this column's device buffer
        // and kept its bytes in the streaming-commit-layer stash (fused_commit). This dense OODS
        // read runs ON GPU, so H2D a TEMPORARY resident copy from the stash, evaluate on device,
        // and drop it immediately (freed on scope exit). Residency during OODS is thus O(one
        // column) rather than the whole 188-column eval set. The temporary is bit-identical to the
        // committed column (same u32 payload), so the OODS value is identical to the resident path.
        // `&self` is preserved: `rehydrate_owned` copies from the stash without mutating the
        // column.
        let mut result = CudaSecureField::zero();
        if fused_commit::is_staged(&evals.values) {
            // T2: split the staged rehydrate H2D from the barycentric kernel compute.
            let t2 = fused_commit::t1_timers_on();
            let h2d_start = t2.then(std::time::Instant::now);
            let tmp = fused_commit::rehydrate_owned(&evals.values);
            if let Some(s) = h2d_start {
                crate::prover::prove_ex_sync();
                fused_commit::t2_add(0, s.elapsed().as_nanos());
            }
            let k_start = t2.then(std::time::Instant::now);
            unsafe {
                interface::bindings::barycentric_eval_at_point_cuda(
                    tmp.device_ptr,
                    weights.device_ptr,
                    evals.domain.size() as i32,
                    &mut result,
                );
            }
            if let Some(s) = k_start {
                crate::prover::prove_ex_sync();
                fused_commit::t2_add(1, s.elapsed().as_nanos());
            }
            return SecureField::from(result);
        }
        unsafe {
            interface::bindings::barycentric_eval_at_point_cuda(
                evals.values.device_ptr,
                weights.device_ptr,
                evals.domain.size() as i32,
                &mut result,
            );
        }
        SecureField::from(result)
    }

    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        let new_size = 1 << log_size;
        assert!(
            new_size >= poly.coeffs.len(),
            "New size must be larger than the old size: new_size={new_size} (log_size={log_size}), poly.coeffs.len()={}",
            poly.coeffs.len()
        );

        let mut new_coeffs = BaseFieldVec::new_zeroes(new_size);
        new_coeffs.copy_from(&poly.coeffs);
        CircleCoefficients::new(new_coeffs)
    }

    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let domain_log_size = domain.log_size();

        assert!(domain.half_coset.is_doubling_of(twiddle_tree.root_coset));
        if domain_log_size <= 3 {
            let cpu_poly = CpuCirclePoly::new(poly.coeffs.to_cpu());

            let cpu_circle_eval =
                CpuBackend::evaluate(&cpu_poly, domain, unsafe { transmute(twiddle_tree) });

            let cuda_eval_values = BaseFieldVec::from_vec(cpu_circle_eval.values.to_vec());
            return CudaCircleEvaluation::new(cpu_circle_eval.domain, cuda_eval_values);
        }
        let values = poly.extend(domain_log_size).coeffs;
        unsafe {
            interface::bindings::ntt_n2b_columns(
                values.device_ptr.as_ptr() as *mut *mut u32,
                values.len().ilog2(),
                1,
                twiddle_tree.twiddles.device_ptr,
                twiddle_tree.twiddles.len() as u32,
                domain.half_coset.size() as u32,
            );
        }

        CircleEvaluation::new(domain, values)
    }

    fn evaluate_into(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
        _buffer: crate::prover::backend::Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // 74951f79 adds evaluate_into (write into a reused mempool buffer). The buffer is purely
        // an allocation-reuse optimization; evaluating normally yields identical values.
        Self::evaluate(poly, domain, twiddles)
    }

    fn evaluate_polynomials(
        polynomials: ColumnVec<CircleCoefficients<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
        store_polynomials_coefficients: bool,
        _pool: &crate::prover::mempool::BaseColumnPool<Self>,
    ) -> Vec<Poly<Self>> {
        // Build indexed list with extended log_size for grouping.
        let mut indexed: Vec<(usize, u32, CircleDomain, CircleCoefficients<Self>)> = polynomials
            .into_iter()
            .enumerate()
            .map(|(i, poly)| {
                let domain = CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain();
                (i, domain.log_size(), domain, poly)
            })
            .collect();

        if indexed.is_empty() {
            return Vec::new();
        }

        // Staging scope for the STREAMED tree1 commit. True ONLY when GATE_AIR_STREAM_COMMIT is set
        // AND this call contains at least one BORROWED (`owns_memory == false`) input column — the
        // structural signature of tree1's un-interpolated `d_cols` main views under
        // GATE_AIR_FUSED_INTERP. tree0/tree2 and every other caller have no borrowed columns, so
        // `stream_tree1` is false and their columns stay fully resident (unchanged). When true, only
        // the LARGE main columns (the interp branch below) are host-staged (dehydrated) — the ~47 GB
        // eval set that must leave the device; the small (multiplicity/witness/program) tree1 columns
        // stay RESIDENT (FULL (A): small cols are cheap to keep, and the row-tilers read a resident
        // column directly per-column, so no uniform all-staged assumption is needed). Only the byte
        // SOURCE of the large columns moves (device -> host stash); committed values are unchanged.
        let stream_tree1 = fused_commit::stream_commit_enabled()
            && indexed
                .iter()
                .any(|(_, _, _, poly)| !poly.coeffs.owns_memory);

        // Option (a) — interaction (tree2) staging scope. True ONLY when
        // GATE_AIR_STREAM_INTERACTION is set AND the interaction-commit arm flag is up (set by
        // the gate_air leaf prover immediately before the tree2 `commit()`; see
        // `fused_commit::arm_interaction_commit`). The interaction eval columns are OWNED (never
        // borrowed), so they never match the `stream_tree1` borrowed-column signature — hence the
        // dedicated trigger. The arm flag is what makes this byte-identity-safe: it fires ONLY for
        // the tree2 interaction commit, never for tree0's preprocessed `extend_evals` fallback
        // (also all-owned, same full size) or tree1's small OWNED columns (which share tree1's
        // call and must stay resident). When true, the OWNED interaction eval columns are
        // host-staged (dehydrated) in the chunked branch below, freeing their device buffers so
        // the composition kernel row-tiles tree2 (`host_trace2` non-null -> `tree2_tiled`) instead
        // of holding it whole-resident. Only the byte SOURCE moves (device -> host stash);
        // committed values are unchanged. `stream_tree1` and `stream_tree2` are mutually exclusive
        // per call (tree1's call has a borrowed column, so its arm flag is down; tree2's call is
        // all-owned).
        let stream_tree2 = fused_commit::stream_interaction_enabled()
            && fused_commit::interaction_commit_armed()
            && indexed.iter().all(|(_, _, _, poly)| poly.coeffs.owns_memory);

        // DIAG (GATE_AIR_NTT_DEBUG): observe, per commit, whether the tree2 interaction streaming
        // actually engages — prints the stream_tree2 decision + its three sub-conditions. The tree2
        // interaction commit is the one with ~28 owned same-size columns; if stream_tree2 is false
        // there, the dehydrate never runs and the whole eval set stays resident (chunk size is then
        // irrelevant — explaining why arm-wiring and NTT_SUBBATCH had no effect).
        let ntt_dbg = std::env::var("GATE_AIR_NTT_DEBUG").is_ok();
        if ntt_dbg {
            eprintln!(
                "[ntt_dbg] evaluate_polynomials: n_cols={} stream_tree1={} stream_tree2={} \
                 si_enabled={} armed={} all_owned={}",
                indexed.len(),
                stream_tree1,
                stream_tree2,
                fused_commit::stream_interaction_enabled(),
                fused_commit::interaction_commit_armed(),
                indexed.iter().all(|(_, _, _, poly)| poly.coeffs.owns_memory),
            );
            // stderr is fully buffered under `>> logfile` redirect and its tail is LOST on abort —
            // flush so this line survives an OOM crash in the same commit (the whole point of it).
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }

        // Unlike `stream_tree1`, tree2 does NOT `clear_stash()`: the tree1 staged columns are
        // still live in the stash (they must survive commit-through-decommit), and tree2's columns
        // are keyed by fresh unique sentinels, so both trees' entries coexist until the next
        // proof's tree1 commit clears them.
        if stream_tree1 {
            // Evict any stale stash entries from a PRIOR streamed commit before staging THIS tree's
            // columns, so a freed pointer key can never alias a fresh live column (which would
            // falsely read as staged — soundness). Called ONCE per tree1 commit here (not per
            // group): tree1 now stages multiple groups (large interp + small chunked) and a
            // per-group clear would wipe an earlier group's staged columns before `build_leaves`
            // rehydrates them. See `fused_commit::clear_stash`.
            fused_commit::clear_stash();
        }

        // Sort by extended log_size to batch same-size NTTs together.
        indexed.sort_by_key(|(_, ls, ..)| *ls);

        let mut results: Vec<(usize, Poly<Self>)> = Vec::with_capacity(indexed.len());
        let mut group_start = 0;

        while group_start < indexed.len() {
            let log_size = indexed[group_start].1;

            // Find end of this size group.
            let mut group_end = group_start + 1;
            while group_end < indexed.len() && indexed[group_end].1 == log_size {
                group_end += 1;
            }

            if log_size <= 3 {
                // Small polynomials: CPU fallback.
                // Drain the group from indexed to take ownership.
                let group: Vec<_> = indexed.drain(group_start..group_end).collect();
                // Adjust group_end since we drained.
                group_end = group_start;
                for (orig_idx, _, domain, poly) in group {
                    let cpu_poly = CpuCirclePoly::new(poly.coeffs.to_cpu());
                    let cpu_eval =
                        CpuBackend::evaluate(&cpu_poly, domain, unsafe { transmute(twiddles) });
                    let cuda_values = BaseFieldVec::from_vec(cpu_eval.values.to_vec());
                    let eval = CircleEvaluation::new(domain, cuda_values);
                    results.push((
                        orig_idx,
                        Poly::new(store_polynomials_coefficients.then_some(poly), eval),
                    ));
                }
            } else {
                // Batch extend + batched NTT. To cap the TRANSIENT device-memory peak during the
                // extend (each fresh eval buffer is one extended column, e.g. ~128MB at 2^25), the
                // group's columns are processed in sub-batches of `ntt_subbatch_size()` rather than
                // one full-width batch: extend+NTT a chunk, keep its evals resident, then move to
                // the next chunk. The cuda mempool caches each chunk's transient NTT scratch for
                // reuse by the next chunk, so the instantaneous peak is bounded by one chunk's
                // worth of fresh allocations instead of all `num_poly` at once. The final resident
                // set (all columns' evals, read by the subsequent lifted-Merkle commit) is
                // unchanged, and the NTT output is byte-identical: each column's NTT is
                // independent, so `ntt_n2b_columns` over a sub-slice yields exactly
                // the same per-column result as over the full batch (batching is
                // purely launch grouping).
                let num_poly = group_end - group_start;
                let eval_domain_size = indexed[group_start].2.half_coset.size() as u32;
                let chunk = ntt_subbatch_size().min(num_poly).max(1);

                let mut values_list: Vec<BaseFieldVec> = Vec::with_capacity(num_poly);

                // STEP 1 fused path (GATE_AIR_FUSED_COMMIT, default OFF). When ON: drive
                // extend+NTT ONE COLUMN AT A TIME, feeding each freshly-extended eval column into
                // the P-S0 streamed Blake2s absorb, then finalize the leaf layer and stash it for
                // the immediately-following `build_leaves` (blake2s.rs) to consume. Eval columns
                // are KEPT RESIDENT (appended to `values_list` exactly as the
                // default path) — no free, no host-stage, no ceiling move. The
                // per-column NTT is byte-identical to the sub-batched NTT (columns
                // are an independent grid dimension; asserted at rfft.cu),
                // and absorbing one column at a time in group order is bit-identical to the fused
                // all-at-once leaf hash by absorb-associativity (see P-S0, blake2s.rs:130-139). The
                // absorb order here is the group's `indexed` order which — for the same-size main
                // tree (a single group) — equals both the original column order AND `commit`'s
                // `sorted_by_key(|c| c.len())` order (stable sort on equal keys). The stash is
                // keyed by the exact eval-column device pointers so `build_leaves`
                // reuses it ONLY for the matching columns and otherwise safely
                // rebuilds.
                // Fix (b) interpolate-in-commit (GATE_AIR_FUSED_INTERP): the gate_air driver feeds
                // tree1's main columns as BORROWED views into `d_cols` holding UN-interpolated
                // base-domain evals (never a D2D copy). A group is the un-interpolated main group iff
                // it contains a borrowed (`owns_memory == false`) input column — the structural
                // signature that distinguishes it from the already-interpolated tree0/tree2/small
                // groups (all owned). When ON, this group runs the fused loop and the loop
                // interpolates each column per-column FIRST (see below).
                let interp_in_commit = fused_commit::interpolate_in_commit_enabled()
                    && indexed[group_start..group_end]
                        .iter()
                        .any(|(_, _, _, poly)| !poly.coeffs.owns_memory);
                let fused = (fused_commit::fused_commit_enabled()
                    && log_size == eval_domain_size.ilog2())
                    || interp_in_commit;
                // STEP 2: under `stream_tree1` (GATE_AIR_STREAM_COMMIT + this tree1 call), host-stage
                // (D2H + deferred free) each LARGE main eval column right after it is produced (the
                // interp branch below). This drops the commit peak below 40 GB at 2^24/2^25. Columns
                // are still pushed to `values_list` (so the tree shape and downstream indexing are
                // unchanged) but their device buffers are freed; post-commit readers
                // rehydrate/host-recover on demand. The SMALL (multiplicity/witness/program) tree1
                // columns stay RESIDENT (FULL (A), see the chunked `else` branch ~line 760): the
                // per-column readers handle a mixed staged+resident tree1, so no all-staged
                // assumption is required. EVERY large column is staged under a UNIQUE stash key: the
                // per-column free is DEFERRED (alloc-next-before-free-prev, see the interp loop) so a
                // just-freed address is never reused for the next column, which previously made half
                // the large columns collide on a shared device_ptr and stay resident (the 94/188
                // corruption). When OFF, columns stay fully resident (step-1 behavior).
                if fused {
                    // `fused` is only reachable via `interpolate_in_commit` (the non-interp fused
                    // condition `log_size == eval_domain_size.ilog2()` is unsatisfiable: a
                    // CircleDomain's `log_size() == half_coset.log_size + 1`, so it is `n == n-1`).
                    // So this block is purely the streamed per-column PRODUCER: it interpolates,
                    // extends, n2b-transforms, dehydrates (host-stages) and pushes each eval column.
                    // The leaf hashing is owned entirely by the subsequent `build_leaves`
                    // (blake2s.rs), which absorbs the (dehydrated) columns by rehydrating from the
                    // host stash — no leaf states are allocated or absorbed here.

                    // Fix (b): ONE reused base-size temp buffer to interpolate each borrowed-view
                    // column into (never in-place on the borrowed `d_cols` view — that would clobber
                    // d_cols, which K4/interaction reuses). Base log = coeffs.len().ilog2(); the base
                    // circle domain's `half_coset.size()` is the b2n `eval_domain_size`, mirroring
                    // `interpolate_columns` (num_poly = 1). Allocated only when interpolate-in-commit
                    // is active for this group.
                    let interp_base_log = indexed[group_start].3.log_size();
                    let mut interp_temp = if interp_in_commit {
                        Some(BaseFieldVec::new_zeroes(1usize << interp_base_log))
                    } else {
                        None
                    };
                    let interp_base_domain =
                        CanonicCoset::new(interp_base_log).circle_domain();
                    let interp_eval_domain_size =
                        interp_base_domain.half_coset.size() as u32;

                    for (_, _, _, poly) in &indexed[group_start..group_end] {
                        let mut col = if let Some(temp) = interp_temp.as_mut() {
                            // Per-column interpolate == the batched b2n in `interpolate_columns`
                            // (columns are an independent NTT grid dimension). The coeffs buffer
                            // handed to the in-place b2n MUST NOT be a borrowed d_cols view.
                            if poly.coeffs.owns_memory {
                                // Owned un-interpolated eval column: interpolate in place, leaving
                                // `poly.coeffs` holding valid interpolated coeffs (matches the
                                // flag-off `interpolate_columns` output), then extend a fresh eval
                                // buffer from it.
                                debug_assert!(
                                    poly.coeffs.owns_memory,
                                    "in-place b2n interpolate on a borrowed view is forbidden"
                                );
                                let mut coeffs_ptr = [poly.coeffs.device_ptr as *mut u32];
                                unsafe {
                                    interface::bindings::ntt_b2n_column(
                                        coeffs_ptr.as_mut_ptr() as *mut *mut u32,
                                        interp_base_log,
                                        1,
                                        twiddles.itwiddles.device_ptr,
                                        twiddles.itwiddles.len() as u32,
                                        interp_eval_domain_size,
                                    );
                                }
                                poly.extend(log_size).coeffs
                            } else {
                                // Borrowed view into d_cols: copy to temp, interpolate temp in
                                // place (leaves d_cols untouched for K4 reuse).
                                temp.copy_from(&poly.coeffs);
                                debug_assert!(
                                    temp.owns_memory,
                                    "in-place b2n interpolate on a borrowed view is forbidden"
                                );
                                let mut temp_ptr = [temp.device_ptr as *mut u32];
                                unsafe {
                                    interface::bindings::ntt_b2n_column(
                                        temp_ptr.as_mut_ptr() as *mut *mut u32,
                                        interp_base_log,
                                        1,
                                        twiddles.itwiddles.device_ptr,
                                        twiddles.itwiddles.len() as u32,
                                        interp_eval_domain_size,
                                    );
                                }
                                // Extend the interpolated coeffs (in `temp`) to the eval buffer.
                                let mut extended = BaseFieldVec::new_zeroes(1usize << log_size);
                                extended.copy_from(temp);
                                extended
                            }
                        } else {
                            poly.extend(log_size).coeffs
                        };
                        // T1 sub-timer: NTT GPU time. Under the timer flag we sync AFTER the launch
                        // to attribute real device time to NTT (vs. it folding into the following
                        // blocking D2H). The sync is diagnostic-only (timer runs), off the perf path.
                        let t1 = fused_commit::t1_timers_on();
                        let ntt_start = t1.then(std::time::Instant::now);
                        let mut col_ptr = [col.device_ptr as *mut u32];
                        unsafe {
                            interface::bindings::ntt_n2b_columns(
                                col_ptr.as_mut_ptr() as *mut *mut u32,
                                log_size,
                                1,
                                twiddles.twiddles.device_ptr,
                                twiddles.twiddles.len() as u32,
                                eval_domain_size,
                            );
                        }
                        if let Some(s) = ntt_start {
                            crate::prover::prove_ex_sync();
                            fused_commit::t1_add(0, s.elapsed().as_nanos());
                        }
                        // STEP 2: host-stage each eval column. `dehydrate_column` D2H-copies its
                        // bytes to the host stash, frees the device buffer (streaming: keeps the
                        // commit peak below 40 GB at 2^24/2^25), and re-keys the column by a UNIQUE
                        // monotonic sentinel (NOT the freed device address), so every large column
                        // stashes under its own entry and no two committed columns can share a key
                        // via mem-pool address reuse. `build_leaves` later rehydrates these.
                        if stream_tree1 {
                            fused_commit::dehydrate_column(&mut col);
                        }
                        values_list.push(col); // resident (step 1) or host-staged (step 2)
                    }
                } else {
                    let mut chunk_off = 0;
                    while chunk_off < num_poly {
                        let chunk_end = (chunk_off + chunk).min(num_poly);

                        let mut chunk_values: Vec<BaseFieldVec> = indexed
                            [group_start + chunk_off..group_start + chunk_end]
                            .iter()
                            .map(|(_, _, _, poly)| poly.extend(log_size).coeffs)
                            .collect();

                        let mut ptrs: Vec<*mut u32> = chunk_values
                            .iter()
                            .map(|v| v.device_ptr as *mut u32)
                            .collect();

                        unsafe {
                            interface::bindings::ntt_n2b_columns(
                                ptrs.as_mut_ptr() as *mut *mut u32,
                                log_size,
                                (chunk_end - chunk_off) as u32,
                                twiddles.twiddles.device_ptr,
                                twiddles.twiddles.len() as u32,
                                eval_domain_size,
                            );
                        }

                        // FULL (A): the small (multiplicity/witness/program) tree1 columns and the
                        // preprocessed tree0 columns stay RESIDENT (owned) — they
                        // are NOT dehydrated. Only the 188 LARGE main columns (the interp branch
                        // above) are host-staged. The downstream row-tilers (quotient/composition)
                        // read a resident column directly from its live device buffer (its
                        // `is_staged` check returns false), so a mixed resident+staged tree1 is
                        // handled per-column — no uniform all-staged assumption. This is the memory
                        // win (the large columns are the ~47 GB) without paying D2H/H2D for the small
                        // columns.
                        //
                        // Option (a) — interaction (tree2) staging: under `stream_tree2`
                        // (GATE_AIR_STREAM_INTERACTION + the armed interaction commit) host-stage
                        // (D2H + free) each OWNED interaction eval column right after its NTT, the
                        // same `dehydrate_column` mechanism tree1's large columns use. This frees
                        // the interaction device buffers so the composition kernel row-tiles tree2.
                        // Columns are still pushed to `values_list` (tree shape / indexing
                        // unchanged) with a sentinel device_ptr; `build_leaves` / OODS / quotient /
                        // decommit rehydrate from the stash on demand, and the composition kernel
                        // supplies them via `host_trace2`. Byte-identical (only the byte SOURCE
                        // moves). NOTE: unlike the tree1 large-column staging, the NTT here is the
                        // BATCHED multi-column launch above (interaction columns are all owned and
                        // same-size, so they never take the borrowed interp branch); the dehydrate
                        // runs per-column after that batched NTT, which is byte-identical to the
                        // per-column NTT+dehydrate (the NTT output is independent per column).
                        if stream_tree2 {
                            for mut col in chunk_values.drain(..) {
                                fused_commit::dehydrate_column(&mut col);
                                values_list.push(col);
                            }
                        } else {
                            values_list.append(&mut chunk_values);
                        }
                        chunk_off = chunk_end;
                        // DIAG (GATE_AIR_NTT_DEBUG): free/pool after each chunk's NTT (+dehydrate if
                        // stream_tree2) — shows whether the per-chunk peak is CAPPED (streaming frees
                        // between chunks) or CLIMBING (eval set accumulating). log_size identifies the
                        // group (interaction group = eval-domain size, ~28 same-size cols).
                        if ntt_dbg {
                            crate::stwo_cuda::cuda_mem_probe(&format!(
                                "ntt_chunk_log{log_size}_off{chunk_off}"
                            ));
                        }
                    }
                }

                // Drain the group to take ownership of polys.
                let group: Vec<_> = indexed.drain(group_start..group_end).collect();
                group_end = group_start;
                for (j, (orig_idx, _, domain, poly)) in group.into_iter().enumerate() {
                    let values =
                        std::mem::replace(&mut values_list[j], BaseFieldVec::new_uninitialized(0));
                    let eval = CircleEvaluation::new(domain, values);
                    results.push((
                        orig_idx,
                        Poly::new(store_polynomials_coefficients.then_some(poly), eval),
                    ));
                }
            }

            group_start = group_end;
        }

        // Restore original order.
        results.sort_by_key(|(idx, _)| *idx);
        results.into_iter().map(|(_, poly)| poly).collect()
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        unsafe {
            let twiddles = BaseFieldVec::new(
                interface::bindings::precompute_twiddles(
                    coset.initial.into(),
                    coset.step.into(),
                    coset.size(),
                ),
                coset.size(),
            );
            let itwiddles = BaseFieldVec::new_uninitialized(coset.size());
            interface::bindings::batch_inverse_base_field(
                twiddles.device_ptr,
                itwiddles.device_ptr,
                coset.size(),
            );
            TwiddleTree {
                root_coset: coset,
                twiddles,
                itwiddles,
            }
        }
    }

    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let (left, right) = poly.coeffs.split_at_mid();
        (
            CircleCoefficients::new(left),
            CircleCoefficients::new(right),
        )
    }
}
#[cfg(test)]
mod tests {
    // use itertools::Itertools;
    use test_log::test;

    use crate::core::poly::circle::{CanonicCoset, CircleDomain};
    use crate::core::{
        circle::{CirclePoint, CirclePointIndex, Coset},
        fields::m31::BaseField,
        // ColumnVec,
    };
    // use crate::prover::backend::cuda::poly::evaluate_native;
    use crate::prover::backend::cuda::CudaBackend;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
    use crate::prover::poly::twiddles::TwiddleTree;
    use crate::prover::poly::BitReversedOrder;
    use crate::stwo_cuda::base_field_vec::BaseFieldVec;

    // use ark_std::start_timer;
    // use ark_std::end_timer;

    // #[test]
    // fn test_new_canonical_ordered() {
    //     let log_size = 4;
    //     let coset = CanonicCoset::new(log_size);
    //     let size: usize = 1 << log_size;
    //     let column_data = (0..size as u32).map(BaseField::from).collect::<Vec<_>>();
    //     let cpu_values = column_data.clone();
    //     let expected_result = CpuBackend::new_canonical_ordered(coset, cpu_values.clone());

    //     let column = BaseFieldVec::from_vec(column_data);
    //     let result = CudaBackend::new_canonical_ordered(coset, column);

    //     assert_eq!(result.values.to_cpu(), expected_result.values);
    //     assert_eq!(
    //         result.domain.iter().collect::<Vec<_>>(),
    //         expected_result.domain.iter().collect::<Vec<_>>()
    //     );
    // }

    #[test]
    fn test_interpolate_evaluate_log24() {
        use crate::prover::poly::circle::CircleEvaluation as CpuCircleEvaluation;

        let log_size = 24u32;
        let size = 1usize << log_size;

        let cpu_values: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();
        let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let domain = coset.circle_domain();

        let cpu_evaluations =
            CpuCircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(domain, cpu_values);
        let gpu_evaluations =
            CircleEvaluation::<CudaBackend, _, BitReversedOrder>::new(domain, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        // Compare interpolation results
        assert_eq!(gpu_poly.coeffs.to_cpu(), cpu_poly.coeffs);

        // Test evaluation on a slightly larger domain
        let eval_coset = CanonicCoset::new(log_size + 1);
        let eval_domain = eval_coset.circle_domain();
        let cpu_twiddles2 = CpuBackend::precompute_twiddles(eval_coset.half_coset());
        let gpu_twiddles2 = CudaBackend::precompute_twiddles(eval_coset.half_coset());
        let cpu_eval = CpuBackend::evaluate(&cpu_poly, eval_domain, &cpu_twiddles2);
        let gpu_eval = CudaBackend::evaluate(&gpu_poly, eval_domain, &gpu_twiddles2);

        assert_eq!(gpu_eval.values.to_cpu(), cpu_eval.values);
    }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 5;

        let half_coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(half_coset);
        let twiddles = CudaBackend::precompute_twiddles(half_coset);

        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
        assert_eq!(
            twiddles.root_coset.iter().collect::<Vec<_>>(),
            expected_result.root_coset.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_extend() {
        let log_size = 20;
        let size = 1 << log_size;
        let new_log_size = log_size + 5;
        let cpu_coeffs = (0..size).map(BaseField::from).collect::<Vec<_>>();
        let cuda_coeffs = BaseFieldVec::from_vec(cpu_coeffs.clone());
        let cpu_poly = CircleCoefficients::<CpuBackend>::new(cpu_coeffs);
        let cuda_poly = CircleCoefficients::<CudaBackend>::new(cuda_coeffs);
        let result = CudaBackend::extend(&cuda_poly, new_log_size);
        let expected_result = CpuBackend::extend(&cpu_poly, new_log_size);
        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
        assert_eq!(result.log_size(), expected_result.log_size());
    }

    // #[test]
    // fn test_interpolate() {
    //     let log_size = 20;

    //     let size = 1 << log_size;

    //     let cpu_values = (1..(size + 1) as u32)
    //         .map(BaseField::from)
    //         .collect::<Vec<_>>();
    //     let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //     let coset = CanonicCoset::new(log_size);
    //     let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //     let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

    //     let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //     let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //     let timer = start_timer!(|| format!("cpu backend interpolate, log_n:{}", log_size));
    //     let expected_result = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //     end_timer!(timer);

    //     let timer = start_timer!(|| format!("gpu backend interpolate, log_n:{}", log_size));
    //     let result = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
    //     end_timer!(timer);

    //     assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
    // }

    // #[test]
    // fn test_interpolate_2() {
    //     let log_size = 5;

    //     let cpu_values = vec![
    //         BaseField::from(1),
    //         BaseField::from(443693538),
    //         BaseField::from(793699796),
    //         BaseField::from(1631104375),
    //         BaseField::from(460025527),
    //         BaseField::from(98131605),
    //         BaseField::from(1292025643),
    //         BaseField::from(1056169651),
    //         BaseField::from(29),
    //         BaseField::from(1645907698),
    //         BaseField::from(300234932),
    //         BaseField::from(2113642380),
    //         BaseField::from(2031046861),
    //         BaseField::from(541052612),
    //         BaseField::from(1857203558),
    //         BaseField::from(5),
    //         BaseField::from(2),
    //         BaseField::from(187770177),
    //         BaseField::from(1190378570),
    //         BaseField::from(1107054997),
    //         BaseField::from(1436440899),
    //         BaseField::from(1555024221),
    //         BaseField::from(2002021885),
    //         BaseField::from(866),
    //         BaseField::from(750797),
    //         BaseField::from(1704111751),
    //         BaseField::from(1874758341),
    //         BaseField::from(960394553),
    //         BaseField::from(1365348280),
    //         BaseField::from(376645196),
    //         BaseField::from(2119137245),
    //         BaseField::from(1),
    //     ];
    //     let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //     let coset = CanonicCoset::new(log_size);
    //     let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //     let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

    //     let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //     let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //     let expected_result = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //     let result = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

    //     assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
    // }

    // #[test]
    // fn test_interpolate_3() {

    //     for log_size in 4..30 {

    //         let size = 1 << log_size;

    //         let cpu_values = (1..(size + 1) as u32)
    //             .map(BaseField::from)
    //             .collect::<Vec<_>>();
    //         let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //         let coset = CanonicCoset::new(log_size);
    //         let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //         let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

    //         let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //         let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //         assert_eq!(cpu_twiddles.twiddles.to_vec(), gpu_twiddles.twiddles.to_vec());

    //         let timer = start_timer!(|| format!("cpu backend interpolate, log_n:{}", log_size));
    //         let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //         end_timer!(timer);

    //         let timer = start_timer!(|| format!("optimize gpu backend interpolate, log_n:{}",
    // log_size));         let gpu_poly = CudaBackend::interpolate(gpu_evaluations,
    // &gpu_twiddles);         end_timer!(timer);

    //         assert_eq!(gpu_poly.coeffs.to_vec(), cpu_poly.coeffs.to_vec());

    //     }

    // }

    // #[test]
    // #[allow(unused_variables)]
    // fn test_evaluate() {
    //     for log_size in 13..26 {

    //         let size = 1 << log_size;

    //         let cpu_values = (1..(size + 1) as u32)
    //             .map(BaseField::from)
    //             .collect::<Vec<_>>();
    //         let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());
    //         let gpu_values_optim = BaseFieldVec::from_vec(cpu_values.clone());

    //         let coset = CanonicCoset::new(log_size);
    //         let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //         let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);
    //         let gpu_evaluations_optim = CudaBackend::new_canonical_ordered(coset,
    // gpu_values_optim);

    //         let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //         let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //         assert_eq!(cpu_twiddles.twiddles.to_vec(), gpu_twiddles.twiddles.to_vec());

    //         let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //         let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
    //         let gpu_poly_optim = CudaBackend::interpolate(gpu_evaluations_optim, &gpu_twiddles);
    //         assert_eq!(gpu_poly_optim.coeffs.to_vec(), cpu_poly.coeffs.to_vec());

    //         let timer = start_timer!(|| format!("cpu backend interpolate, log_n:{}", log_size));
    //         let expected_result = CpuBackend::evaluate(&cpu_poly, coset.circle_domain(),
    // &cpu_twiddles);         end_timer!(timer);

    //         let timer = start_timer!(|| format!("native gpu backend  interpolate, log_n:{}",
    // log_size));         let result = evaluate_native(&gpu_poly, coset.circle_domain(),
    // &gpu_twiddles);         end_timer!(timer);

    //         let timer = start_timer!(|| format!("optimize gpu backend interpolate, log_n:{}",
    // log_size));         let result_optim = CudaBackend::evaluate(&gpu_poly_optim,
    // coset.circle_domain(), &gpu_twiddles);         end_timer!(timer);

    //         assert_eq!(result_optim.values.to_cpu(), expected_result.values);
    //     }

    // }

    // #[test]
    // fn test_eval_at_point() {
    //     let log_size = 20;

    //     let size = 1 << log_size;
    //     let coset = CanonicCoset::new(log_size);
    //     let point = SECURE_FIELD_CIRCLE_GEN;

    //     let cpu_values = (1..(size + 1) as u32)
    //         .map(BaseField::from)
    //         .collect::<Vec<_>>();

    //     let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());
    //     let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);
    //     let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());
    //     let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
    //     let result = CudaBackend::eval_at_point(&gpu_poly, point);

    //     let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //     let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //     let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);

    //     let expected_result = CpuBackend::eval_at_point(&cpu_poly, point.clone());

    //     assert_eq!(result, expected_result);
    // }

    #[test]
    fn test_evaluate_small_poly_on_large_domain() {
        // This tests the exact scenario in accumulator finalize:
        // A polynomial created at log_size=20 evaluated on domain of log_size=24
        use crate::prover::poly::circle::CircleEvaluation as CpuCircleEvaluation;

        const SMALL_LOG_SIZE: u32 = 20;
        const LARGE_LOG_SIZE: u32 = 24;

        let small_size = 1usize << SMALL_LOG_SIZE;
        let large_size = 1usize << LARGE_LOG_SIZE;

        // Create values at small size
        let cpu_values: Vec<BaseField> =
            (0..small_size).map(|i| BaseField::from(i as u32)).collect();
        let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

        let small_coset = CanonicCoset::new(SMALL_LOG_SIZE);
        let small_domain = small_coset.circle_domain();
        let large_coset = CanonicCoset::new(LARGE_LOG_SIZE);
        let large_domain = large_coset.circle_domain();

        // Create evaluations
        let cpu_evaluations =
            CpuCircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(small_domain, cpu_values);
        let gpu_evaluations =
            CircleEvaluation::<CudaBackend, _, BitReversedOrder>::new(small_domain, gpu_values);

        // Precompute twiddles for interpolation (small domain)
        let cpu_small_twiddles = CpuBackend::precompute_twiddles(small_coset.half_coset());
        let gpu_small_twiddles = CudaBackend::precompute_twiddles(small_coset.half_coset());

        // Interpolate to get polynomials
        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_small_twiddles);
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_small_twiddles);

        // Verify interpolation matches
        assert_eq!(gpu_poly.coeffs.to_cpu(), cpu_poly.coeffs);

        // Precompute twiddles for large domain evaluation
        let cpu_large_twiddles = CpuBackend::precompute_twiddles(large_coset.half_coset());
        let gpu_large_twiddles = CudaBackend::precompute_twiddles(large_coset.half_coset());

        // Evaluate both on the LARGE domain
        let cpu_eval_result = CpuBackend::evaluate(&cpu_poly, large_domain, &cpu_large_twiddles);
        let gpu_eval_result = CudaBackend::evaluate(&gpu_poly, large_domain, &gpu_large_twiddles);

        let cpu_result = cpu_eval_result.values;
        let gpu_result = gpu_eval_result.values.to_cpu();

        assert_eq!(cpu_result.len(), large_size);
        assert_eq!(gpu_result.len(), large_size);

        // Check first 1000 elements
        assert_eq!(
            cpu_result[..1000],
            gpu_result[..1000],
            "First 1000 elements mismatch"
        );
        // Check last 1000 elements
        assert_eq!(
            cpu_result[large_size - 1000..],
            gpu_result[large_size - 1000..],
            "Last 1000 elements mismatch"
        );
        // Check middle elements
        let mid = large_size / 2;
        assert_eq!(
            cpu_result[mid..mid + 1000],
            gpu_result[mid..mid + 1000],
            "Middle 1000 elements mismatch"
        );
    }

    #[test]
    fn test_interpolate_from_fib() {
        let eval = CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
            CircleDomain {
                half_coset: Coset {
                    initial_index: CirclePointIndex(33554432),
                    initial: CirclePoint {
                        x: BaseField::from(579625837),
                        y: BaseField::from(1690787918),
                    },
                    step_size: CirclePointIndex(134217728),
                    step: CirclePoint {
                        x: BaseField::from(590768354),
                        y: BaseField::from(978592373),
                    },
                    log_size: 4,
                },
            },
            BaseFieldVec::from_vec(vec![
                BaseField::from(1),
                BaseField::from(443693538),
                BaseField::from(793699796),
                BaseField::from(1631104375),
                BaseField::from(460025527),
                BaseField::from(98131605),
                BaseField::from(1292025643),
                BaseField::from(1056169651),
                BaseField::from(29),
                BaseField::from(1645907698),
                BaseField::from(300234932),
                BaseField::from(2113642380),
                BaseField::from(2031046861),
                BaseField::from(541052612),
                BaseField::from(1857203558),
                BaseField::from(5),
                BaseField::from(2),
                BaseField::from(187770177),
                BaseField::from(1190378570),
                BaseField::from(1107054997),
                BaseField::from(1436440899),
                BaseField::from(1555024221),
                BaseField::from(2002021885),
                BaseField::from(866),
                BaseField::from(750797),
                BaseField::from(1704111751),
                BaseField::from(1874758341),
                BaseField::from(960394553),
                BaseField::from(1365348280),
                BaseField::from(376645196),
                BaseField::from(2119137245),
                BaseField::from(1),
            ]),
        );
        let twiddles = vec![
            BaseField::from(785043271),
            BaseField::from(1260750973),
            BaseField::from(736262640),
            BaseField::from(1553669210),
            BaseField::from(479120236),
            BaseField::from(225856549),
            BaseField::from(197700101),
            BaseField::from(1079800039),
            BaseField::from(1911378744),
            BaseField::from(1577470940),
            BaseField::from(1334497267),
            BaseField::from(2085743640),
            BaseField::from(477953613),
            BaseField::from(125103457),
            BaseField::from(1977033713),
            BaseField::from(2005527287),
            BaseField::from(251924953),
            BaseField::from(636875771),
            BaseField::from(48903418),
            BaseField::from(1896945393),
            BaseField::from(1514613395),
            BaseField::from(870936612),
            BaseField::from(1297878576),
            BaseField::from(583555490),
            BaseField::from(640817200),
            BaseField::from(1702126977),
            BaseField::from(1054411686),
            BaseField::from(648593218),
            BaseField::from(1014093253),
            BaseField::from(2137011181),
            BaseField::from(81378258),
            BaseField::from(789857006),
            BaseField::from(838195206),
            BaseField::from(1774253895),
            BaseField::from(1739004854),
            BaseField::from(262191051),
            BaseField::from(206059115),
            BaseField::from(212443077),
            BaseField::from(1796741361),
            BaseField::from(883753057),
            BaseField::from(2140339328),
            BaseField::from(404685994),
            BaseField::from(9803698),
            BaseField::from(68458636),
            BaseField::from(14530030),
            BaseField::from(228509164),
            BaseField::from(1038945916),
            BaseField::from(134155457),
            BaseField::from(579625837),
            BaseField::from(1690787918),
            BaseField::from(1641940819),
            BaseField::from(2121318970),
            BaseField::from(1952787376),
            BaseField::from(1580223790),
            BaseField::from(1013961365),
            BaseField::from(280947147),
            BaseField::from(1179735656),
            BaseField::from(1241207368),
            BaseField::from(1415090252),
            BaseField::from(2112881577),
            BaseField::from(590768354),
            BaseField::from(978592373),
            BaseField::from(32768),
            BaseField::from(1),
        ];
        let itwiddles = vec![
            BaseField::from(1541158724),
            BaseField::from(16208603),
            BaseField::from(62823040),
            BaseField::from(1642210396),
            BaseField::from(1631996251),
            BaseField::from(1007591000),
            BaseField::from(1874949287),
            BaseField::from(1849862501),
            BaseField::from(781334166),
            BaseField::from(132945364),
            BaseField::from(1278220752),
            BaseField::from(214347122),
            BaseField::from(1165838173),
            BaseField::from(2054194025),
            BaseField::from(1234096940),
            BaseField::from(1721693449),
            BaseField::from(622651690),
            BaseField::from(1373671071),
            BaseField::from(82740187),
            BaseField::from(1683898894),
            BaseField::from(1918467639),
            BaseField::from(1186332607),
            BaseField::from(1296073347),
            BaseField::from(401388709),
            BaseField::from(1383565722),
            BaseField::from(656788371),
            BaseField::from(1787268380),
            BaseField::from(1809670981),
            BaseField::from(99372120),
            BaseField::from(765975505),
            BaseField::from(774809712),
            BaseField::from(348924564),
            BaseField::from(2029303208),
            BaseField::from(959596234),
            BaseField::from(1051468699),
            BaseField::from(721860568),
            BaseField::from(1767118503),
            BaseField::from(218253990),
            BaseField::from(1356867335),
            BaseField::from(1955048591),
            BaseField::from(559361447),
            BaseField::from(1046725194),
            BaseField::from(448375059),
            BaseField::from(1036402186),
            BaseField::from(2138687850),
            BaseField::from(1268642696),
            BaseField::from(1381082522),
            BaseField::from(559888787),
            BaseField::from(248349974),
            BaseField::from(969924856),
            BaseField::from(1461702947),
            BaseField::from(655012266),
            BaseField::from(1385854532),
            BaseField::from(1859156789),
            BaseField::from(349252128),
            BaseField::from(421110815),
            BaseField::from(1160411471),
            BaseField::from(1518526074),
            BaseField::from(490549293),
            BaseField::from(1942501404),
            BaseField::from(991237807),
            BaseField::from(775648038),
            BaseField::from(65536),
            BaseField::from(1),
        ];
        let root_coset = Coset {
            initial_index: CirclePointIndex(8388608),
            initial: CirclePoint {
                x: BaseField::from(785043271),
                y: BaseField::from(1260750973),
            },
            step_size: CirclePointIndex(33554432),
            step: CirclePoint {
                x: BaseField::from(579625837),
                y: BaseField::from(1690787918),
            },
            log_size: 6,
        };
        let twiddle_tree = TwiddleTree::<CudaBackend> {
            root_coset,
            twiddles: BaseFieldVec::from_vec(twiddles),
            itwiddles: BaseFieldVec::from_vec(itwiddles),
        };

        let cpu_evaluation = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            eval.domain,
            eval.values.to_cpu(),
        );
        let cpu_twiddle_tree = TwiddleTree::<CpuBackend> {
            root_coset: twiddle_tree.root_coset.clone(),
            twiddles: twiddle_tree.twiddles.to_cpu(),
            itwiddles: twiddle_tree.itwiddles.to_cpu(),
        };
        let expected_result = CpuBackend::interpolate(cpu_evaluation, &cpu_twiddle_tree);
        let result = CudaBackend::interpolate(eval, &twiddle_tree);
        assert_eq!(expected_result.coeffs, result.coeffs.to_cpu());
    }

    // #[test_log::test]
    // fn test_interpolate_columns() {
    //     // use crate::prover::backend::cuda::poly::interpolate_columns_native;
    //     let log_number_of_columns = 7;

    //     for log_size in 13..17 {
    //         let size = 1 << log_size;
    //         let number_of_columns = 1 << log_number_of_columns;
    //         let cpu_values = (1..(size + 1) as u32).map(BaseField::from).collect_vec();
    //         let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //         let coset = CanonicCoset::new(log_size);
    //         let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //         let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

    //         let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //         let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //         let cpu_columns = (0..number_of_columns)
    //             .map(|_index| cpu_evaluations.clone())
    //             .collect_vec();
    //         let gpu_columns = (0..number_of_columns)
    //             .map(|_index| gpu_evaluations.clone())
    //             .collect_vec();

    //         let timer = start_timer!(|| format!("cpu backend interpolate_columns, column:{}
    // log_n:{}", 1<<log_number_of_columns, log_size));         let expected_result =
    // CpuBackend::interpolate_columns(cpu_columns, &cpu_twiddles);         end_timer!(timer);

    //         // let timer = start_timer!(|| format!("gpu backend native interpolate_columns,
    // column:{} log_n:{}", 1<<log_number_of_columns, log_size));         // let result =
    // interpolate_columns_native(gpu_columns.clone(), &gpu_twiddles);         //
    // end_timer!(timer);

    //         let timer = start_timer!(|| format!("cuda backend optimize interpolate_columns,
    // column:{} log_n:{}", 1<<log_number_of_columns, log_size));         let result_optim =
    // CudaBackend::interpolate_columns(gpu_columns, &gpu_twiddles);         end_timer!(timer);

    //         let expected_coeffs = expected_result
    //             .iter()
    //             .map(|poly| poly.coeffs.clone())
    //             .collect_vec();
    //         let coeffs = result
    //             .iter()
    //             .map(|poly| poly.coeffs.clone().to_cpu())
    //             .collect_vec();
    //         let coeffs_optim = result_optim
    //             .iter()
    //             .map(|poly| poly.coeffs.clone().to_cpu())
    //             .collect_vec();

    //         assert_eq!(expected_coeffs, coeffs_optim);
    //     }
    // }

    // #[allow(unused_variables)]
    // #[test_log::test]
    // fn test_evaluate_columns() {
    //     let log_blowup_factor = 2;
    //     let log_number_of_columns = 7;

    //     for log_size in 13..20-log_blowup_factor {

    //         let size = 1 << log_size;
    //         let number_of_columns = 1 << log_number_of_columns;

    //         let cpu_values = (1..(size + 1) as u32)
    //             .map(BaseField::from)
    //             .collect::<Vec<_>>();
    //         let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //         let trace_coset = CanonicCoset::new(log_size);
    //         let cpu_evaluations = CpuBackend::new_canonical_ordered(trace_coset, cpu_values);
    //         let gpu_evaluations = CudaBackend::new_canonical_ordered(trace_coset,
    // gpu_values.clone());         let gpu_evaluations_ref =
    // CudaBackend::new_canonical_ordered(trace_coset, gpu_values);

    //         let interpolation_coset = CanonicCoset::new(log_size + log_blowup_factor);
    //         let cpu_twiddles = CpuBackend::precompute_twiddles(interpolation_coset.half_coset());
    //         let gpu_twiddles =
    // CudaBackend::precompute_twiddles(interpolation_coset.half_coset());

    //         let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //         let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

    //         assert_eq!(gpu_poly.coeffs.to_vec(), cpu_poly.coeffs.to_vec());

    //         let mut cpu_columns: Vec<CircleCoefficients<CpuBackend>> = ColumnVec::from(
    //             (0..number_of_columns)
    //                 .map(|_index| cpu_poly.clone())
    //                 .collect_vec(),
    //         );
    //         let mut gpu_columns = ColumnVec::from(
    //             (0..number_of_columns)
    //                 .map(|_index| gpu_poly.clone())
    //                 .collect_vec(),
    //         );

    //         let timer = start_timer!(|| format!("cpu backend optimize evaluate_polynomials,
    // column:{} log_n:{}", 1<<log_number_of_columns, log_size));         let expected_result =
    // CpuBackend::evaluate_polynomials(&mut cpu_columns, log_blowup_factor, &cpu_twiddles);
    //         end_timer!(timer);

    //         let timer = start_timer!(|| format!("cuda backend evaluate_polynomials, column:{}
    // log_n:{}", 1<<log_number_of_columns, log_size));         let result =
    // CudaBackend::evaluate_polynomials(&mut gpu_columns, log_blowup_factor, &gpu_twiddles);
    //         end_timer!(timer);

    //         let expected_values = expected_result
    //             .iter()
    //             .map(|eval| eval.clone().values)
    //             .collect_vec();
    //         let values = result
    //             .iter()
    //             .map(|eval| eval.clone().values.to_cpu())
    //             .collect_vec();

    //         assert_eq!(values, expected_values);
    //     }
    // }

    #[test]
    fn test_eval_at_point_log24() {
        use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;

        const LOG_SIZE: u32 = 24;
        let size = 1usize << LOG_SIZE;

        // Create a polynomial of log_size=24
        let cpu_values: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();
        let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(LOG_SIZE);
        let domain = coset.circle_domain();

        let cpu_evaluations =
            CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(domain, cpu_values);
        let gpu_evaluations =
            CircleEvaluation::<CudaBackend, _, BitReversedOrder>::new(domain, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        // Verify polynomials match
        assert_eq!(
            gpu_poly.coeffs.to_cpu(),
            cpu_poly.coeffs,
            "Polynomial coeffs mismatch"
        );

        // Test eval_at_point at SECURE_FIELD_CIRCLE_GEN (this is what's used in OODS)
        let point = SECURE_FIELD_CIRCLE_GEN;
        let cpu_result = CpuBackend::eval_at_point(&cpu_poly, point);
        let gpu_result = CudaBackend::eval_at_point(&gpu_poly, point);

        assert_eq!(
            gpu_result, cpu_result,
            "eval_at_point mismatch at SECURE_FIELD_CIRCLE_GEN"
        );

        // Test at another arbitrary point
        let point2 = CirclePoint::get_point(12345678);
        let cpu_result2 = CpuBackend::eval_at_point(&cpu_poly, point2);
        let gpu_result2 = CudaBackend::eval_at_point(&gpu_poly, point2);

        assert_eq!(
            gpu_result2, cpu_result2,
            "eval_at_point mismatch at arbitrary point"
        );
    }

    #[test]
    fn test_barycentric_weights_gpu_matches_cpu() {
        use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
        use crate::core::fields::qm31::SecureField;

        let point = SECURE_FIELD_CIRCLE_GEN;

        for log_size in [4u32, 5, 8, 10] {
            let coset = CanonicCoset::new(log_size);
            let cpu_weights = CpuBackend::barycentric_weights(coset, point);
            let gpu_weights = CudaBackend::barycentric_weights(coset, point);
            let gpu_weights_cpu = gpu_weights.to_vec();
            assert_eq!(
                cpu_weights, gpu_weights_cpu,
                "barycentric_weights mismatch at log_size={}",
                log_size
            );
        }
    }

    #[test]
    fn test_barycentric_eval_at_point_gpu_matches_cpu() {
        use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
        use crate::core::fields::qm31::SecureField;

        let point = SECURE_FIELD_CIRCLE_GEN;

        for log_size in [4u32, 5, 8, 10] {
            let coset = CanonicCoset::new(log_size);
            let domain = coset.circle_domain();
            let values: Vec<BaseField> = (0..domain.size())
                .map(|i| BaseField::from(i as u32))
                .collect();

            let cpu_eval =
                CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(domain, values.clone());
            let gpu_eval = CircleEvaluation::<CudaBackend, _, BitReversedOrder>::new(
                domain,
                BaseFieldVec::from_vec(values),
            );

            let cpu_weights = CpuBackend::barycentric_weights(coset, point);
            let gpu_weights = CudaBackend::barycentric_weights(coset, point);

            let cpu_result = CpuBackend::barycentric_eval_at_point(&cpu_eval, &cpu_weights);
            let gpu_result = CudaBackend::barycentric_eval_at_point(&gpu_eval, &gpu_weights);
            assert_eq!(
                cpu_result, gpu_result,
                "barycentric_eval_at_point mismatch at log_size={}",
                log_size
            );
        }
    }
}
