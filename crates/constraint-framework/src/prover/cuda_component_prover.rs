//! `ComponentProver<CudaBackend>` for `FrameworkComponent`.
//!
//! Unlike `GpuBackend` (obelyzk), whose column layout is byte-identical to `SimdBackend`,
//! the device-resident `CudaBackend` stores committed columns as device `BaseFieldVec`s. A
//! `mem::transmute` to `SimdBackend` (as done in `gpu_component_prover`) is therefore UNSOUND.
//!
//! Correctness-first v1: we perform a REAL conversion. Each committed column referenced by the
//! `Trace<'_, CudaBackend>` is copied to host via `Column::to_cpu()`, producing an owned
//! `Trace<'_, CpuBackend>`. We then run the audited `CpuBackend` constraint-quotient evaluation
//! into a host `SecureColumnByCoords<CpuBackend>` seeded with the current accumulator contents,
//! and upload the result back into the `DomainEvaluationAccumulator<CudaBackend>`'s device column.
//!
//! This mirrors the SIMD small-trace CPU fallback in `component_prover.rs`: it shares the exact
//! same `accumulate_pointwise_cpu` routine and the same `random_coeff_powers` (split off the
//! Cuda accumulator), so the accumulated composition-polynomial column is bit-identical to the
//! SimdBackend / CpuBackend result. The constraint-evaluation D2H/H2D round trip is acceptable
//! for v1 (constraint eval is a small fraction of prove time vs commit/FRI); an on-device CUDA
//! constraint kernel is future work.
//!
//! # Why there is no device-resident GPU constraint path here (2026-06-28 investigation)
//!
//! Two candidate GPU paths were assessed and BOTH rejected on soundness grounds:
//!
//! 1. Reviving obelyzk's generic `GpuDomainEvaluator` (`gpu_domain.rs`) against `CudaBackend`
//!    device columns is UNSOUND: its `off == 0` fast path calls
//!    `VeryPackedBaseColumn::transform_under_ref(col)` on the column's backing store, which
//!    reinterprets *host* SIMD memory. A `CudaBackend` `BaseFieldVec` is a raw device pointer, so
//!    that transmute is undefined behavior; its non-zero-offset path would also issue one
//!    single-element D2H copy (`Column::at`) per masked value. (Independently, obelyzk's evaluator
//!    already produced WRONG values on real gate_air even on the host-layout `GpuBackend` — see the
//!    note in `gpu_component_prover.rs` — so it is not a trustworthy reference to port.)
//!
//! 2. The NitrooZK device path (`stwo_cuda/cuda/evaluate_constraints.cu`, FFI-exposed as
//!    `bindings::evaluate_constraint_quotients_on_domain`) is NOT generic: it `switch`es on an
//!    `eval_id` (FNV-1a hash of a component name) into hand-written, per-component CUDA kernels
//!    (blake/poseidon/cairo opcodes/range-checks/wide_fibonacci). gate_air's `GateEval` is not in
//!    that switch, so the binding would return `false` ("unsupported, fall back to CPU"). Enabling
//!    it for gate_air would require authoring a NEW, soundness-critical, gate_air-specific CUDA
//!    kernel by hand — exactly the previously-abandoned, box-unvalidated risk we must avoid here.
//!    There is no GENERIC `FrameworkEval -> CUDA` path in this tree.
//!
//! Therefore the audited host-delegate below is the ONLY correct path and is the DEFAULT. The env
//! var `CUDA_CONSTRAINT_CPU_FALLBACK` is honored for forward-compatibility / A-B testing: it is
//! read on every call, but since no trustworthy GPU path exists yet, both settings currently route
//! to this same audited host-delegate. When a verified device kernel lands, gate the new path on
//! `!cpu_fallback_forced()` and leave this delegate reachable via `CUDA_CONSTRAINT_CPU_FALLBACK=1`.

use std::borrow::Cow;

use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::fields::m31::BaseField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{ComponentProver, DomainEvaluationAccumulator, Poly, Trace};

use super::component_prover::{
    accumulate_pointwise_cpu, get_constraint_quotient_inputs, ConstraintQuotientInputs,
};
use super::cuda_constraint_kernel::{
    gpu_constraints_opt_in, registered_gpu_constraint_kernel, GpuConstraintDispatch,
};
use crate::{
    FrameworkComponent, FrameworkEval, INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX,
    PREPROCESSED_TRACE_IDX,
};

/// Whether the operator has forced the audited CPU-delegate constraint path via
/// `CUDA_CONSTRAINT_CPU_FALLBACK=1`. Read fresh on each call (cheap; once per component per prove).
///
/// Contract for future work: the DEFAULT (unset / "0") is meant to select the on-device GPU
/// constraint path once one is verified; "1" forces this audited host-delegate. Until a
/// trustworthy GPU path exists, BOTH settings route to the host-delegate, so this function only
/// affects diagnostics today.
fn cpu_fallback_forced() -> bool {
    matches!(
        std::env::var("CUDA_CONSTRAINT_CPU_FALLBACK").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// PANIC (no silent fallback) when the big gate_air MAIN component falls to the host-delegate
/// constraint path on `CudaBackend`. That path runs composition_eval ~60x slower than the GPU
/// kernel, so silently taking it turns a GPU prove into a benchmark of the WRONG (slow) code — the
/// exact regression that let a stale kernel be measured undetected. On the GPU/CudaBackend path the
/// gate_air kernel is REQUIRED, so reaching the host delegate for the MAIN component is a hard error.
///
/// STRUCTURAL FINGERPRINT: fires for the gate_air MAIN component ONLY (large AND many constraints),
/// never for the small fixed-size table components (rc / program / boundary), whose host-delegation
/// is normal (they have no GPU kernel). A table is EITHER small (rc 2^16, program ~2^4) OR — if
/// sample-scaled (boundary = n_shots*512) — carries only a handful of constraints (booleanity +
/// finalize_logup). The gate_air MAIN component is the only one that is BOTH large (>= 2^18 rows;
/// ~2^22-2^25 in the benchmark) AND carries many constraints (the 22-col chain-lookup AIR: 15
/// algebraic + 7 LogUp = 22). Requiring BOTH `n_constraints >= 15` AND `log_n_rows >= 18` separates
/// MAIN from every table with wide margin (tables have <= ~3 constraints), and neither bound is tied
/// to the exact column count, so it survives AIR-shape churn. NOTE the old `n_constraints > 50`
/// warning threshold was a stale relic of the 188-col AIR (157 constraints); the 26->22-col
/// reduction dropped MAIN to 22 constraints, so `22 > 50` went false and the warning silently died,
/// which is exactly how the slow host-delegate run went unnoticed. `>= 15` sits below the 22-col
/// AIR and far above any table.
///
/// ESCAPE HATCH: honored ONLY when the operator has EXPLICITLY forced the host path via
/// `CUDA_CONSTRAINT_CPU_FALLBACK=1` (the intentional CPU-vs-GPU composition byte-identity diff), in
/// which case the host delegate is the deliberate path and must not panic. This function is only
/// reachable from `ComponentProver<CudaBackend>`, so the CPU/SimdBackend build never calls it.
fn panic_if_main_host_delegate(n_constraints: usize, log_n_rows: u32) {
    let is_gate_air_main = n_constraints >= 15 && log_n_rows >= 18;
    if !is_gate_air_main {
        return;
    }
    // Deliberate host-delegate — the operator explicitly asked for it, either by forcing the CPU
    // fallback (CUDA_CONSTRAINT_CPU_FALLBACK=1, the CPU-vs-GPU byte-identity diff) or by opting out
    // of the GPU constraint path entirely (CUDA_GPU_CONSTRAINTS=0). Both are sanctioned host runs.
    if cpu_fallback_forced() || !gpu_constraints_opt_in() {
        return;
    }
    panic!(
        "gate_air MAIN component fell to the audited HOST-DELEGATE constraint path on CudaBackend \
         (the fast gate_air GPU constraint kernel did NOT engage) — composition_eval would run \
         ~60x slower, benchmarking the WRONG path. The kernel either was not registered \
         (gate_air_cuda_kernel::register), declined on a structural mismatch (is_gate_air_main \
         decline-guard constants stale vs the AIR), or its drawn relation was not installed \
         (set_gate_air_relation). Refusing to silently fall back. \
         (n_constraints={n_constraints}, log_n_rows={log_n_rows}). \
         To intentionally run the host delegate (e.g. the CPU-vs-GPU byte-identity diff), set \
         CUDA_CONSTRAINT_CPU_FALLBACK=1."
    );
}

/// Copies a single committed `Poly<CudaBackend>` to a host `Poly<CpuBackend>` by moving its
/// evaluations (and FFT-basis coefficients, when present) off the device via `to_cpu()`.
fn poly_to_cpu(poly: &Poly<CudaBackend>) -> Poly<CpuBackend> {
    let evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
        poly.evals.domain,
        poly.evals.values.to_cpu(),
    );
    let coeffs = poly
        .coeffs
        .as_ref()
        .map(|c| CircleCoefficients::<CpuBackend>::new(c.coeffs.to_cpu()));
    Poly::new(coeffs, evals)
}

/// A tiny valid host placeholder `Poly<CpuBackend>` used for the columns this component does NOT
/// read. It is never indexed by the eval (see [`build_scoped_host_polys`]), so its contents are
/// irrelevant; it merely fills the [`TreeVec`] slots so global column indexing is preserved.
fn dummy_host_poly() -> Poly<CpuBackend> {
    // log_size must be > 0 (CanonicCoset rejects 0); size-2 is the minimal valid placeholder
    // domain. The poly is never dereferenced by the eval, so its size is irrelevant.
    let domain = CanonicCoset::new(1).circle_domain();
    let evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
        domain,
        vec![BaseField::zero(); domain.size()],
    );
    Poly::new(None, evals)
}

/// STEP 2 (GATE_AIR_STREAM_COMMIT): rehydrate this component's host-staged eval columns back to the
/// device so the GPU constraint kernel can read them, preserving the EXACT global [`TreeVec`]
/// shape.
///
/// Returns `None` when NO column this component reads is staged — the common/legacy case — so the
/// caller dispatches on the committed device trace unchanged (byte-for-byte). When at least one
/// read column is staged, returns an owned `TreeVec` in which:
///   * every column THIS component reads (its [`stwo::core::pcs::TreeSubspan`]s + the preprocessed
///     column indices) is a device-resident rehydration ([`fused_commit::rehydrate_owned`], H2D
///     from the stash) if staged, else a device clone of the resident committed column;
///   * every other slot is a tiny never-read placeholder ([`dummy_device_poly`]), keeping the
///     global `(tree, column)` indexing bit-identical so `sub_tree(trace_locations)` /
///     `[PREPROCESSED_TRACE_IDX][idx]` resolve to the same data as the committed trace.
///
/// The returned polys OWN their device buffers; dropping the value frees them, so composition
/// residency is bounded to this component's read set and released afterwards. The rehydrated bytes
/// equal the committed bytes (same u32 payload), so the composition result is identical.
fn build_scoped_device_trace<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    device_polys: &TreeVec<ColumnVec<&Poly<CudaBackend>>>,
) -> Option<TreeVec<ColumnVec<Poly<CudaBackend>>>> {
    use stwo::prover::backend::cuda::fused_commit;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

    // Mark which (tree, column) positions the eval reads.
    let mut needed: Vec<Vec<bool>> = device_polys
        .iter()
        .map(|tree| vec![false; tree.len()])
        .collect();
    for location in component.trace_locations() {
        for col in location.col_start..location.col_end {
            needed[location.tree_index][col] = true;
        }
    }
    for &idx in component.preprocessed_column_indices() {
        needed[PREPROCESSED_TRACE_IDX][idx] = true;
    }

    // Only rehydrate if a NEEDED column is actually staged; otherwise leave the committed trace
    // alone (legacy/resident path, byte-for-byte unchanged).
    let any_needed_staged = device_polys.iter().enumerate().any(|(t, tree)| {
        tree.iter()
            .enumerate()
            .any(|(c, poly)| needed[t][c] && fused_commit::is_staged(&poly.evals.values))
    });
    if !any_needed_staged {
        return None;
    }

    Some(TreeVec::new(
        device_polys
            .iter()
            .enumerate()
            .map(|(tree_index, tree)| {
                tree.iter()
                    .enumerate()
                    .map(|(col, poly)| {
                        if !needed[tree_index][col] {
                            return dummy_device_poly();
                        }
                        // COMPOSITION_TILING_SCOPE (route c): a NEEDED, STAGED column in tree0
                        // (preprocessed) or tree1 (main) is NOT rehydrated whole here — that whole-
                        // column H2D is exactly the ~47 GB residency that still OOMs at 2^24.
                        // Instead we pass through a NON-OWNING
                        // `BaseFieldVec` carrying the column's ORIGINAL
                        // (freed) device pointer, which is the fused_commit HOST_STASH key. The
                        // downstream gate_air GPU kernel detects the stage via `staged_host_ptr`,
                        // builds its host-tile-source table, and row-tiles the H2D per block — so
                        // this device pointer is used ONLY as a stash key,
                        // never dereferenced. `owns_memory` is false, so
                        // Drop never double-frees the already-freed committed buffer.
                        //
                        // F2-b / Option B (tree2 COMPOSITION streaming): tree2 (interaction) is now
                        // ALSO passed through as a NON-OWNING stash-key column, exactly like
                        // tree0/tree1. The downstream gate_air kernel row-tiles tree2 and replaces
                        // the scattered `-1` composition read with an offset-0 read on 4 precomputed
                        // shifted columns (interaction_shift_neg1), so tree2 no longer needs to be
                        // held whole (~14 GiB @2^26). The kernel resolves the staged bytes via
                        // `staged_host_ptr` and H2Ds per block. Previously tree2 was rehydrated WHOLE
                        // here (the H1 residency wall); that whole-rehydrate is now gone.
                        // Non-staged needed columns are still device-cloned so the owned trace fully
                        // owns its buffers (legacy path, byte-for-byte unchanged).
                        let is_input_tree = tree_index == PREPROCESSED_TRACE_IDX
                            || tree_index == ORIGINAL_TRACE_IDX
                            || tree_index == INTERACTION_TRACE_IDX;
                        let staged = fused_commit::is_staged(&poly.evals.values);
                        let evals_values = if staged && is_input_tree {
                            // Non-owning passthrough of the stash-key pointer (row-tiled
                            // downstream).
                            BaseFieldVec::from_borrowed_ptr(
                                poly.evals.values.device_ptr,
                                poly.evals.values.size,
                            )
                        } else if staged {
                            // tree2 (or any other resident-required tree): rehydrate whole.
                            fused_commit::rehydrate_owned(&poly.evals.values)
                        } else {
                            poly.evals.values.clone()
                        };
                        let evals =
                            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                                poly.evals.domain,
                                evals_values,
                            );
                        // Coeffs are not read by the eval-domain constraint kernel; omit them.
                        Poly::new(None, evals)
                    })
                    .collect()
            })
            .collect(),
    ))
}

/// A tiny valid device placeholder `Poly<CudaBackend>` for the columns this component does NOT read
/// (never dereferenced by the eval; fills the [`TreeVec`] slot so global indexing is preserved).
fn dummy_device_poly() -> Poly<CudaBackend> {
    use stwo::prover::backend::Column;
    let domain = CanonicCoset::new(1).circle_domain();
    let values =
        <CudaBackend as stwo::prover::backend::ColumnOps<BaseField>>::Column::zeros(domain.size());
    let evals = CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, values);
    Poly::new(None, evals)
}

/// Builds a host copy of `device_polys` that preserves the EXACT global [`TreeVec`] shape (same
/// number of trees, same number of columns per tree, in the same positions) but only performs a
/// device->host copy ([`poly_to_cpu`]) for the columns THIS component actually consumes.
///
/// The audited host eval ([`get_constraint_quotient_inputs`]) reads exactly:
///   * every column in each [`stwo::core::pcs::TreeSubspan`] of `trace_locations` (the
///     `col_start..col_end` range within `tree_index`), via
///     `trace.polys.sub_tree(trace_locations)`,
///   * plus, in tree [`PREPROCESSED_TRACE_IDX`], the columns at `preprocessed_column_indices`
///     (which fully replace the preprocessed sub-tree slot).
/// Every other column is a dead copy in the old full-trace path. Here those slots get a shared
/// [`dummy_host_poly`] which the eval never dereferences. Because the materialized columns sit at
/// their original global `(tree, column)` positions, `sub_tree(trace_locations)` and
/// `[PREPROCESSED_TRACE_IDX][idx]` resolve to bit-identical data versus the full-trace copy.
///
/// Returns owned per-column storage; the caller builds the borrowed `Trace` from it.
fn build_scoped_host_polys<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    device_polys: &TreeVec<ColumnVec<&Poly<CudaBackend>>>,
) -> TreeVec<ColumnVec<Poly<CpuBackend>>> {
    // Mark which (tree, column) positions are read by the eval.
    let mut needed: Vec<Vec<bool>> = device_polys
        .iter()
        .map(|tree| vec![false; tree.len()])
        .collect();

    for location in component.trace_locations() {
        for col in location.col_start..location.col_end {
            needed[location.tree_index][col] = true;
        }
    }
    for &idx in component.preprocessed_column_indices() {
        needed[PREPROCESSED_TRACE_IDX][idx] = true;
    }

    // Materialize only the needed columns; everything else is a never-read placeholder.
    TreeVec::new(
        device_polys
            .iter()
            .enumerate()
            .map(|(tree_index, tree)| {
                tree.iter()
                    .enumerate()
                    .map(|(col, poly)| {
                        if needed[tree_index][col] {
                            poly_to_cpu(poly)
                        } else {
                            dummy_host_poly()
                        }
                    })
                    .collect()
            })
            .collect(),
    )
}

impl<E: FrameworkEval + Sync> ComponentProver<CudaBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
    ) {
        if self.n_constraints() == 0 {
            return;
        }

        // No trustworthy device-resident GPU constraint path exists yet (see module docs), so we
        // always take the audited host-delegate. Reading the flag here keeps the A/B contract live
        // and documents which path ran; today it is informational only.
        let _cpu_fallback_forced = cpu_fallback_forced();

        // ============================================================================
        // OPT-IN device-resident GPU constraint kernel (circuit-specific, registered downstream).
        //
        // Taken ONLY when `CUDA_GPU_CONSTRAINTS=1`, a kernel was installed via
        // `set_gpu_constraint_kernel`, and `CUDA_CONSTRAINT_CPU_FALLBACK` is NOT forcing the host
        // path. The kernel inspects the component (e.g. by structural fingerprint) and returns
        // `false` for any component that is not its target AIR, in which case we fall through to
        // the audited host-delegate below. This is a SEPARATE, additive branch IN FRONT of the
        // host delegate; the host-delegate code below is unchanged and remains the DEFAULT + the
        // fallback. No host constraint-eval math is affected by this branch.
        // ============================================================================
        if gpu_constraints_opt_in() && !_cpu_fallback_forced {
            if let Some(kernel) = registered_gpu_constraint_kernel() {
                // STEP 2 (GATE_AIR_STREAM_COMMIT): the streamed commit freed this component's
                // committed eval columns (bytes live in the streaming-commit-layer stash). The GPU
                // constraint kernel reads those columns' device buffers, so we REHYDRATE only the
                // columns THIS component reads (via `trace_locations` +
                // `preprocessed_column_indices`) back to the device (H2D from the
                // stash), build a resident device trace, and dispatch the SAME GPU
                // kernel on it — composition stays on GPU (no host-delegate),
                // and the `panic_if_main_host_delegate` path is NOT reached for staged shards. The
                // rehydrated columns are OWNED by `resident_polys` and freed when it drops at the
                // end of this block, so device residency during composition is
                // bounded to this component's read set (transient), then released.
                // Bit-identical to the resident path (same committed bytes, same
                // kernel). When nothing is staged (legacy/resident
                // path) `build_scoped_device_trace` returns `None` and we dispatch on the committed
                // trace directly — byte-for-byte unchanged.
                let resident_polys = build_scoped_device_trace(self, &trace.polys);
                let scoped_trace;
                let dispatch_trace: &Trace<'_, CudaBackend> = match &resident_polys {
                    Some(polys) => {
                        scoped_trace = Trace {
                            polys: polys.as_cols_ref(),
                        };
                        &scoped_trace
                    }
                    None => trace,
                };

                // Device-resident constraint-quotient inputs (no D2H) on the CudaBackend trace.
                let inputs = get_constraint_quotient_inputs(
                    self,
                    dispatch_trace,
                    evaluation_accumulator.evaluation_mode(),
                );
                let dispatch = GpuConstraintDispatch {
                    inputs: &inputs,
                    n_constraints: self.n_constraints(),
                    claimed_sum: self.claimed_sum(),
                    log_n_rows: self.eval.log_size(),
                    accumulator: evaluation_accumulator,
                };
                if kernel(dispatch) {
                    return;
                }
                // Kernel declined (not its target AIR) -> fall through to the audited host
                // delegate.
            }
        }

        // Reaching here means this component takes the audited host-delegate constraint path. That
        // is normal for the small table components, but a hard error for the big gate_air MAIN
        // component on CudaBackend: the GPU kernel is REQUIRED there, so silently running the ~60x
        // slower host delegate would benchmark the wrong path. PANIC (no silent fallback) unless the
        // operator explicitly forced the host path (CUDA_CONSTRAINT_CPU_FALLBACK=1).
        panic_if_main_host_delegate(self.n_constraints(), self.eval.log_size());

        // Move ONLY this component's committed trace polynomials to host. The audited host eval
        // (`get_constraint_quotient_inputs`) reads exactly the columns in `trace_locations` plus
        // the `preprocessed_column_indices`; every other column was a dead D2H copy in the old
        // full-trace path. `build_scoped_host_polys` preserves the global TreeVec shape (so the
        // eval's `sub_tree(trace_locations)` / `[PREPROCESSED_TRACE_IDX][idx]` indexing is
        // bit-identical), but only copies the needed columns off the device. `cpu_polys` owns the
        // data; `cpu_trace` borrows from it, matching the `&'a Poly<B>` lifetime expected by
        // `Trace`.
        let cpu_polys = build_scoped_host_polys(self, &trace.polys);
        let cpu_trace = Trace {
            polys: cpu_polys.as_cols_ref(),
        };

        // Build the constraint-quotient inputs on the host trace, using the Cuda accumulator's
        // evaluation mode (the mode is backend-agnostic).
        let ConstraintQuotientInputs {
            eval_domain,
            trace_domain,
            trace: cpu_trace_cols,
            denom_inv,
        } = get_constraint_quotient_inputs(
            self,
            &cpu_trace,
            evaluation_accumulator.evaluation_mode(),
        );

        // Grab the Cuda accumulator's column and its slice of `random_coeff_powers`. This is the
        // SAME split as the SimdBackend/CpuBackend impls, guaranteeing identical coefficients.
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        // Run the audited CPU constraint evaluation, seeded with the current (device) accumulator
        // contents copied to host.
        let trace_cols = cpu_trace_cols.as_cols_ref().map_cols(
            |c: &Cow<'_, CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>| c.as_ref(),
        );
        let host_result: SecureColumnByCoords<CpuBackend> = accumulate_pointwise_cpu(
            self,
            trace_cols,
            eval_domain.log_size(),
            trace_domain.log_size(),
            denom_inv,
            &accum.random_coeff_powers,
            &accum.col.to_cpu(),
        );

        // Upload the host result back into the device accumulator column, coordinate by
        // coordinate, via `FromIterator<BaseField> for BaseFieldVec`.
        *accum.col = SecureColumnByCoords {
            columns: host_result.columns.map(|c| c.into_iter().collect()),
        };
    }
}
