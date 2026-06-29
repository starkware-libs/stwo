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
//!    already produced WRONG values on real gate_air even on the host-layout `GpuBackend` — see
//!    the note in `gpu_component_prover.rs` — so it is not a trustworthy reference to port.)
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
use crate::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

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

/// Builds a host copy of `device_polys` that preserves the EXACT global [`TreeVec`] shape (same
/// number of trees, same number of columns per tree, in the same positions) but only performs a
/// device->host copy ([`poly_to_cpu`]) for the columns THIS component actually consumes.
///
/// The audited host eval ([`get_constraint_quotient_inputs`]) reads exactly:
///   * every column in each [`stwo::core::pcs::TreeSubspan`] of `trace_locations` (the
///     `col_start..col_end` range within `tree_index`), via `trace.polys.sub_tree(trace_locations)`,
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
                // Device-resident constraint-quotient inputs (no D2H) on the CudaBackend trace.
                let inputs = get_constraint_quotient_inputs(
                    self,
                    trace,
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
                // Kernel declined (not its target AIR) -> fall through to the audited host delegate.
            }
        }

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
        } = get_constraint_quotient_inputs(self, &cpu_trace, evaluation_accumulator.evaluation_mode());

        // Grab the Cuda accumulator's column and its slice of `random_coeff_powers`. This is the
        // SAME split as the SimdBackend/CpuBackend impls, guaranteeing identical coefficients.
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        // Run the audited CPU constraint evaluation, seeded with the current (device) accumulator
        // contents copied to host.
        let trace_cols = cpu_trace_cols
            .as_cols_ref()
            .map_cols(|c: &Cow<'_, CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>| {
                c.as_ref()
            });
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
