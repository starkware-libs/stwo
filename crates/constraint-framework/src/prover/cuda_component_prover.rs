//! `ComponentProver<CudaBackend>` for `FrameworkComponent`.
//!
//! The device-resident `CudaBackend` stores committed columns as device `BaseFieldVec`s, so a
//! `mem::transmute` to `SimdBackend` is UNSOUND. Instead we perform a REAL conversion: each
//! committed column read by the eval is copied to host via `Column::to_cpu()`, we run the audited
//! constraint-quotient evaluation on host (seeded with the current accumulator contents), and
//! upload the result back into the device accumulator column. The D2H/H2D round trip is acceptable
//! because constraint eval is a small fraction of prove time vs commit/FRI.
//!
//! An OPT-IN, circuit-specific, device-resident GPU constraint kernel can be registered downstream
//! (see the branch in `evaluate_constraint_quotients_on_domain`); when present and enabled it runs
//! ahead of this host-delegate. Absent such a kernel, this audited host-delegate is the ONLY correct
//! path and the DEFAULT.
//!
//! # Why no GENERIC device-resident GPU constraint path exists here
//!
//! Two candidate generic GPU paths were rejected on soundness grounds:
//!
//! 1. A generic `GpuDomainEvaluator` against `CudaBackend` device columns is UNSOUND: its `off == 0`
//!    fast path calls `VeryPackedBaseColumn::transform_under_ref(col)`, which reinterprets *host*
//!    SIMD memory. A `CudaBackend` `BaseFieldVec` is a raw device pointer, so that transmute is
//!    undefined behavior; its non-zero-offset path would also issue one single-element D2H copy
//!    (`Column::at`) per masked value.
//!
//! 2. The NitrooZK device path (`stwo_cuda/cuda/evaluate_constraints.cu`, FFI-exposed as
//!    `bindings::evaluate_constraint_quotients_on_domain`) is NOT generic: it `switch`es on an
//!    `eval_id` (FNV-1a hash of a component name) into hand-written, per-component CUDA kernels
//!    (blake/poseidon/cairo opcodes/range-checks/wide_fibonacci). A downstream AIR whose eval is
//!    not in that switch would return `false` ("unsupported, fall back to CPU"). Enabling it for
//!    such an AIR would require authoring a NEW, soundness-critical, AIR-specific CUDA kernel by
//!    hand. There is no GENERIC `FrameworkEval -> CUDA` path in this tree.
//!
//! The env var `CUDA_CONSTRAINT_CPU_FALLBACK=1` forces this host-delegate even when a downstream GPU
//! kernel is registered.

use std::borrow::Cow;

use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::air::Component;
use stwo::core::fields::m31::BaseField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::column::{BaseColumn, VeryPackedSecureColumnByCoords};
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{ComponentProver, DomainEvaluationAccumulator, Poly, Trace};

use super::component_prover::{
    accumulate_pointwise_cpu, get_constraint_quotient_inputs, ConstraintQuotientInputs,
};
use super::cuda_constraint_kernel::{
    gpu_constraints_opt_in, registered_expected_kernel_guard, registered_gpu_constraint_kernel,
    GpuConstraintDispatch,
};
use super::SimdDomainEvaluator;
use crate::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

/// Whether the operator has forced the audited CPU-delegate constraint path via
/// `CUDA_CONSTRAINT_CPU_FALLBACK=1`. Read fresh on each call (cheap; once per component per prove).
/// When true, the opt-in GPU constraint branch is skipped and this component always host-delegates.
fn cpu_fallback_forced() -> bool {
    matches!(
        std::env::var("CUDA_CONSTRAINT_CPU_FALLBACK").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// PANIC (no silent fallback) when a component that a downstream plugin declared "expected on GPU"
/// falls to the host-delegate constraint path on `CudaBackend`. That path runs composition_eval
/// ~60x slower than the GPU kernel, so silently taking it turns a GPU prove into a benchmark of the
/// WRONG (slow) code — the exact regression that let a stale kernel be measured undetected. When a
/// plugin has registered its kernel it is REQUIRED for that plugin's MAIN component, so reaching
/// the host delegate for it is a hard error.
///
/// This backend is CIRCUIT-AGNOSTIC: it does NOT know which component is "big enough that a missing
/// kernel is a bug". That knowledge lives in a DOWNSTREAM plugin, which installs a predicate via
/// `set_expected_kernel_guard`. This function consults the registered guard:
///   * No guard registered (`None`) => a generic backend with no plugin => NEVER force-panic (a
///     host-delegate is always a sanctioned path). This is the default.
///   * A guard registered => panic iff the guard returns `true` for this component (its MAIN AIR)
///     AND the operator did not explicitly opt into a host run.
///
/// The plugin's guard is the sole owner of the fingerprint, separating the large many-constraint
/// MAIN AIR from every small fixed-size table component, which host-delegates normally.
///
/// ESCAPE HATCH: honored ONLY when the operator has EXPLICITLY forced the host path via
/// `CUDA_CONSTRAINT_CPU_FALLBACK=1`, or opted out of the GPU constraint path entirely
/// (`CUDA_GPU_CONSTRAINTS=0`); both are sanctioned host runs. This function is only reachable from
/// `ComponentProver<CudaBackend>`, so the CPU/SimdBackend build never calls it.
fn panic_if_main_host_delegate(n_constraints: usize, log_n_rows: u32) {
    // No downstream plugin registered an "expected-on-GPU" guard -> generic backend, never panic.
    let Some(guard) = registered_expected_kernel_guard() else {
        return;
    };
    if !guard(n_constraints, log_n_rows) {
        return;
    }
    // ESCAPE HATCH (see fn doc): the operator explicitly asked for the host path.
    if cpu_fallback_forced() || !gpu_constraints_opt_in() {
        return;
    }
    panic!(
        "A plugin-declared MAIN component (expected-kernel guard matched) fell to the audited \
         HOST-DELEGATE constraint path on CudaBackend (its fast GPU constraint kernel did NOT \
         engage) — composition_eval would run ~60x slower, benchmarking the WRONG path. The kernel \
         either was not registered (plugin `register()`), declined on a structural mismatch (the \
         kernel's own decline-guard constants stale vs the AIR), or its drawn relation was not \
         installed. Refusing to silently fall back. \
         (n_constraints={n_constraints}, log_n_rows={log_n_rows}). \
         To intentionally run the host delegate (e.g. the CPU-vs-GPU byte-identity diff), set \
         CUDA_CONSTRAINT_CPU_FALLBACK=1."
    );
}

/// Copies a single committed `Poly<CudaBackend>` to a host `Poly<CpuBackend>` by moving its
/// evaluations (and FFT-basis coefficients, when present) off the device via `to_cpu()`.
fn poly_to_cpu(poly: &Poly<CudaBackend>) -> Poly<CpuBackend> {
    let host_values = poly.evals.values.to_cpu();
    let evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
        poly.evals.domain,
        host_values,
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

        // Read once; gates the opt-in GPU branch below (when set, force the audited host-delegate).
        let _cpu_fallback_forced = cpu_fallback_forced();

        // OPT-IN device-resident GPU constraint kernel (circuit-specific, registered downstream).
        // Taken ONLY when `CUDA_GPU_CONSTRAINTS=1`, a kernel was installed via
        // `set_gpu_constraint_kernel`, and `CUDA_CONSTRAINT_CPU_FALLBACK` is NOT forcing the host
        // path. The kernel inspects the component (e.g. by structural fingerprint) and returns
        // `false` for any component that is not its target AIR, in which case we fall through to
        // the audited host-delegate below. This is a SEPARATE, additive branch IN FRONT of the
        // host delegate; the host-delegate code below is unchanged and remains the DEFAULT + the
        // fallback. No host constraint-eval math is affected by this branch.
        if gpu_constraints_opt_in() && !_cpu_fallback_forced {
            if let Some(kernel) = registered_gpu_constraint_kernel() {
                // The committed eval columns are resident on device; dispatch the GPU kernel on the
                // committed trace directly. Device-resident constraint-quotient inputs (no D2H).
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
                // Kernel declined (not its target AIR) -> fall through to the audited host
                // delegate.
            }
        }

        // Reaching here means this component takes the audited host-delegate. Normal for small table
        // components; a hard error for a plugin-declared MAIN component (see fn doc for the rationale).
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
        // evaluation mode (the mode is backend-agnostic). The eval-domain extension / subdomain
        // borrow (`get_trace_columns` inside) stays on `CpuBackend`; only the pointwise evaluation
        // over those columns is moved to the SIMD (packed + rayon-parallel) evaluator below.
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

        // Copy the current (device) accumulator contents to host once; used to seed BOTH the SIMD
        // and the small-domain scalar fallback so the fold is `existing + new`, identical to the
        // old scalar path.
        let seed_cpu: SecureColumnByCoords<CpuBackend> = accum.col.to_cpu();

        // SIMD (packed + rayon-parallel) pointwise constraint evaluation.
        //
        // This REPLACES the serial scalar `accumulate_pointwise_cpu` over the eval domain. It is a
        // faithful transplant of the audited `ComponentProver<SimdBackend>` body: the same
        // small-domain scalar fallback, the same `SimdDomainEvaluator` per packed row, and the same
        // `chunk.packed_at() + row_res * row_denom_inv` fold, driven by the same
        // `random_coeff_powers` slice split off the Cuda accumulator above.
        //
        // We inline the body (rather than call `ComponentProver<SimdBackend>::…`) because that impl
        // requires a `DomainEvaluationAccumulator<SimdBackend>`, whose only public constructor
        // (`::new`) needs the base `random_coeff` — which is not available here (we hold only the
        // already-derived, split powers). Reconstructing it would require a shared-crate API change.
        //
        // Rows are independent (per-row write, no reduction) so the rayon order is irrelevant. The
        // host->SIMD repack (`BaseColumn::from_cpu` / `SecureColumnByCoords::<SimdBackend>::from_cpu`)
        // is positional (logical index `i` -> packed `i / N_LANES`, lane `i % N_LANES`) and
        // order-preserving, so the eval-domain trace columns and the seed carry the same logical
        // bit-reversed order as the `CpuBackend` values.

        // Repack the prepared eval-domain trace columns (still `CpuBackend`) into `SimdBackend`
        // columns, preserving the global TreeVec shape. Owned storage; the SIMD evaluator borrows
        // from it.
        let simd_trace_cols: TreeVec<
            Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        > = cpu_trace_cols.as_cols_ref().map_cols(
            |c: &Cow<'_, CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>| {
                let c = c.as_ref();
                CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                    c.domain,
                    BaseColumn::from_cpu(&c.values),
                )
            },
        );

        // Small-domain scalar fallback, mirroring `component_prover.rs`: below the packed threshold
        // the SIMD impl itself drops to the SAME scalar `accumulate_pointwise_cpu`, so we do too.
        // This keeps tiny components (program/boundary) on the exact scalar path as before.
        if trace_domain.log_size() < LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS {
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
                &seed_cpu,
            );
            // Upload the host result back into the device accumulator column, coordinate by
            // coordinate, via `FromIterator<BaseField> for BaseFieldVec`.
            *accum.col = SecureColumnByCoords {
                columns: host_result.columns.map(|c| c.into_iter().collect()),
            };
            return;
        }

        // Seed a SIMD accumulator column with the current (device) accumulator contents. The SIMD
        // loop folds `+= row_res * denom_inv` INTO this column, producing `existing + new` exactly
        // like the scalar path (which seeds `accumulate_pointwise_cpu` with `accum.col.to_cpu()`).
        let mut simd_col: SecureColumnByCoords<SimdBackend> =
            SecureColumnByCoords::<SimdBackend>::from_cpu(seed_cpu);

        {
            let col = unsafe { VeryPackedSecureColumnByCoords::transform_under_mut(&mut simd_col) };

            let range = 0..(1 << (eval_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS));

            #[cfg(not(feature = "parallel"))]
            let iter = range.zip(col.chunks_mut(1));

            #[cfg(feature = "parallel")]
            let iter = range.into_par_iter().zip(col.par_chunks_mut(1));

            // Define any `self` values outside the loop to prevent the compiler thinking there is a
            // `Sync` requirement on `Self`.
            let self_eval = &self.eval;
            let self_claimed_sum = self.claimed_sum();

            iter.for_each(|(vec_row, mut chunk)| {
                let trace_cols = simd_trace_cols.as_cols_ref();

                // Evaluate constraints at row.
                let eval = SimdDomainEvaluator::new(
                    &trace_cols,
                    vec_row,
                    &accum.random_coeff_powers,
                    trace_domain.log_size(),
                    eval_domain.log_size(),
                    self_eval.log_size(),
                    self_claimed_sum,
                );
                let row_res = self_eval.evaluate(eval).row_res;

                // Finalize row.
                unsafe {
                    let row_denom_inv = VeryPackedBaseField::broadcast(
                        denom_inv[vec_row
                            >> (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)],
                    );
                    chunk.set_packed(0, chunk.packed_at(0) + row_res * row_denom_inv);
                }
            });
        }

        // Upload the SIMD result back into the device accumulator column. `simd_col.to_cpu()`
        // recovers the flat per-coordinate `Vec<BaseField>` in the SAME logical order, then each is
        // collected into a device `BaseFieldVec` via `FromIterator<BaseField>` — identical to the
        // scalar path's upload.
        let host_result = simd_col.to_cpu();
        *accum.col = SecureColumnByCoords {
            columns: host_result.columns.map(|c| c.into_iter().collect()),
        };
    }
}
