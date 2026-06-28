//! Generic registration hook for a circuit-specific, device-resident GPU constraint kernel.
//!
//! The generic `ComponentProver<CudaBackend>` (see [`super::cuda_component_prover`]) takes the
//! audited host-delegate by default. NitrooZK's CUDA constraint path is NOT generic — there is no
//! `FrameworkEval -> CUDA` lowering, so every AIR needs a hand-written, soundness-critical CUDA
//! kernel. To keep the `stwo` crate circuit-agnostic, that per-AIR kernel (and its structural
//! fingerprint) lives entirely in a DOWNSTREAM crate, which installs it here via
//! [`set_gpu_constraint_kernel`].
//!
//! When `CUDA_GPU_CONSTRAINTS=1` AND a kernel is installed, the generic prover offers each
//! component to the kernel first; the kernel returns `false` for any component that is not its
//! target AIR, and the prover falls back to the audited host-delegate. No host constraint-eval
//! math is affected, and the default (no kernel installed / env unset) is unchanged.

use std::sync::Mutex;

use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::DomainEvaluationAccumulator;

use super::component_prover::ConstraintQuotientInputs;

/// Everything a downstream GPU constraint kernel needs to evaluate one component's constraint
/// quotients on-device and accumulate into the (device) composition-polynomial column.
///
/// All trace data in `inputs` is DEVICE-RESIDENT (no D2H): `inputs.trace` holds
/// `Cow<CircleEvaluation<CudaBackend>>` whose `.values` are device `BaseFieldVec`s, built by the
/// generic prover via the (public) `get_constraint_quotient_inputs`.
pub struct GpuConstraintDispatch<'a, 'b> {
    /// Device-resident eval-domain trace columns + denominator inverses.
    pub inputs: &'a ConstraintQuotientInputs<'b, CudaBackend>,
    /// Number of constraints of this component (== accumulator column width to request).
    pub n_constraints: usize,
    /// The component's claimed LogUp sum (used for the cumulative-sum shift).
    pub claimed_sum: SecureField,
    /// log2 of the number of real (pre-extension) trace rows.
    pub log_n_rows: u32,
    /// The device accumulator to seed-and-accumulate into.
    pub accumulator: &'a mut DomainEvaluationAccumulator<CudaBackend>,
}

/// A downstream GPU constraint kernel. Returns `true` if it handled the component (the device
/// accumulator was written), `false` if the component is not its target AIR — in which case the
/// generic prover falls back to the audited host-delegate. MUST NOT change host constraint math.
pub type GpuConstraintKernel = for<'a, 'b> fn(GpuConstraintDispatch<'a, 'b>) -> bool;

static GPU_CONSTRAINT_KERNEL: Mutex<Option<GpuConstraintKernel>> = Mutex::new(None);

/// Install the process-wide GPU constraint kernel. Call once before proving; later calls overwrite.
pub fn set_gpu_constraint_kernel(kernel: GpuConstraintKernel) {
    *GPU_CONSTRAINT_KERNEL.lock().unwrap() = Some(kernel);
}

/// The currently-registered GPU constraint kernel, if any.
pub(crate) fn registered_gpu_constraint_kernel() -> Option<GpuConstraintKernel> {
    *GPU_CONSTRAINT_KERNEL.lock().unwrap()
}

/// Whether the operator opted into the device-resident GPU constraint path
/// (`CUDA_GPU_CONSTRAINTS=1`). Default (unset / "0") keeps the audited host-delegate.
pub fn gpu_constraints_opt_in() -> bool {
    matches!(
        std::env::var("CUDA_GPU_CONSTRAINTS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}
