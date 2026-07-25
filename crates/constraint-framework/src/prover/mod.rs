mod assert;
mod component_prover;
mod cpu_domain;
mod logup;
pub mod relation_tracker;
mod simd_domain;

// Device-resident CUDA backend support. `cuda_component_prover` provides
// `ComponentProver<CudaBackend>` (audited host-delegate by default), and `cuda_constraint_kernel`
// is the circuit-agnostic registration hook a downstream crate uses to install a per-AIR GPU
// constraint kernel. The kernel itself lives downstream — this crate stays circuit-agnostic.
#[cfg(feature = "cuda")]
mod cuda_component_prover;
#[cfg(feature = "cuda")]
mod cuda_constraint_kernel;

pub use assert::{AssertEvaluator, assert_constraints_on_polys, assert_constraints_on_trace};
// Public surface for a downstream GPU constraint kernel: the registration hook + the
// device-resident constraint-quotient inputs the generic prover builds for it.
#[cfg(feature = "cuda")]
pub use component_prover::{ConstraintQuotientInputs, get_constraint_quotient_inputs};
pub use cpu_domain::CpuDomainEvaluator;
#[cfg(feature = "cuda")]
pub use cuda_constraint_kernel::{
    ExpectedKernelGuard, GpuConstraintDispatch, GpuConstraintKernel, set_expected_kernel_guard,
    set_gpu_constraint_kernel,
};
pub use logup::{FractionWriter, LogupColGenerator, LogupTraceGenerator};
pub use simd_domain::SimdDomainEvaluator;
