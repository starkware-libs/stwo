use stwo_prover::core::backend::web::webgpu::qm31::{GpuM31, GpuQM31};

use super::constants::*;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuOriginalColumn {
    pub coeffs: [GpuM31; (N_LANES * N_ORIGINAL_ROWS) as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuExtendedColumn {
    pub data: [GpuM31; (N_LANES * N_EXTENDED_ROWS) as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Twiddles {
    pub circle_twiddles: [GpuM31; N_CIRCLE_TWIDDLES_SIZE as usize],
    pub circle_twiddles_size: u32,
    pub line_twiddles_flat: [GpuM31; N_LINE_TWIDDLES_FLAT_SIZE as usize],
    pub line_twiddles_layer_count: u32,
    pub line_twiddles_sizes: [u32; N_LINE_TWIDDLES_SIZE as usize],
    pub line_twiddles_offsets: [u32; N_LINE_TWIDDLES_SIZE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuLookupElements {
    pub z: GpuQM31,
    pub alpha: GpuQM31,
    pub alpha_powers: [GpuQM31; N_STATE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct ComputeCompositionPolynomialInput {
    pub original_trace: [GpuOriginalColumn; N_ORIGINAL_TRACE_COLUMNS as usize],
    pub twiddles: Twiddles,
    pub denom_inv: [GpuM31; 4],
    pub random_coeff_powers: [GpuQM31; N_CONSTRAINTS as usize],
    pub lookup_elements: GpuLookupElements,
    pub trace_domain_log_size: u32,
    pub eval_domain_log_size: u32,
    pub cumsum_shift: GpuQM31,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct ComputeCompositionPolynomialOutput {
    pub poly: [[GpuQM31; N_LANES as usize]; N_EXTENDED_ROWS as usize],
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ExtendTraceOutput {
    pub extended_trace: [GpuExtendedColumn; N_ORIGINAL_TRACE_COLUMNS as usize],
}
