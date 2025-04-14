// Note: depends on qm31.wgsl, fraction.wgsl, utils.wgsl
// Define constants
const N_ROWS: u32 = 256;
const N_EXTENDED_ROWS: u32 = N_ROWS * 4;
const N_STATE: u32 = 16;
const N_INSTANCES_PER_ROW: u32 = 8;
const N_COLUMNS: u32 = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const N_INTERACTION_COLUMNS: u32 = N_INSTANCES_PER_ROW * 4;
const N_HALF_FULL_ROUNDS: u32 = 4;
const FULL_ROUNDS: u32 = 2u * N_HALF_FULL_ROUNDS;
const N_PARTIAL_ROUNDS: u32 = 14;
const N_LANES: u32 = 16;
const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const LOG_N_LANES: u32 = 4;
const N_WORKGROUPS: u32 = N_EXTENDED_ROWS * N_LANES / THREADS_PER_WORKGROUP;
const THREADS_PER_WORKGROUP: u32 = 256;
const N_CONSTRAINTS: u32 = 1144;
const R: CM31 = CM31(M31(2u), M31(1u));
const ONE = QM31(CM31(M31(1u), M31(0u)), CM31(M31(0u), M31(0u)));
const DUMMY: u32 = 1004;
const N_ORIGINAL_COLUMN_SIZE: u32 = N_LANES * N_ROWS;
const N_EXTENDED_COLUMN_SIZE: u32 = N_LANES * N_EXTENDED_ROWS;

const N_LINE_TWIDDLES_SIZE: u32 = N_EXTENDED_ROWS * N_LANES;
const N_LINE_TWIDDLES_FLAT_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
const N_CIRCLE_TWIDDLES_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
const N_ORIGINAL_TRACE_COLUMNS: u32 = 1 + N_COLUMNS + N_INTERACTION_COLUMNS;

fn butterfly(v0: ptr<function, M31>, v1: ptr<function, M31>, twid: M31) {
    let tmp = m31_mul(*v1, twid);
    *v1 = m31_sub(*v0, tmp);
    *v0 = m31_add(*v0, tmp);
}

struct BaseColumn {
    data: array<array<M31, N_LANES>, N_EXTENDED_ROWS>,
}

struct OriginalColumn {
    data: array<M31, N_ORIGINAL_COLUMN_SIZE>,
}

struct Extended1DColumn {
    data: array<M31, N_EXTENDED_COLUMN_SIZE>,
}

struct LookupElements {
    z: QM31,
    alpha: QM31,
    alpha_powers: array<QM31, N_STATE>,
}

struct ComputeCompositionPolynomialInput {
    original_trace: array<OriginalColumn, N_ORIGINAL_TRACE_COLUMNS>,
    twiddles: Twiddles,
    denom_inv: array<M31, 4>,
    random_coeff_powers: array<QM31, N_CONSTRAINTS>,
    lookup_elements: LookupElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: QM31,
}

struct ComputeCompositionPolynomialOutput {
    poly: array<array<QM31, N_LANES>, N_EXTENDED_ROWS>,
}

struct Twiddles {
    circle_twiddles: array<M31, N_CIRCLE_TWIDDLES_SIZE>,
    circle_twiddles_size: u32,
    line_twiddles_flat: array<M31, N_LINE_TWIDDLES_FLAT_SIZE>,
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: array<u32, N_LINE_TWIDDLES_SIZE>,
    line_twiddles_offsets: array<u32, N_LINE_TWIDDLES_SIZE>,
}

struct ExtendTraceOutput {
    extended_trace: array<Extended1DColumn, N_ORIGINAL_TRACE_COLUMNS>,
}

@group(0) @binding(0)
var<storage, read> trace_input: ComputeCompositionPolynomialInput;

@group(0) @binding(1)
var<storage, read_write> composition_polynomial_output: ComputeCompositionPolynomialOutput;

@group(0) @binding(2)
var<storage, read_write> trace_output: ExtendTraceOutput;

@compute @workgroup_size(256)
fn evaluate_line_twiddle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_size = 256u;
    let y_dim_size = 256u;

    let size = N_EXTENDED_COLUMN_SIZE;
    let thread_id_x = global_id.x;
    let original_column_size = N_ORIGINAL_COLUMN_SIZE;
    let copy_chunk_size = (original_column_size + thread_size - 1u) / thread_size;
    let copy_chunk_start = thread_id_x * copy_chunk_size;
    let copy_chunk_end = min(copy_chunk_start + copy_chunk_size, original_column_size);

    let thread_id_y = global_id.y;
    let polynomial_chunk_size = (N_ORIGINAL_TRACE_COLUMNS + y_dim_size) / y_dim_size;
    let polynomial_start = polynomial_chunk_size * thread_id_y;
    let polynomial_end = min(polynomial_start + polynomial_chunk_size, N_ORIGINAL_TRACE_COLUMNS);

    // copy input.coeffs to trace_output.evals
    for (var polynomial_id = polynomial_start; polynomial_id < polynomial_end; polynomial_id = polynomial_id + 1u) {
        for (var j = copy_chunk_start; j < copy_chunk_end; j = j + 1u) {
            trace_output.extended_trace[polynomial_id].data[j] = trace_input.original_trace[polynomial_id].data[j];
        }
    }

    workgroupBarrier();

    // Process line_twiddles
    var layer = trace_input.twiddles.line_twiddles_layer_count - 1u;
    loop {
        let layer_size = trace_input.twiddles.line_twiddles_sizes[layer];
        let layer_offset = trace_input.twiddles.line_twiddles_offsets[layer];
        let step = 1u << (layer + 1u);
        
        for (var h = 0u; h < layer_size; h = h + 1u) {
            let t = trace_input.twiddles.line_twiddles_flat[layer_offset + h];
            let idx0_offset = (h << (layer + 2u));

            for (var l = thread_id_x; l < step; l = l + thread_size) {
                let idx0 = idx0_offset + l;
                let idx1 = idx0 + step;

                for (var polynomial_id = polynomial_start; polynomial_id < polynomial_end; polynomial_id = polynomial_id + 1u) {
                    var val0 = trace_output.extended_trace[polynomial_id].data[idx0];
                    var val1 = trace_output.extended_trace[polynomial_id].data[idx1];
                
                    butterfly(&val0, &val1, t);
                    
                    trace_output.extended_trace[polynomial_id].data[idx0] = val0;
                    trace_output.extended_trace[polynomial_id].data[idx1] = val1;
                }
            }

            workgroupBarrier();
        }

        if (layer == 0u) { break; }  
        layer = layer - 1u;
    }
}

@compute @workgroup_size(256)
fn evaluate_circle_twiddle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_size = 256u;
    let y_dim_size = 256u;
    let size = N_EXTENDED_COLUMN_SIZE / 2;

    let thread_id_x = global_id.x;
    let chunk_size = (size + workgroup_size - 1u) / workgroup_size;
    let chunk_start = thread_id_x * chunk_size;
    let chunk_end = min(chunk_start + chunk_size, size);

    let thread_id_y = global_id.y;
    let polynomial_chunk_size = (N_ORIGINAL_TRACE_COLUMNS + y_dim_size) / y_dim_size;
    let polynomial_start = polynomial_chunk_size * thread_id_y;
    let polynomial_end = min(polynomial_start + polynomial_chunk_size, N_ORIGINAL_TRACE_COLUMNS);

    // store_debug_value(thread_id, global_id.y);
    for (var i = chunk_start; i < chunk_end; i = i + 1u) {
        let idx0 = i << 1u;
        let idx1 = idx0 + 1u;

        for (var polynomial_id = polynomial_start; polynomial_id < polynomial_end; polynomial_id = polynomial_id + 1u) {
            var val0 = trace_output.extended_trace[polynomial_id].data[idx0];
            var val1 = trace_output.extended_trace[polynomial_id].data[idx1];

            butterfly(&val0, &val1, trace_input.twiddles.circle_twiddles[i]);

            trace_output.extended_trace[polynomial_id].data[idx0] = val0;
            trace_output.extended_trace[polynomial_id].data[idx1] = val1;
        }
    }
}
