// Note: depends on qm31.wgsl, utils.wgsl constants.wgsl

fn butterfly(v0: ptr<function, M31>, v1: ptr<function, M31>, twid: M31) {
    let tmp = m31_mul(*v1, twid);
    * v1 = m31_sub(*v0, tmp);
    * v0 = m31_add(*v0, tmp);
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

//------------------------------------------------------------------------------
//  evaluate_line_twiddle_per_poly32
//
//  One work-item (thread) processes **one polynomial column**.
//
//  Phase 1 – “large layers”            : operate directly in device storage
//  Phase 2 – “small layers + circle”   : copy 256-u32 chunks into a
//                                        thread-local scratch array,
//                                        finish the remaining line-twiddle
//                                        layers *and* the final circle-twiddle
//                                        (step = 1), then write the chunk back.
//------------------------------------------------------------------------------

// Size of a thread-local scratch tile (256 u32 = 1 KiB).
// 32 threads × 1 KiB ≈ 32 KiB, matching Metal's per-threadgroup LDS budget.
const CHUNK_SIZE: u32 = 128u;

@compute @workgroup_size(16)
fn evaluate_line_twiddle_per_poly32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let poly_id = global_id.x + global_id.y * 16u;
    if (poly_id >= N_ORIGINAL_TRACE_COLUMNS) {
        return;
    }

    //------------------------------------------------------------------------
    // 1. Copy coeffs → output evals (storage → storage, 1 : 1)
    //------------------------------------------------------------------------
    for (var j: u32 = 0u; j < N_ORIGINAL_COLUMN_SIZE; j = j + 1u) {
        trace_output.extended_trace[poly_id].data[j] = trace_input.original_trace[poly_id].data[j];
    }

    //------------------------------------------------------------------------
    // 2. Line-twiddle “large layers” – operate in place in storage
    //    Stop when a single butterfly block (step*2) fits in CHUNK_SIZE.
    //------------------------------------------------------------------------
    let num_layers = trace_input.twiddles.line_twiddles_layer_count;
    var layer = num_layers - 1u;
    loop {
        let step = 1u << (layer + 1u);
        if (step * 2u <= CHUNK_SIZE) {
            break;
        }
        let layer_size = trace_input.twiddles.line_twiddles_sizes[layer];
        let layer_offset = trace_input.twiddles.line_twiddles_offsets[layer];

        // Iterate over all butterfly blocks in this layer
        for (var h: u32 = 0u; h < layer_size; h = h + 1u) {
            let t = trace_input.twiddles.line_twiddles_flat[layer_offset + h];
            let base_idx = h << (layer + 2u);

            // Plain Cooley–Tukey butterfly within the block
            for (var l: u32 = 0u; l < step; l = l + 1u) {
                let idx0 = base_idx + l;
                let idx1 = idx0 + step;

                var v0 = trace_output.extended_trace[poly_id].data[idx0];
                var v1 = trace_output.extended_trace[poly_id].data[idx1];
                butterfly(&v0, &v1, t);
                trace_output.extended_trace[poly_id].data[idx0] = v0;
                trace_output.extended_trace[poly_id].data[idx1] = v1;
            }
        }
        if (layer == 0u) {
            break;
        }
        layer = layer - 1u;
    }

    //------------------------------------------------------------------------
    // 3. Scratch-tile phase – finish remaining line layers + circle twiddle
    //------------------------------------------------------------------------
    let num_chunks = (N_EXTENDED_COLUMN_SIZE + CHUNK_SIZE - 1u) / CHUNK_SIZE;
    var scratch: array<M31, CHUNK_SIZE>;
    for (var chunk_id: u32 = 0u; chunk_id < num_chunks; chunk_id = chunk_id + 1u) {
        let base = chunk_id * CHUNK_SIZE;
        let real_size = min(CHUNK_SIZE, N_EXTENDED_COLUMN_SIZE - base);

        //--------------------------------------------------------------------
        // 3-A. Copy current chunk from storage → scratch
        //--------------------------------------------------------------------
        for (var i: u32 = 0u; i < real_size; i = i + 1u) {
            scratch[i] = trace_output.extended_trace[poly_id].data[base + i];
        }

        //--------------------------------------------------------------------
        // 3-B. Remaining (small) line-twiddle layers in scratch
        //--------------------------------------------------------------------
        var l = layer;
        loop {
            let step = 1u << (l + 1u);
            let layer_size = trace_input.twiddles.line_twiddles_sizes[l];
            let layer_offset = trace_input.twiddles.line_twiddles_offsets[l];

            let h_start = base >> (l + 2u);
            let h_end = (base + real_size - 1u) >> (l + 2u);
            let h_count = h_end - h_start + 1u;

            for (var h_local: u32 = 0u; h_local < h_count; h_local = h_local + 1u) {
                let h_global = h_start + h_local;
                if (h_global >= layer_size) {
                    continue;
                }
                let t = trace_input.twiddles.line_twiddles_flat[layer_offset + h_global];

                // Convert global base index → scratch-local index
                let base_idx_local = (h_global << (l + 2u)) - base;

                for (var s: u32 = 0u; s < step; s = s + 1u) {
                    let idx0 = base_idx_local + s;
                    let idx1 = idx0 + step;
                    var v0 = scratch[idx0];
                    var v1 = scratch[idx1];
                    butterfly(&v0, &v1, t);
                    scratch[idx0] = v0;
                    scratch[idx1] = v1;
                }
            }
            if (l == 0u) {
                break;
            }
            l = l - 1u;
        }

        //--------------------------------------------------------------------
        // 3-C. Circle-twiddle (step = 1) in scratch
        //--------------------------------------------------------------------
        // Treat (idx0, idx1) as even/odd pair; global pair index = (base+idx0)/2
        for (var idx0: u32 = 0u; idx0 + 1u < real_size; idx0 = idx0 + 2u) {
            let idx1 = idx0 + 1u;
            let pair_global = (base + idx0) >> 1u;
            let t = trace_input.twiddles.circle_twiddles[pair_global];

            var v0 = scratch[idx0];
            var v1 = scratch[idx1];
            butterfly(&v0, &v1, t);
            scratch[idx0] = v0;
            scratch[idx1] = v1;
        }

        //--------------------------------------------------------------------
        // 3-D. Copy scratch → storage
        //--------------------------------------------------------------------
        for (var i: u32 = 0u; i < real_size; i = i + 1u) {
            trace_output.extended_trace[poly_id].data[base + i] = scratch[i];
        }
    }
}
