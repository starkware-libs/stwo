// Note: depends on qm31.wgsl, utils.wgsl

// Initialize EXTERNAL_ROUND_CONSTS with explicit values
const EXTERNAL_ROUND_CONSTS: array<array<u32, N_STATE>, FULL_ROUNDS> = array<array<u32, N_STATE>, FULL_ROUNDS>(
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
);

// Initialize INTERNAL_ROUND_CONSTS with explicit values
const INTERNAL_ROUND_CONSTS: array<u32, N_PARTIAL_ROUNDS> = array<u32, N_PARTIAL_ROUNDS>(
    1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234
);

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

struct Twiddles {
    circle_twiddles: array<M31, N_CIRCLE_TWIDDLES_SIZE>,
    circle_twiddles_size: u32,
    line_twiddles_flat: array<M31, N_LINE_TWIDDLES_FLAT_SIZE>,
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: array<u32, N_LINE_TWIDDLES_SIZE>,
    line_twiddles_offsets: array<u32, N_LINE_TWIDDLES_SIZE>,
}

struct ComputeCompositionPolynomialInput {
    original_trace: array<OriginalColumn, N_ORIGINAL_TRACE_COLUMNS>,
    twiddles: Twiddles,
    denom_inv: array<M31, 4>,
    random_coeff_powers: array<QM31, N_CONSTRAINTS>,
    lookup_elements: LookupElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    cumsum_shift: QM31,
}

struct ComputeCompositionPolynomialOutput {
    poly: array<array<QM31, N_LANES>, N_EXTENDED_ROWS>,
}

struct RelationEntry {
    multiplicity: QM31,
    values: array<M31, N_STATE>,
}

struct ExtendTraceOutput {
    extended_trace: array<Extended1DColumn, N_ORIGINAL_TRACE_COLUMNS>,
}

struct State16 { 
    data: array<M31, N_STATE> 
}

@group(0) @binding(0)
var<storage, read> input: ComputeCompositionPolynomialInput;

@group(0) @binding(1)
var<storage, read_write> output: ComputeCompositionPolynomialOutput;

@group(0) @binding(2)
var<storage, read_write> extend_trace_output: ExtendTraceOutput;

var<private> constraint_index: u32 = 0u;

var<private> fracs_index: u32 = 0u;

var<private> fracs: array<Fraction, N_TOTAL_FRACS> = array<Fraction, N_TOTAL_FRACS>(
    ZERO_FRACTION, ZERO_FRACTION, ZERO_FRACTION, ZERO_FRACTION,
    ZERO_FRACTION, ZERO_FRACTION, ZERO_FRACTION, ZERO_FRACTION,
    ZERO_FRACTION, ZERO_FRACTION, ZERO_FRACTION, ZERO_FRACTION,
    ZERO_FRACTION, ZERO_FRACTION, ZERO_FRACTION, ZERO_FRACTION
);

var<private> is_finalized: bool = false;

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn compute_composition_polynomial(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let workgroup_index =  
        workgroup_id.x +
        workgroup_id.y * num_workgroups.x +
        workgroup_id.z * num_workgroups.x * num_workgroups.y;

    let global_invocation_index = workgroup_index * THREADS_PER_WORKGROUP + local_invocation_index; // [0, 512)

    var vec_index = global_invocation_index / N_LANES;
    var inner_vec_index = global_invocation_index % N_LANES;
    var col_index = 0u;

    for (var rep_i = 0u; rep_i < N_INSTANCES_PER_ROW; rep_i++) {
        var state: State16 = State16(array<M31, N_STATE>());
        for (var j = 0u; j < N_STATE; j++) {
            state.data[j] = next_trace_mask(col_index, vec_index, inner_vec_index);
            col_index += 1u;
        }
        var initial_state = state;

        // 4 full rounds
        for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
            for (var j = 0u; j < N_STATE; j++) {
                state.data[j] = m31_add(state.data[j], M31(EXTERNAL_ROUND_CONSTS[i][j]));
            }
            state = apply_external_round_matrix_state16(state);
            for (var j = 0u; j < N_STATE; j++) {
                state.data[j] = m31_pow5(state.data[j]);
            }
            for (var j = 0u; j < N_STATE; j++) {
                var m_1 = next_trace_mask(col_index, vec_index, inner_vec_index);
                let constraint = m31_sub(state.data[j], m_1);
                add_constraint(constraint, vec_index, inner_vec_index);

                state.data[j] = m_1;
                col_index += 1u;
            }
        }
        // Partial rounds
        for (var i = 0u; i < N_PARTIAL_ROUNDS; i++) {
            state.data[0] = m31_add(state.data[0], M31(INTERNAL_ROUND_CONSTS[i]));
            state = apply_internal_round_matrix_state16(state);
            state.data[0] = m31_pow5(state.data[0]);
            var m_1 = next_trace_mask(col_index, vec_index, inner_vec_index);
            let constraint = m31_sub(state.data[0], m_1);
            add_constraint(constraint, vec_index, inner_vec_index);

            state.data[0] = m_1;
            col_index += 1u;
        }
        // 4 full rounds
        for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
            for (var j = 0u; j < N_STATE; j++) {
                state.data[j] = m31_add(state.data[j], M31(EXTERNAL_ROUND_CONSTS[i + N_HALF_FULL_ROUNDS][j]));
            }
            state = apply_external_round_matrix_state16(state);
            for (var j = 0u; j < N_STATE; j++) {
                state.data[j] = m31_pow5(state.data[j]);
            }
            for (var j = 0u; j < N_STATE; j++) {
                var m_1 = next_trace_mask(col_index, vec_index, inner_vec_index);
                let constraint = m31_sub(state.data[j], m_1);
                add_constraint(constraint, vec_index, inner_vec_index);
                state.data[j] = m_1;
                col_index += 1u;
            }
        }
        add_to_relation_single(RelationEntry(ONE, initial_state.data));
        add_to_relation_single(RelationEntry(qm31_neg(ONE), state.data));
    }
    finalize_logup_in_pairs(vec_index, inner_vec_index);

    let row = vec_index * N_STATE + inner_vec_index;
    let denom_inv = input.denom_inv[row >> input.trace_domain_log_size];
    output.poly[vec_index][inner_vec_index] = qm31_mul(output.poly[vec_index][inner_vec_index], QM31(CM31(denom_inv, M31(0u)), CM31(M31(0u), M31(0u))));
}

fn flatten_idx(vec_index: u32, inner_vec_index: u32) -> u32 {
    return vec_index * N_LANES + inner_vec_index;
}

fn add_constraint(constraint: M31, vec_index: u32, inner_vec_index: u32) {
    add_constraint_qm31(QM31(CM31(constraint, M31(0u)), CM31(M31(0u), M31(0u))), vec_index, inner_vec_index);
}

fn add_constraint_qm31(constraint: QM31, vec_index: u32, inner_vec_index: u32) {
    var new_add = qm31_mul(constraint, input.random_coeff_powers[constraint_index]);
    output.poly[vec_index][inner_vec_index] = qm31_add(output.poly[vec_index][inner_vec_index], new_add);
    constraint_index += 1u;
}

fn add_to_relation_single(entry: RelationEntry) {
    var combined_value = QM31(CM31(M31(0u), M31(0u)), CM31(M31(0u), M31(0u)));
    for (var j = 0u; j < N_STATE; j++) {
        let value = QM31(CM31(entry.values[j], M31(0u)), CM31(M31(0u), M31(0u)));
        combined_value = qm31_add(combined_value, qm31_mul(input.lookup_elements.alpha_powers[j], value));
    }

    combined_value = qm31_sub(combined_value, input.lookup_elements.z);
    var frac = Fraction(entry.multiplicity, combined_value);
    write_logup_frac_single(frac);
}

fn write_logup_frac_single(frac: Fraction) {
    if (fracs_index == 0u) {
        is_finalized = false;
    }
    fracs[fracs_index] = frac;
    fracs_index += 1u;
}

fn finalize_logup_in_pairs(vec_index: u32, inner_vec_index: u32) {
    if (is_finalized) {
        return;
    }

    var prev_col_cumsum = QM31(CM31(M31(0u), M31(0u)), CM31(M31(0u), M31(0u)));
    var last_interaction_col_index = 0u;

    // All batches except the last are cumulatively summed in new interaction columns.
    for (var i = 0u; i < fracs_index - 2u; i += 2u) {
        var cur_frac = fraction_add(fracs[i], fracs[i + 1u]);

        var cur_cumsum = next_interaction_trace_mask(last_interaction_col_index, vec_index, inner_vec_index);
        var diff = qm31_sub(cur_cumsum, prev_col_cumsum);
        prev_col_cumsum = cur_cumsum;
        var constraint = qm31_sub(qm31_mul(diff, cur_frac.denominator), cur_frac.numerator);
        add_constraint_qm31(constraint, vec_index, inner_vec_index);
        last_interaction_col_index += 4u;
    }

    // last batch
    let frac = fraction_add(fracs[fracs_index - 2u], fracs[fracs_index - 1u]);
    
    var cur_cumsum = next_interaction_trace_mask(last_interaction_col_index, vec_index, inner_vec_index);
    var prev_row_cumsum = next_interaction_trace_mask_offset(last_interaction_col_index, vec_index, inner_vec_index, -1);

    var diff = qm31_sub(qm31_sub(cur_cumsum, prev_row_cumsum), prev_col_cumsum);
    var fixed_diff = qm31_add(diff, input.cumsum_shift);

    var constraint = qm31_sub(qm31_mul(fixed_diff, frac.denominator), frac.numerator);
    add_constraint_qm31(constraint, vec_index, inner_vec_index);
    is_finalized = true;
}

fn next_trace_mask(col_index: u32, vec_index: u32, inner_vec_index: u32) -> M31 {
    let v0: M31 = extend_trace_output.extended_trace[col_index + N_EXTENDED_TRACE_OFFSET].data[flatten_idx(vec_index, inner_vec_index)];

    return v0;
}

fn next_interaction_trace_mask(col_index: u32, vec_index: u32, inner_vec_index: u32) -> QM31 {
    let base = col_index + N_INTERACTION_TRACE_OFFSET;
    let i = flatten_idx(vec_index, inner_vec_index);
    let v0: M31 = extend_trace_output.extended_trace[base].data[i];
    let v1: M31 = extend_trace_output.extended_trace[base + 1].data[i];
    let v2: M31 = extend_trace_output.extended_trace[base + 2].data[i];
    let v3: M31 = extend_trace_output.extended_trace[base + 3].data[i];

    return qm31_4(v0, v1, v2, v3);
}

fn next_interaction_trace_mask_offset(col_index: u32, vec_index: u32, inner_vec_index: u32, offset: i32) -> QM31 {
    var curr_row = vec_index * N_STATE + inner_vec_index;

    var row = offset_bit_reversed_circle_domain_index(curr_row, input.trace_domain_log_size, input.eval_domain_log_size, offset);

    var new_vec_index = row / N_LANES;
    var new_inner_vec_index = row % N_LANES;

    let base = col_index + N_INTERACTION_TRACE_OFFSET;
    let i = flatten_idx(new_vec_index, new_inner_vec_index);
    let v0: M31 = extend_trace_output.extended_trace[base].data[i];
    let v1: M31 = extend_trace_output.extended_trace[base + 1].data[i];
    let v2: M31 = extend_trace_output.extended_trace[base + 2].data[i];
    let v3: M31 = extend_trace_output.extended_trace[base + 3].data[i];

    let ret_val = QM31(CM31(v0, v1), CM31(v2, v3));
    return ret_val;
}

fn apply_external_round_matrix_state16(state: State16) -> State16 {
    var modified_state = state.data;
    for (var i = 0u; i < 4u; i++) {
        var x = array<M31, 4>(
            state.data[4 * i],
            state.data[4 * i + 1],
            state.data[4 * i + 2],
            state.data[4 * i + 3],
        );

        let t0 = m31_add(x[0], x[1]);
        let t02 = m31_add(t0, t0);
        let t1 = m31_add(x[2], x[3]);
        let t12 = m31_add(t1, t1);
        let t2 = m31_add(m31_add(x[1], x[1]), t1);
        let t3 = m31_add(m31_add(x[3], x[3]), t0);
        let t4 = m31_add(m31_add(t12, t12), t3);
        let t5 = m31_add(m31_add(t02, t02), t2);
        let t6 = m31_add(t3, t5);
        let t7 = m31_add(t2, t4);

        modified_state[4 * i] = t6;
        modified_state[4 * i + 1] = t5;
        modified_state[4 * i + 2] = t7;
        modified_state[4 * i + 3] = t4;
    }
    for (var j = 0u; j < 4u; j++) {
        let s = m31_add(m31_add(modified_state[j], modified_state[j + 4]), m31_add(modified_state[j + 8], modified_state[j + 12]));
        for (var i = 0u; i < 4u; i++) {
            modified_state[4 * i + j] = m31_add(modified_state[4 * i + j], s);
        }
    }
    return State16(modified_state);
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix_state16(state: State16) -> State16 {
    var sum = state.data[0];
    for (var i = 1u; i < N_STATE; i++) {
        sum = m31_add(sum, state.data[i]);
    }

    var result = State16(array<M31, N_STATE>());
    for (var i = 0u; i < N_STATE; i++) {
        let factor = partial_reduce(1u << (i + 1));
        result.data[i] = m31_add(m31_mul(M31(factor), state.data[i]), sum);
    }

    return result;
}
