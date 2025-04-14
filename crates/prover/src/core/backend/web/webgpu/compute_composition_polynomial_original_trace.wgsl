// Note: depends on qm31.wgsl, fraction.wgsl, utils.wgsl
// Define constants
const N_ROWS: u32 = 512;
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
const MAX_ARRAY_LOG_SIZE: u32 = 20;
const MAX_ARRAY_SIZE: u32 = 1u << MAX_ARRAY_LOG_SIZE;
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

const N_TRACES: u32 = 1 + N_COLUMNS + N_INTERACTION_COLUMNS;
const N_PREPROCESSED_TRACE_OFFSET: u32 = 0u;
const N_EXTENDED_TRACE_OFFSET: u32 = N_PREPROCESSED_TRACE_OFFSET + 1u;
const N_INTERACTION_TRACE_OFFSET: u32 = N_EXTENDED_TRACE_OFFSET + N_COLUMNS;

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
    total_sum: QM31,
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

@group(0) @binding(0)
var<storage, read> input: ComputeCompositionPolynomialInput;

@group(0) @binding(1)
var<storage, read_write> output: ComputeCompositionPolynomialOutput;

@group(0) @binding(2)
var<storage, read_write> extend_trace_output: ExtendTraceOutput;

var<private> constraint_index: u32 = 0u;

var<private> prev_col_cumsum: QM31 = QM31(CM31(M31(0u), M31(0u)), CM31(M31(0u), M31(0u)));

var<private> cur_frac: Fraction = ZERO_FRACTION;

var<private> is_first: M31 = M31(0u);

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
        var state: array<M31, N_STATE>;
        for (var j = 0u; j < N_STATE; j++) {
            state[j] = next_trace_mask(col_index, vec_index, inner_vec_index);
            col_index += 1u;
        }
        var initial_state = state;

        // 4 full rounds
        for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
            for (var j = 0u; j < N_STATE; j++) {
                state[j] = m31_add(state[j], M31(EXTERNAL_ROUND_CONSTS[i][j]));
            }
            state = apply_external_round_matrix(state);
            for (var j = 0u; j < N_STATE; j++) {
                state[j] = m31_pow5(state[j]);
            }
            for (var j = 0u; j < N_STATE; j++) {
                var m_1 = next_trace_mask(col_index, vec_index, inner_vec_index);
                let constraint = m31_sub(state[j], m_1);
                add_constraint(constraint, vec_index, inner_vec_index);

                state[j] = m_1;
                col_index += 1u;
            }
        }
        // Partial rounds
        for (var i = 0u; i < N_PARTIAL_ROUNDS; i++) {
            state[0] = m31_add(state[0], M31(INTERNAL_ROUND_CONSTS[i]));
            state = apply_internal_round_matrix(state);
            state[0] = m31_pow5(state[0]);
            var m_1 = next_trace_mask(col_index, vec_index, inner_vec_index);
            let constraint = m31_sub(state[0], m_1);
            add_constraint(constraint, vec_index, inner_vec_index);

            state[0] = m_1;
            col_index += 1u;
        }
        // 4 full rounds
        for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
            for (var j = 0u; j < N_STATE; j++) {
                state[j] = m31_add(state[j], M31(EXTERNAL_ROUND_CONSTS[i + N_HALF_FULL_ROUNDS][j]));
            }
            state = apply_external_round_matrix(state);
            for (var j = 0u; j < N_STATE; j++) {
                state[j] = m31_pow5(state[j]);
            }
            for (var j = 0u; j < N_STATE; j++) {
                var m_1 = next_trace_mask(col_index, vec_index, inner_vec_index);
                let constraint = m31_sub(state[j], m_1);
                add_constraint(constraint, vec_index, inner_vec_index);
                state[j] = m_1;
                col_index += 1u;
            }
        }

        // Store the final relation constraints
        let relation_constraint_1 = qm31_mul(ONE, QM31(CM31(state[0], M31(0u)), CM31(M31(0u), M31(0u))));
        let relation_constraint_2 = qm31_mul(qm31_neg(ONE), QM31(CM31(state[0], M31(0u)), CM31(M31(0u), M31(0u))));

        add_to_relation(array<RelationEntry, 2>(
            RelationEntry(ONE, initial_state),
            RelationEntry(qm31_neg(ONE), state)
        ), vec_index, inner_vec_index, rep_i);
    }
    finalize_logup(vec_index, inner_vec_index);

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

fn add_to_relation(entries: array<RelationEntry, 2>, vec_index: u32, inner_vec_index: u32, rep_i: u32) {
    var frac_sum = Fraction(QM31(CM31(M31(0u), M31(0u)), CM31(M31(0u), M31(0u))), QM31(CM31(M31(1u), M31(0u)), CM31(M31(0u), M31(0u))));
    for (var i = 0u; i < 2; i++) {
        var combined_value = QM31(CM31(M31(0u), M31(0u)), CM31(M31(0u), M31(0u)));
        for (var j = 0u; j < N_STATE; j++) {
            let value = QM31(CM31(entries[i].values[j], M31(0u)), CM31(M31(0u), M31(0u)));
            combined_value = qm31_add(combined_value, qm31_mul(input.lookup_elements.alpha_powers[j], value));
        }

        combined_value = qm31_sub(combined_value, input.lookup_elements.z);

        frac_sum = fraction_add(frac_sum, Fraction(entries[i].multiplicity, combined_value));
    }
    write_logup_frac(frac_sum, vec_index, inner_vec_index, rep_i);
}

fn write_logup_frac(frac: Fraction, vec_index: u32, inner_vec_index: u32, rep_i: u32) {
    if (!fraction_eq(cur_frac, ZERO_FRACTION)) {
        var interaction_col_index = (rep_i - 1u) * 4; // TODO: Improve this.
        var cur_cumsum = next_interaction_trace_mask(interaction_col_index, vec_index, inner_vec_index);
        var diff = qm31_sub(cur_cumsum, prev_col_cumsum);
        prev_col_cumsum = cur_cumsum;
        var constraint = qm31_sub(qm31_mul(diff, cur_frac.denominator), cur_frac.numerator);
        add_constraint_qm31(constraint, vec_index, inner_vec_index);
    } else {
        is_first = extend_trace_output.extended_trace[N_PREPROCESSED_TRACE_OFFSET].data[flatten_idx(vec_index, inner_vec_index)];
        is_finalized = false;
    }
    cur_frac = frac;
}

fn finalize_logup(vec_index: u32, inner_vec_index: u32) {
    if (is_finalized) {
        return;
    }
    // TODO: add support for when claimed_sum is not None.
    var last_interaction_col_index = (N_INSTANCES_PER_ROW - 1u) * 4u;

    var cur_cumsum = next_interaction_trace_mask(last_interaction_col_index, vec_index, inner_vec_index);

    var prev_row_cumsum = next_interaction_trace_mask_offset(last_interaction_col_index, vec_index, inner_vec_index, -1);

    var total_sum_mod = qm31_mul(QM31(CM31(is_first, M31(0u)), CM31(M31(0u), M31(0u))), input.total_sum);

    var fixed_prev_row_cumsum = qm31_sub(prev_row_cumsum, total_sum_mod);

    var diff = qm31_sub(qm31_sub(cur_cumsum, fixed_prev_row_cumsum), prev_col_cumsum);
    var constraint = qm31_sub(qm31_mul(diff, cur_frac.denominator), cur_frac.numerator);
    add_constraint_qm31(constraint, vec_index, inner_vec_index);
    is_finalized = true;
}

fn next_trace_mask(col_index: u32, vec_index: u32, inner_vec_index: u32) -> M31 {
    return extend_trace_output.extended_trace[col_index + N_EXTENDED_TRACE_OFFSET].data[flatten_idx(vec_index, inner_vec_index)];
}

fn next_interaction_trace_mask(col_index: u32, vec_index: u32, inner_vec_index: u32) -> QM31 {
    // get the next 4 values in the interaction trace columns
    return QM31(
        CM31(
            extend_trace_output.extended_trace[col_index + N_INTERACTION_TRACE_OFFSET].data[flatten_idx(vec_index, inner_vec_index)],
            extend_trace_output.extended_trace[col_index + N_INTERACTION_TRACE_OFFSET + 1].data[flatten_idx(vec_index, inner_vec_index)]
        ),
        CM31(
            extend_trace_output.extended_trace[col_index + N_INTERACTION_TRACE_OFFSET + 2].data[flatten_idx(vec_index, inner_vec_index)],
            extend_trace_output.extended_trace[col_index + N_INTERACTION_TRACE_OFFSET + 3].data[flatten_idx(vec_index, inner_vec_index)]
        )
    );
}

fn next_interaction_trace_mask_offset(col_index: u32, vec_index: u32, inner_vec_index: u32, offset: i32) -> QM31 {
    var curr_row = vec_index * N_STATE + inner_vec_index;

    var row = offset_bit_reversed_circle_domain_index(curr_row, input.trace_domain_log_size, input.eval_domain_log_size, offset);

    var new_vec_index = row / N_LANES;
    var new_inner_vec_index = row % N_LANES;
    return QM31(
        CM31(
            extend_trace_output.extended_trace[col_index + N_INTERACTION_TRACE_OFFSET].data[flatten_idx(new_vec_index, new_inner_vec_index)],
            extend_trace_output.extended_trace[col_index + N_INTERACTION_TRACE_OFFSET + 1].data[flatten_idx(new_vec_index, new_inner_vec_index)]
        ),
        CM31(
            extend_trace_output.extended_trace[col_index + N_INTERACTION_TRACE_OFFSET + 2].data[flatten_idx(new_vec_index, new_inner_vec_index)],
            extend_trace_output.extended_trace[col_index + N_INTERACTION_TRACE_OFFSET + 3].data[flatten_idx(new_vec_index, new_inner_vec_index)]
        )
    );
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix(state: array<M31, N_STATE>) -> array<M31, N_STATE> {
    // Applies circ(2M4, M4, M4, M4).
    var modified_state = state;
    for (var i = 0u; i < 4u; i++) {
        let partial_state = array<M31, 4>(
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        );
        let modified_partial_state = apply_m4(partial_state);
        modified_state[4 * i] = modified_partial_state[0];
        modified_state[4 * i + 1] = modified_partial_state[1];
        modified_state[4 * i + 2] = modified_partial_state[2];
        modified_state[4 * i + 3] = modified_partial_state[3];
    }
    for (var j = 0u; j < 4u; j++) {
        let s = m31_add(m31_add(modified_state[j], modified_state[j + 4]), m31_add(modified_state[j + 8], modified_state[j + 12]));
        for (var i = 0u; i < 4u; i++) {
            modified_state[4 * i + j] = m31_add(modified_state[4 * i + j], s);
        }
    }
    return modified_state;
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix(state: array<M31, N_STATE>) -> array<M31, N_STATE> {
    var sum = state[0];
    for (var i = 1u; i < N_STATE; i++) {
        sum = m31_add(sum, state[i]);
    }

    var result = array<M31, N_STATE>();
    for (var i = 0u; i < N_STATE; i++) {
        let factor = partial_reduce(1u << (i + 1));
        result[i] = m31_add(m31_mul(M31(factor), state[i]), sum);
    }

    return result;
}

/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4(x: array<M31, 4>) -> array<M31, 4> {
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
    return array<M31, 4>(t6, t5, t7, t4);
}
