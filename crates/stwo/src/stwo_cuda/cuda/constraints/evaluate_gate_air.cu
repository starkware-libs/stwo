// ============================================================================
// gate_air MAIN component constraint kernel — PHASE 1 (algebraic core).
//
// !!! BOX-UNVALIDATED CUDA — cannot be compiled on the laptop (no nvcc). !!!
// Transcribed line-for-line from `impl FrameworkEval for GateEval`
// (gate-air-leaf/src/main.rs, QUBIT-MEMORY + ts=pc+1 (inlined) + single-`d` rc-table encoding).
// Build + validate on the GPU box.
//
// QUBIT-MEMORY LAYOUT (main.rs, 19 cols, ACCESS_BLOCK=4): the main (witness) trace is:
//   [0..4) opcode one-hots; [4..8) target addr/prev_ts/v/d;
//   [8..12) ctrl_a access; [12..16) ctrl_b access; [16..19) ab/fire/delta.
// ts (= pc+1) and the target's v_after (= v_before+delta) are INLINED, NOT columns.
// `enabler`, `shot_id`, `pc`, `pc_in_prog` live in the PREPROCESSED tree (tree0), read via
// eval0.get_preprocessed_column() in that call order (GATE_AIR_N_PREPROCESSED = 4). `pc` feeds
// the inlined ts = pc + 1.
//
// Pipeline (mirror of evaluate_memory_address_to_id.cu):
//   1. pre_kernel  : per eval-domain row, run the 15 ALGEBRAIC add_constraints
//                    (-> numerators[row] = row_res) and emit the 10 LogUp
//                    relation entries (-> intermediate_fractions). PHASE 1.
//                    (ts=pc+1 inlined + single-`d` rc-table: per active access 1 ts algebraic
//                    constraint (RANGE recon only; PIN dropped) + 1 rc LOOKUP (TAG_RC, d);
//                    the v_after equality is dropped (inlined). Algebraic 19->15,
//                    LogUp entries 10 (3 qubitmem pairs + 3 rc-d + 1 program).)
//   2. post_kernel : generic_constraint_post_kernel (evaluate_common.cuh) folds
//                    the 10 fractions into 5 batches (all pairs)
//                    and adds the 5 LogUp cumsum constraints. PHASE 2 SOUNDNESS
//                    GATE — wired here but NOT trusted until box accumulator-diff
//                    is zero.
//   3. finalize    : generic_constraint_quotients_finalize_kernel — quotient =
//                    numerators[row] * denom_inv[row >> trace_log_size], written
//                    into the 4 accumulator coord columns honoring
//                    should_accumulate. Identical to accumulate_pointwise_cpu.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_common.cuh"
#include "evaluate_gate_air.cuh"

#define GATE_AIR_THREAD_COUNT_MAX 256

// ----------------------------------------------------------------------------
// interaction_shift_neg1 (F2-b / Option B): materialize the shifted `prev_row_cumsum`
// as its OWN aligned column so the composition post_kernel reads it at offset 0.
//
// The composition post_kernel's last LogUp batch reads prev_row_cumsum at the `-1`
// offset via offset_bit_reversed_circle_domain_index(row, dom, eval, -1) — a fixed,
// DATA-INDEPENDENT permutation of the eval domain (utils.cuh:120). For each of the 4
// last-LogUp QM31 coord columns we precompute, once per proof, a shifted copy:
//     shifted[row] = src[ offset_bit_reversed_circle_domain_index(row, dom, eval, -1) ].
// This is a pure gather (one thread per row) in the SAME style as the K4 prefix-sum
// index kernels. After the shift, BOTH `cur_cumsum` (src[row]) and `prev_row_cumsum`
// (shifted[row]) are offset-0 pointwise reads, so tree2 tiles row-by-row exactly like
// tree0/tree1 (the scattered `-1` no longer forces all 20 interaction cols resident).
//
// Byte-identical by construction: shifted[row] holds exactly the value the current
// scattered read produces, including the coset-boundary wrap (obr_index computes the
// correct wrapped target for every row). The scalar `cumsum_shift` correction in the
// post_kernel is orthogonal and UNCHANGED.
__global__ void interaction_shift_neg1_kernel(
    const m31 *src,
    m31 *shifted,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int eval_domain_size
) {
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;
    const unsigned int target_row = offset_bit_reversed_circle_domain_index(
        row, domain_log_size, eval_domain_log_size, -1);
    shifted[row] = src[target_row];
}

// ----------------------------------------------------------------------------
// Relation slicing helper.
//
// gate_air draws ONE width-6 relation (`relation!(GateRel, 6)`, main.rs) shared by all three
// logical relations (qubitmem / rc / program); they are kept distinct only by the integer TAG
// prepended as values[0] (main.rs). Every emit combines against the SAME (z, alpha, alpha_powers),
// using only the first N alpha_powers for an N-wide tuple — see the Rust `combine`
// (constraint-framework logup.rs), which folds `alpha_powers[0..values.len()]` and subtracts z.
//
// The `CudaEvaluator::add_to_relation<N>(RelationEntry<N>)` overload requires a `RelationEntry<N>`
// whose `relation` field is a `LookupElementsBasic<N>`. The gate relation is
// `LookupElementsBasic<6>`, so for the N<6 entries (qubitmem N=5, rc N=3) we build the matching
// `LookupElementsBasic<N>` by copying z, alpha, and the first N alpha_powers. This yields
// bit-identical `combine` (the math only touches alpha_powers[0..N]). N==6 (program) needs no slice.
template<int N>
DEVICE_FORCEINLINE LookupElementsBasic<N> gate_relation_slice(
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &rel
) {
    LookupElementsBasic<N> sliced;
    sliced.z = rel.z;
    sliced.alpha = rel.alpha;
    for (int i = 0; i < N; ++i) {
        sliced.alpha_powers[i] = rel.alpha_powers[i];
    }
    return sliced;
}

// ----------------------------------------------------------------------------
// Per-access masks (qubit-memory), in the EXACT order `access_masks` (main.rs) consumes columns
// from the main trace: addr, prev_ts, v, d. 4 columns per access (ACCESS_BLOCK).
// ts is NOT a column — it is the inlined `pc + 1`.
// ----------------------------------------------------------------------------
struct GateAccessMasks {
    m31 addr;
    m31 prev_ts;
    m31 v;
    m31 d;        // ts-ordering diff d = ts - prev_ts - 1 = pc - prev_ts (range-checked into [0,2^rc_log))
};

// Mirror of `access_masks` (main.rs): pull 4 consecutive main-trace masks (addr,prev_ts,v then
// d — matching cell_at's per-access column order).
template<typename EvaluatorT>
DEVICE_FORCEINLINE GateAccessMasks gate_access_masks(EvaluatorT &eval) {
    GateAccessMasks a;
    a.addr    = eval.next_trace_mask();
    a.prev_ts = eval.next_trace_mask();
    a.v       = eval.next_trace_mask();
    a.d       = eval.next_trace_mask();
    return a;
}

// Mirror of `add_qubitmem_pair` (main.rs): emit the chain Use(predecessor) + Yield(successor) pair
// for one access, gated by `active`. `ts` = the inlined timestamp (pc+1, shared across the step);
// `v_out` = value written forward (v_after = v_before+delta for the target, v for a control read).
//   Use   [+active] : (TAG_QUBITMEM, shot, addr, prev_ts, v_before)
//   Yield [-active] : (TAG_QUBITMEM, shot, addr, ts=pc+1, v_out)
// Order is Use then Yield — load-bearing (fraction index order).
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_add_qubitmem_pair(
    EvaluatorT &eval,
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &relation,
    m31 shot_id,
    const GateAccessMasks &a,
    m31 ts,
    m31 v_out,
    m31 active
) {
    qm31 mult = { { active, 0 }, { 0, 0 } };            // E::EF::from(active)
    m31 use_values[5] = { m31(GATE_AIR_TAG_QUBITMEM), shot_id, a.addr, a.prev_ts, a.v };
    RelationEntry<5> use_entry(gate_relation_slice<5>(relation), mult, use_values);
    eval.template add_to_relation<5>(use_entry);

    m31 neg_active = neg(active);
    qm31 neg_mult = { { neg_active, 0 }, { 0, 0 } };
    m31 yield_values[5] = { m31(GATE_AIR_TAG_QUBITMEM), shot_id, a.addr, ts, v_out };
    RelationEntry<5> yield_entry(gate_relation_slice<5>(relation), neg_mult, yield_values);
    eval.template add_to_relation<5>(yield_entry);
}

// Mirror of `add_rc_lookup` (main.rs): emit the SINGLE rc-table range-check LOOKUP for one access,
// gated by `active`. `d` is looked up as (TAG_RC, d); the rc supply table supplies each in-range
// value. One term/access (mirrored by gen_main_interaction and the in-circuit MainGate).
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_add_rc_lookup(
    EvaluatorT &eval,
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &relation,
    const GateAccessMasks &a,
    m31 active
) {
    qm31 mult = { { active, 0 }, { 0, 0 } };            // E::EF::from(active)
    m31 d_values[2] = { m31(GATE_AIR_TAG_RC), a.d };
    RelationEntry<2> d_entry(gate_relation_slice<2>(relation), mult, d_values);
    eval.template add_to_relation<2>(d_entry);
}

// Mirror of `add_ts_range` (main.rs): emit the ONE flag-gated degree-1 ts-ordering ALGEBRAIC
// constraint for one access:
//   RANGE: active * ((pc+1) - prev_ts - 1 - d) = 0 with d = ts - prev_ts - 1 = pc - prev_ts —
//          pins the witness `d` column to the diff (d is range-checked by gate_add_rc_lookup, not
//          here). `ts` is the inlined `pc + 1`. The old PIN constraint is GONE (ts is structurally
//          pc+1, so the pin is vacuous).
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_add_ts_range(
    EvaluatorT &eval,
    m31 ts,
    const GateAccessMasks &a,
    m31 active
) {
    // RANGE recon: active * ((ts - prev_ts - 1) - d), d = ts - prev_ts - 1 = pc - prev_ts.
    m31 recon = sub(sub(ts, a.prev_ts), m31(1));
    eval.add_constraint(mul(active, sub(recon, a.d)));
}

// ----------------------------------------------------------------------------
// pre_kernel: one thread per eval-domain row. Transcribes GateEval::evaluate.
// ----------------------------------------------------------------------------
template<typename EvaluatorT>
__launch_bounds__(GATE_AIR_THREAD_COUNT_MAX, 2)
__global__ void evaluate_gate_air_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    GateAirEval *gate_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array,
    unsigned row_offset,
    unsigned tile_rows
) {
    // Candidate 2 (row-tiling): this launch processes the eval-domain rows
    // [row_offset, row_offset + tile_rows). The thread's GLOBAL eval-domain row
    // is row_offset + local index — this MUST stay global because every trace
    // read (trace_evaluations[col][row]) and every mask-offset
    // (offset_bit_reversed_circle_domain_index(row, ...)) indexes by global row.
    // Only the `intermediate_fractions` buffer is tiled: the caller passes a
    // pointer biased back by row_offset*logup_counts, so the evaluator's
    // global-row index (row*logup_counts) lands at the correct tile-local slot
    // d_fractions[(row-row_offset)*logup_counts + i]. numerators[] /
    // constraint_index_array[] stay full-size and are indexed by global row.
    const unsigned local = threadIdx.x + blockDim.x * blockIdx.x;
    if (local >= tile_rows) {
        return;
    }
    const unsigned row = row_offset + local;

    // Two evaluators sharing the SAME row_res / constraint_index / fraction
    // index accumulation. Each evaluator tracks its own col_index per
    // interaction. The PREPROCESSED columns (tree0) and the MAIN columns (tree1)
    // are two separate device pointer tables, so we mirror
    // memory_address_to_id: use eval0 for trace0 reads (preprocessed) and eval
    // for trace1 reads (main), carrying row_res / constraint_index /
    // fraction_index forward by hand.
    //
    // PREPROCESSED (main.rs GateEval::evaluate). The Rust `evaluate()` reads FOUR
    // preprocessed columns up front, in this EXACT call order:
    //     enabler, shot_id, pc, pc_in_prog
    // (= GateEval's `preprocessed_column_indices` order, which is the order the
    // InfoEvaluator records `get_preprocessed_column` calls; the CUDA component
    // prover gathers `trace0_evaluations` in precisely that index order — see
    // constraint-framework component.rs / component_prover.rs). eval0 therefore
    // reads them sequentially via col_index[0] in the SAME order. The main trace is
    // the 19-column qubit-memory layout (4 opcode masks + target(4) + 2*ctrl(4) + 3;
    // ts = pc+1 and v_after = v_before+delta are inlined, not columns).

    EvaluatorT eval0(
        trace0_evaluations, random_coeff_powers,
        0, row, qm31{{0, 0}, {0, 0}},
        0, cumsum_shift, domain_log_size, eval_domain_log_size,
        intermediate_fractions, logup_counts
    );
    // Preprocessed (tree0) reads, in GateEval call order (main.rs:778-781):
    //   enabler, shot_id, pc, pc_in_prog.
    m31 enabler    = eval0.get_preprocessed_column();  // gate_enabler   (main.rs:778)
    m31 shot_id    = eval0.get_preprocessed_column();  // gate_shot_id   (main.rs:779)
    m31 pc         = eval0.get_preprocessed_column();  // gate_pc        (main.rs:780)
    m31 pc_in_prog = eval0.get_preprocessed_column();  // gate_pc_in_prog(main.rs:781)

    EvaluatorT eval(
        trace1_evaluations, random_coeff_powers,
        0, row, qm31{{0, 0}, {0, 0}},
        0, cumsum_shift, domain_log_size, eval_domain_log_size,
        intermediate_fractions, logup_counts
    );

    // --- Main-trace masks (19 cols), in declaration order (main.rs GateEval::evaluate). ---
    // Header is ONLY the 4 opcode masks (enabler/shot_id/pc/pc_in_prog are tree0).
    m31 is_nop     = eval.next_trace_mask();   // col 0
    m31 is_not     = eval.next_trace_mask();   // col 1
    m31 is_cnot    = eval.next_trace_mask();   // col 2
    m31 is_toffoli = eval.next_trace_mask();   // col 3

    // target access (addr,prev_ts,v,d) cols 4..8. ts (=pc+1) and v_after (=v_before+delta)
    // are NOT columns — inlined below.
    GateAccessMasks target = gate_access_masks(eval);   // cols 4..8
    GateAccessMasks ctrl_a = gate_access_masks(eval);   // cols 8..12
    GateAccessMasks ctrl_b = gate_access_masks(eval);   // cols 12..16

    m31 ab    = eval.next_trace_mask();   // col 16
    m31 fire  = eval.next_trace_mask();   // col 17
    m31 delta = eval.next_trace_mask();   // col 18

    // ts = pc + 1 (inlined affine of the preprocessed pc, shared by all accesses of the step).
    m31 ts = add(pc, m31(1));
    // v_after = v_before + delta (inlined; target.v is v_before).
    m31 v_after = add(target.v, delta);

    // ===================== ALGEBRAIC CONSTRAINTS (part 1: 12) =====================
    // Order MUST match GateEval::evaluate exactly. (The 3 ts-ordering accesses add 3 more algebraic
    // constraints — ONE RANGE recon per access — emitted below in `add_ts_range` position, for 15.
    // The old PIN per access and the v_after equality are dropped.)
    // [1-4] opcode booleanity op*(op-1) for is_nop/is_not/is_cnot/is_toffoli.
    eval.add_constraint(mul(is_nop,     sub(is_nop,     m31(1))));
    eval.add_constraint(mul(is_not,     sub(is_not,     m31(1))));
    eval.add_constraint(mul(is_cnot,    sub(is_cnot,    m31(1))));
    eval.add_constraint(mul(is_toffoli, sub(is_toffoli, m31(1))));
    // [5] one-hot sum = enabler: enabler - is_nop - is_not - is_cnot - is_toffoli.
    eval.add_constraint(
        sub(sub(sub(sub(enabler, is_nop), is_not), is_cnot), is_toffoli)
    );

    // active sub-expressions. NOT columns — computed inline.
    m31 a_active = add(is_cnot, is_toffoli);
    m31 b_active = is_toffoli;

    // [6-9] value booleanity v*(v-1) for target.v, v_after(=v_before+delta), ctrl_a.v, ctrl_b.v (in
    // this order). Booleanity on the derived v_after keeps the target's written memory value a bit.
    eval.add_constraint(mul(target.v, sub(target.v, m31(1))));
    eval.add_constraint(mul(v_after,  sub(v_after,  m31(1))));
    eval.add_constraint(mul(ctrl_a.v, sub(ctrl_a.v, m31(1))));
    eval.add_constraint(mul(ctrl_b.v, sub(ctrl_b.v, m31(1))));

    // --- Gate-apply on the memory values. ---
    m31 a_bit = ctrl_a.v;
    m31 b_bit = ctrl_b.v;
    m31 t_bit = target.v;   // v_before

    // [10] ab - a_bit*b_bit.
    eval.add_constraint(sub(ab, mul(a_bit, b_bit)));
    // [11] fire - is_not - is_cnot*a_bit - is_toffoli*ab.
    eval.add_constraint(
        sub(sub(sub(fire, is_not), mul(is_cnot, a_bit)), mul(is_toffoli, ab))
    );
    // [12] delta - fire + 2*t_bit*fire  (delta = v_after - v_before = fire*(1 - 2*v_before)).
    // (The old [13] v_after - v_before - delta = 0 equality is dropped — v_after is now inlined
    // as v_before + delta.)
    eval.add_constraint(
        add(sub(delta, fire), mul(mul(t_bit, fire), m31(2)))
    );

    // ===================== LOGUP RELATION ENTRIES (10) =====================
    // PHASE 1 emits the 10 entries (the post_kernel turns them into 5 LogUp batch constraints in
    // PHASE 2: all pairs). Order MUST match `evaluate()` / gen_main_interaction exactly:
    //   qubitmem target/ctrl_a/ctrl_b Use/Yield (6), rc target/ctrl_a/ctrl_b d (3), program (1).
    // The 10-entry stream folds into 5 pairs: (t_use,t_yield)(a_use,a_yield)(b_use,b_yield)
    //   (rc_t, rc_a)(rc_b, program).
    // The ts-ordering RANGE recon is a degree-1 algebraic add_constraint (gate_add_ts_range), NOT a
    // relation entry; it is emitted AFTER the rc LOOKUPs and BEFORE the program emit (the same
    // interleave as main.rs `evaluate()`), so the relation-batch order stays qubitmem, rc, program.
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &rel = gate_eval->relation;

    // 1,2. qubitmem target: Use(+enabler)/Yield(-enabler); ts = pc+1, v_out = v_after = v_before+delta.
    gate_add_qubitmem_pair(eval, rel, shot_id, target, ts, v_after, enabler);
    // 3,4. qubitmem ctrl_a: Use(+a_active)/Yield(-a_active); ts = pc+1, read propagates value (v_out = v).
    gate_add_qubitmem_pair(eval, rel, shot_id, ctrl_a, ts, ctrl_a.v, a_active);
    // 5,6. qubitmem ctrl_b: Use(+b_active)/Yield(-b_active); ts = pc+1.
    gate_add_qubitmem_pair(eval, rel, shot_id, ctrl_b, ts, ctrl_b.v, b_active);

    // 7 / 8 / 9. rc range-check LOOKUPs (single `d`) for target / ctrl_a / ctrl_b — emitted
    // as a group AFTER the qubitmem pairs (mirrors main.rs `add_rc_lookup` × 3 order).
    gate_add_rc_lookup(eval, rel, target, enabler);
    gate_add_rc_lookup(eval, rel, ctrl_a, a_active);
    gate_add_rc_lookup(eval, rel, ctrl_b, b_active);

    // ts-ordering RANGE recon: 1 degree-1 algebraic add_constraint per access (#13..15), NOT relation
    // entries. ts = pc+1 inlined; the old PIN per access is gone (main.rs `add_ts_range` order).
    gate_add_ts_range(eval, ts, target, enabler);
    gate_add_ts_range(eval, ts, ctrl_a, a_active);
    gate_add_ts_range(eval, ts, ctrl_b, b_active);

    // 10. program (+enabler).
    //   opcode_scalar = is_not*1 + is_cnot*2 + is_toffoli*3.
    //   [TAG_PROGRAM, pc_in_prog, opcode_scalar, target.addr, ctrl_a.addr, ctrl_b.addr]
    {
        m31 opcode_scalar = add(add(is_not, mul(is_cnot, m31(2))), mul(is_toffoli, m31(3)));
        m31 prog[6] = {
            m31(GATE_AIR_TAG_PROGRAM), pc_in_prog, opcode_scalar,
            target.addr, ctrl_a.addr, ctrl_b.addr
        };
        qm31 mult = { { enabler, 0 }, { 0, 0 } };
        RelationEntry<6> entry(gate_relation_slice<6>(rel), mult, prog);
        eval.template add_to_relation<6>(entry);
    }

    // Persist the algebraic count so the post_kernel resumes the random-coeff
    // index at the right spot (mirror memory_address_to_id.cu:91-92).
    constraint_index_array[row] = eval.constraint_index;
    numerators[row] = eval.row_res;
}

// ----------------------------------------------------------------------------
// Tiled post_kernel (Candidate 2 row-tiling).
//
// BYTE-IDENTICAL copy of generic_constraint_post_kernel (evaluate_common.cuh
// :45-117) with TWO mechanical changes for row-tiling, nothing else:
//   * the global eval-domain row is row_offset + local thread index, and the
//     bound check is `local >= tile_rows` (instead of row >= eval_domain_size);
//   * `intermediate_fractions` is the tile-biased pointer supplied by the host
//     (d_fractions - row_offset*logup_counts), so the row*logup_counts indexing
//     reads the tile-local slots written by the tiled pre_kernel.
// The per-row math (Fraction::sum, the cumsum masks via next_extension_
// interaction_mask, diff/fixed_diff, add_constraint_ext, finalize_logup_in_pairs
// folding) is copied verbatim — do NOT alter it. We keep a private copy here so
// the shared generic_constraint_post_kernel (89 callers) is left untouched.
template<typename EvaluatorT>
__launch_bounds__(GATE_AIR_THREAD_COUNT_MAX, 2)
__global__ void evaluate_gate_air_post_kernel_tiled(
    qm31 *numerators,
    Fraction *intermediate_fractions,
    unsigned *constraint_index_array,
    m31 **trace2_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int logup_counts,
    unsigned int last_batch,
    qm31 cumsum_shift,
    unsigned row_offset,
    unsigned tile_rows,
    // F2-b / Option B: when true, the 4 shifted last-LogUp cumsum coords are appended
    // to trace2_evaluations at indices [logup_cols*4 .. logup_cols*4 + 4) and the last
    // batch reads prev_row_cumsum from them at OFFSET 0 (both tileable) instead of via
    // the scattered {0,-1} mask on the source coords. When false, the byte-for-byte
    // legacy resident path (scattered -1 read on the source coords) runs unchanged.
    bool tree2_shifted
) {
    const unsigned local = threadIdx.x + blockDim.x * blockIdx.x;
    if (local >= tile_rows) return;
    const unsigned row = row_offset + local;

    EvaluatorT evaluator(
        trace2_evaluations,
        random_coeff_powers,
        constraint_index_array[row],
        row,
        numerators[row],
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    const unsigned logup_interaction = 2;
    qm31 prev_col_cumsum = { {0, 0}, {0, 0} };

    // Process complete batches
    for (unsigned i = 0; i < last_batch; ++i) {
        const Fraction cur_frac = Fraction::sum(&intermediate_fractions[2 * i + row * logup_counts], 2);

        qm31 cur_cumsum_arr[2] = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} } };
        int offsets[2] = { 0, 0 };
        evaluator.next_extension_interaction_mask(logup_interaction, offsets, 1, cur_cumsum_arr);

        const qm31 cur_cumsum = cur_cumsum_arr[0];
        const qm31 diff = sub(cur_cumsum, prev_col_cumsum);
        prev_col_cumsum = cur_cumsum;

        const qm31 constraint_val = sub(mul(diff, cur_frac.denominator), cur_frac.numerator);
        evaluator.add_constraint_ext(constraint_val);
    }

    // Process remaining fractions
    {
        unsigned remaining_fractions = logup_counts - last_batch * 2;
        const Fraction frac_sum = Fraction::sum(&intermediate_fractions[last_batch * 2 + row * logup_counts], remaining_fractions);

        qm31 cur_cumsum;
        qm31 prev_row_cumsum;
        if (tree2_shifted) {
            // F2-b / Option B: cur_cumsum from the source coords (cols [16..20)) at
            // offset 0, prev_row_cumsum from the appended shifted coords (cols [20..24))
            // at offset 0. Both offset-0 pointwise reads over the row-tiled trace2
            // pointer table — no scattered `-1`, so the tile slice is self-contained.
            // The two consecutive next_extension_interaction_mask calls advance
            // col_index[2]: 16->20 (source) then 20->24 (shifted). BYTE-IDENTICAL to
            // the scattered read below because shifted[row] == src[obr_index(row,-1)].
            int off0[1] = { 0 };
            qm31 cur_arr[1] = { { {0, 0}, {0, 0} } };
            evaluator.next_extension_interaction_mask(logup_interaction, off0, 1, cur_arr);
            cur_cumsum = cur_arr[0];

            qm31 prev_arr[1] = { { {0, 0}, {0, 0} } };
            evaluator.next_extension_interaction_mask(logup_interaction, off0, 1, prev_arr);
            prev_row_cumsum = prev_arr[0];
        } else {
            // Legacy resident path: scattered `-1` read on the source coords (cols
            // [16..20)) — BYTE-FOR-BYTE the pre-tiling behavior.
            int offsets2[2] = { 0, -1 };
            qm31 cumsum2[2] = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} } };
            evaluator.next_extension_interaction_mask(logup_interaction, offsets2, 2, cumsum2);
            cur_cumsum = cumsum2[0];
            prev_row_cumsum = cumsum2[1];
        }

        const qm31 diff = sub(sub(cur_cumsum, prev_row_cumsum), prev_col_cumsum);
        const qm31 fixed_diff = add(diff, cumsum_shift);

        const qm31 constraint_val = sub(mul(fixed_diff, frac_sum.denominator), frac_sum.numerator);

        evaluator.add_constraint_ext(constraint_val);
    }

    numerators[row] = evaluator.row_res;
}

// ----------------------------------------------------------------------------
// Host entry: launch pre -> post -> finalize (mirror evaluate_memory_address_to_id).
// ----------------------------------------------------------------------------
// Resolve the row-tile size (rows/block). GATE_AIR_TILE_ROWS is the preferred
// knob (COMPOSITION_TILING_SCOPE §3, default in [2^20, 2^22]); GATE_AIR_COMP_TILE
// is kept as a back-compat alias for the fraction-only tiling that shipped in
// steps 1-2. Default 2^20. Clamped to eval_domain_size by the caller.
static unsigned gate_air_resolve_tile_rows(bool resident) {
    // Improvement 3 (secondary): on the RESIDENT path the tiling only bounds the
    // d_fractions transient (numerators is full-domain regardless), and removing the
    // per-tile syncs means fewer/larger tiles are strictly better (fewer launch waves,
    // longer kernels to pipeline). Raise the default to 2^22 so a 2^27 eval domain runs
    // ~32 tiles instead of ~128; this only enlarges d_fractions
    // (tile_rows*logup_counts*sizeof(Fraction) = 2^22*10*32B ~= 1.34 GiB vs ~320 MiB at
    // 2^20, a ~1 GiB transient the resident shard has VRAM headroom for). The STAGED
    // path keeps the 2^20 default because a bigger tile_rows also grows every staged
    // per-column tile buffer + per-block H2D slice (the residency ceiling the staging
    // path exists to hold down). An explicit GATE_AIR_TILE_ROWS / GATE_AIR_COMP_TILE
    // env override wins on BOTH paths and is unchanged.
    unsigned tile_rows = resident ? (1u << 22) : (1u << 20);
    if (const char *env = std::getenv("GATE_AIR_TILE_ROWS")) {
        unsigned long parsed = std::strtoul(env, nullptr, 10);
        if (parsed > 0) tile_rows = (unsigned)parsed;
    } else if (const char *env2 = std::getenv("GATE_AIR_COMP_TILE")) {
        unsigned long parsed = std::strtoul(env2, nullptr, 10);
        if (parsed > 0) tile_rows = (unsigned)parsed;
    }
    return tile_rows;
}

extern "C"
void evaluate_gate_air(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    // COMPOSITION_TILING_SCOPE (route c): host-tile-source pointers for tree0/tree1.
    // When BOTH are non-null the columns are host-staged (GATE_AIR_STREAM_COMMIT):
    // instead of holding all trace0_evaluations_len + trace1_evaluations_len eval
    // columns resident (~47 GB @2^24), we H2D only the current row-block's
    // contiguous slice of each column into a REUSED tile buffer and index it with a
    // biased pointer. host_trace{0,1}[c] is the base of column c's committed eval
    // bytes in the SAME eval-domain order the kernel indexes, so the physical slice
    // [tile_start, tile_start+this_tile) is exactly the logical row-block. When
    // NULL (legacy/resident path) the trace{0,1}_evaluations device pointers are
    // used whole — BYTE-FOR-BYTE the pre-tiling behavior. tree2 is ALWAYS resident
    // (its post_kernel `-1` LogUp-cumsum offset is a bit-reversed scattered index, a
    // row-block halo would read wrong bytes — scope §1.3/§1.4), so it has no host
    // supply here; post_kernel + generic_constraint_post_kernel are unchanged.
    const uint32_t * const *host_trace0,
    const uint32_t * const *host_trace1,
    // F2-b / Option B: per-column host-tile-source table for tree2 (interaction),
    // mirroring host_trace0/1. Entry c = the column's committed host stash bytes if
    // STAGED (row-tiled H2D per block), or NULL if RESIDENT (kernel uses the live
    // trace2_evaluations[c] pointer whole). A wholly-null table => tree2 fully
    // resident; the kernel then keeps tree2 resident AND takes the scattered `-1`
    // legacy read (byte-for-byte). When non-null, tree2 is row-tiled like tree0/1 and
    // the 4 shifted last-LogUp coords are precomputed (interaction_shift_neg1) so the
    // last batch reads prev_row_cumsum at offset 0. MUST match the Rust FFI + entry.cu.
    const uint32_t * const *host_trace2,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,
    bool use_assert_evaluator,
    cudaStream_t stream
) {
    (void)number_of_columns;

    GateAirEval *gate_eval = (GateAirEval *) eval;
    const unsigned eval_domain_size = 1u << eval_domain_log_size;

    // FULL (A) per-column mixed supply: the input columns are a MIX of host-staged
    // (the 19 tree1 eval columns, dehydrated by GATE_AIR_STREAM_COMMIT — their
    // trace{0,1}_evaluations device pointers are FREED stash keys, must not be
    // dereferenced) and RESIDENT (the 4 tree0 preprocessed + the small tree1
    // multiplicity/witness/program columns, kept live on device — their
    // trace{0,1}_evaluations pointers are valid full-eval-domain buffers). The host
    // supplies host_trace{0,1} as PER-COLUMN tables: entry c is the column's committed
    // host bytes if staged, or NULL if that column is resident. A whole table may be
    // null when EVERY column of that tree is resident (the tree0 case). `tiled_input`
    // is true iff ANY column is staged (either table non-null); then the per-block loop
    // handles each column independently. When both tables are null (legacy/resident
    // shard) the whole-buffer resident path runs — BYTE-FOR-BYTE the pre-tiling
    // behavior.
    //
    // Read semantics preserved (verified): every gate_air trace read is
    // trace_evaluations[c][GLOBAL row] (next_interaction_mask, eval_at_row.cuh) at the
    // single component eval_domain_size — NO per-column lift/log_size remap. So a
    // resident column is supplied to the tiled pre_kernel as its LIVE full-size buffer
    // biased by -tile_start (biased[c] = trace_evaluations[c] - tile_start), exactly
    // like a staged tile buffer, and col[global_row] reads the identical element the
    // whole-buffer resident path reads. Byte-identical (same +off/-tile_start trick as
    // the quotient Site-1 fix). No H2D, no re-staging for resident columns.
    const bool tiled_input = (host_trace0 != nullptr) || (host_trace1 != nullptr);

    // F2-b / Option B: tree2 (interaction) is row-tiled iff a host_trace2 table is
    // supplied. Then the scattered `-1` composition read is replaced by an offset-0
    // read on 4 precomputed shifted columns (interaction_shift_neg1), so tree2 no
    // longer needs to be held whole (~14 GiB @2^26). When host_trace2 is null, tree2
    // stays fully resident and the post_kernel takes the legacy scattered `-1` read
    // (byte-for-byte unchanged). The 4 shifted coords are the LAST 4 tree2 columns
    // (the last LogUp batch's 4 QM31 coords, indices [len2-4, len2)); the shifted
    // copies are appended at indices [len2, len2+4) in the tiled tree2 pointer table.
    const bool tree2_tiled = (host_trace2 != nullptr);
    const unsigned N_SHIFT = 4;  // SECURE_EXTENSION_DEGREE coords of the last LogUp col

    // Improvement 3: on the RESIDENT base path (no host-staged tree0/1 columns and no
    // tiled tree2) the per-tile double-buffer H2D never runs, so there is nothing to
    // overlap-hide and the two per-tile cudaStreamSynchronize (after pre_kernel and
    // after post_kernel) are pure host<->device stalls that also serialize consecutive
    // tiles' kernels. Because every kernel of every tile is enqueued on the SAME
    // `stream`, stream ordering already guarantees tile b's pre precedes its post and
    // precedes tile b+1's kernels — the only shared per-tile scratch (d_fractions,
    // reused via frac_biased) is written by tile b's pre, read by tile b's post, then
    // overwritten by tile b+1's pre, all serialized on `stream`. So on the resident
    // path we drop the two mid-loop syncs (keeping a non-blocking cudaGetLastError()
    // launch-error check per tile) and issue ONE cudaStreamSynchronize after the loop
    // and finalize, before any host read of results. The STAGED path
    // (tiled_input || tree2_tiled) keeps its per-tile syncs EXACTLY as before: its
    // async ping-pong H2D reuses the tile buffers across blocks and depends on those
    // syncs for correctness. tile kernels / numerators layout / finalize are unchanged.
    const bool resident_path = !tiled_input && !tree2_tiled;

    // Resident device pointer tables. Under tiled_input, tree0/tree1 tables are
    // REBUILT per block into d_tile{0,1}_ptrs (staged cols -> biased tile buffer,
    // resident cols -> biased live buffer), so we do not clone the whole
    // trace{0,1}_evaluations here (staged entries are freed stash keys). tree2 is
    // always resident.
    m31 **d_trace0 = tiled_input ? nullptr
        : clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **d_trace1 = tiled_input ? nullptr
        : clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    // tree2 resident pointer table: only cloned whole when tree2 is NOT tiled (the
    // legacy scattered `-1` path reads all 20 columns resident). Under tree2_tiled the
    // per-block pointer table (d_tile2_ptrs) is rebuilt each iteration and this whole
    // clone is skipped.
    m31 **d_trace2 = tree2_tiled ? nullptr
        : clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    // NOTE: cuda_alloc_zeroes_uint32_t's argument is a COUNT OF uint32_t, not a
    // byte count. A qm31 is 4 uint32_t, so `eval_domain_size` qm31 accumulators
    // need `4 * eval_domain_size` u32 (2.0 GiB at 2^26). The previous
    // `sizeof(qm31) * eval_domain_size` passed 16 * eval_domain_size — a 4x
    // over-allocation (8.0 GiB at 2^26) whose 32-bit value (2^31) additionally
    // truncated in the old `int` helper param. Both are fixed here: correct u32
    // count + 64-bit clean multiply into the now-size_t helper. Byte-identical:
    // numerators[row] is only ever indexed for row < eval_domain_size.
    qm31 *numerators =
        (qm31 *) cuda_alloc_zeroes_uint32_t((size_t)4 * eval_domain_size);

    GateAirEval *d_gate_eval = cuda_malloc<GateAirEval>(1);
    cuda_mem_copy_host_to_device<GateAirEval>(gate_eval, d_gate_eval, 1);

    // ----- Row-tiling (COMPOSITION_TILING_SCOPE route c). -----
    // Steps 1-2 already row-tiled the d_fractions INTERMEDIATE (a ~12 GB @2^24
    // transient) with the same tile_start/this_tile passed to the kernels; the
    // kernels index every trace read by GLOBAL row and only the fraction pointer
    // was biased. This completes route (c): under tiled_input we ALSO row-tile the
    // 19 main + 4 preprocessed INPUT columns so their residency is
    // O(this_tile * (trace0_len + trace1_len)) instead of the full ~47 GB eval set.
    // The eval is pure pointwise for tree0/tree1 (offset 0, no next-row mask), so a
    // row-block is a contiguous committed-byte slice and biased-pointer indexing is
    // byte-trivially correct.
    unsigned tile_rows = gate_air_resolve_tile_rows(resident_path);
    if (tile_rows > eval_domain_size) tile_rows = eval_domain_size;

    Fraction *d_fractions =
        cuda_malloc<Fraction>((size_t)tile_rows * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    // Per-block tile buffers for tree0/tree1 STAGED columns only. One device buffer
    // per STAGED column of length tile_rows, REUSED across all blocks; a RESIDENT
    // column gets NO tile buffer (tileX_bufs[c] == nullptr) because it is supplied by
    // its live full-size device buffer, not H2D. Device residency for the staged
    // inputs = (num_staged_tree0 + num_staged_tree1) * tile_rows * 4B. The per-block
    // biased pointer tables are rebuilt into d_tile{0,1}_ptrs each iteration.
    // SYNCHRONOUS per-block H2D is landed here (correctness first, scope §2); the
    // async/pinned double-buffer overlap is DEFERRED (note below).
    m31 **tile0_bufs = nullptr;   // host array of device buffers (or null), trace0_len entries
    m31 **tile1_bufs = nullptr;   // host array of device buffers (or null), trace1_len entries
    m31 **d_tile0_ptrs = nullptr; // device pointer table (biased) for tree0
    m31 **d_tile1_ptrs = nullptr; // device pointer table (biased) for tree1
    if (tiled_input) {
        tile0_bufs = (m31 **)std::malloc(sizeof(m31 *) * trace0_evaluations_len);
        tile1_bufs = (m31 **)std::malloc(sizeof(m31 *) * trace1_evaluations_len);
        // Allocate a tile buffer ONLY for a staged column (non-null host entry). A
        // resident column (null host entry, or a wholly-null table) uses its live
        // device buffer, so no tile buffer is needed.
        for (unsigned c = 0; c < trace0_evaluations_len; ++c)
            tile0_bufs[c] = (host_trace0 != nullptr && host_trace0[c] != nullptr)
                ? (m31 *)cuda_malloc_uint32_t(tile_rows) : nullptr;
        for (unsigned c = 0; c < trace1_evaluations_len; ++c)
            tile1_bufs[c] = (host_trace1 != nullptr && host_trace1[c] != nullptr)
                ? (m31 *)cuda_malloc_uint32_t(tile_rows) : nullptr;
        d_tile0_ptrs = cuda_malloc<m31 *>(trace0_evaluations_len);
        d_tile1_ptrs = cuda_malloc<m31 *>(trace1_evaluations_len);
    }

    // ----- F2-b / Option B: tree2 tiling + shifted-column build. -----
    // Under tree2_tiled we (1) build the 4 shifted last-LogUp cumsum columns ONCE (a
    // data-independent gather, interaction_shift_neg1) and stash their bytes on the
    // host, and (2) allocate reused per-block tile buffers for all 20 tree2 columns
    // plus the 4 shifted columns, and a device pointer table of length
    // trace2_evaluations_len + N_SHIFT rebuilt each block. Device residency for tree2
    // at composition drops from all 20 resident to
    // (20 + 4) * tile_rows * 4B tile buffers (~a few hundred MiB at tile_rows=2^20).
    // The shifted columns are prover-internal (NOT committed / not in the Merkle
    // tree) — a pure recomputation of committed cumsum values.
    m31 **tile2_bufs = nullptr;      // reused device tile buffers, (len2 + N_SHIFT) entries
    m31 **d_tile2_ptrs = nullptr;    // device pointer table (biased), len2 + N_SHIFT
    // Host stash for the 4 shifted columns (full eval domain). Owned here; freed at end.
    uint32_t *shift_host[4] = { nullptr, nullptr, nullptr, nullptr };
    const unsigned len2 = trace2_evaluations_len;
    if (tree2_tiled) {
        // The 4 shifted coords are the LAST N_SHIFT tree2 columns.
        const unsigned shift_first = (len2 >= N_SHIFT) ? (len2 - N_SHIFT) : 0;

        // --- Build the 4 shifted columns once (whole-column device gather). ---
        for (unsigned k = 0; k < N_SHIFT; ++k) {
            const unsigned c = shift_first + k;
            // Obtain the whole SOURCE column on device: H2D from the host stash if
            // staged, else D2D-copy the resident live buffer. Transient (freed below).
            m31 *d_src = cuda_malloc_uint32_t(eval_domain_size);
            if (host_trace2[c] != nullptr) {
                cuda_mem_copy_host_to_device<uint32_t>(
                    (uint32_t *)host_trace2[c], (uint32_t *)d_src, eval_domain_size);
            } else {
                // Resident source column: D2D copy from its live full-domain buffer.
                cuda_mem_copy_device_to_device<m31>(
                    trace2_evaluations[c], d_src, eval_domain_size);
            }
            m31 *d_shift = cuda_malloc_uint32_t(eval_domain_size);
            int sblock = eval_domain_size < GATE_AIR_THREAD_COUNT_MAX
                ? (int)eval_domain_size : GATE_AIR_THREAD_COUNT_MAX;
            int snblocks = (eval_domain_size + sblock - 1) / sblock;
            interaction_shift_neg1_kernel<<<snblocks, sblock, 0, stream>>>(
                d_src, d_shift, domain_log_size, eval_domain_log_size, eval_domain_size);
            ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
            ASSERT_CUDA_SUCCESS(cudaGetLastError());
            // D2H the shifted column into an owned host buffer, then free both device
            // transients so the shifted build does NOT add resident footprint into the
            // composition plateau (only the reused tile buffers survive).
            shift_host[k] = (uint32_t *)std::malloc(sizeof(uint32_t) * eval_domain_size);
            copy_uint32_t_vec_from_device_to_host(
                (uint32_t *)d_shift, shift_host[k], (int)eval_domain_size);
            ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
            cuda_free_memory(d_src);
            cuda_free_memory(d_shift);
        }

        // --- Tile buffers: 20 source cols + 4 shifted cols, all reused per block. ---
        tile2_bufs = (m31 **)std::malloc(sizeof(m31 *) * (len2 + N_SHIFT));
        for (unsigned c = 0; c < len2 + N_SHIFT; ++c)
            tile2_bufs[c] = (m31 *)cuda_malloc_uint32_t(tile_rows);
        d_tile2_ptrs = cuda_malloc<m31 *>(len2 + N_SHIFT);
    }

    timer global_timer;
    global_timer.start("evaluate_gate_air");

    // gate_air uses `finalize_logup_in_pairs` => batching[i] = i/2. For logup_counts=10 that is
    // 5 full pairs (batches 0..4) = 5 batches (no singleton tail).
    // The post_kernel folds each pair, reads the interaction cumsum mask (offset
    // 0 for full batches; offsets {0,-1} + cumsum_shift for the last), and adds
    // `diff*denom - num`. This is EXACTLY finalize_logup_in_pairs (lib.rs:185-221).
    // It is wired now but is NOT trusted until the box accumulator-diff is zero
    // (Phase 2 validation).
    std::vector<unsigned> batching(logup_counts);
    for (unsigned i = 0; i < logup_counts; ++i) batching[i] = i / 2;
    unsigned last_batch = batching[logup_counts - 1];

    // ----- Tiled pre_kernel + post_kernel over eval-domain row tiles -----
    for (unsigned tile_start = 0; tile_start < eval_domain_size; tile_start += tile_rows) {
        unsigned this_tile = eval_domain_size - tile_start;
        if (this_tile > tile_rows) this_tile = tile_rows;

        // Bias the fraction pointer back by tile_start*logup_counts so the
        // evaluator's GLOBAL-row index (row*logup_counts) writes/reads into the
        // tile-local buffer slot (row-tile_start)*logup_counts. The kernels never
        // touch d_fractions outside [0, this_tile*logup_counts) because every
        // active thread has row in [tile_start, tile_start+this_tile).
        Fraction *frac_biased = d_fractions - (size_t)tile_start * logup_counts;

        // Select the tree0/tree1 pointer tables the pre_kernel dereferences. Under
        // tiled_input, build a PER-COLUMN biased pointer table. For each column c:
        //   * STAGED (host_traceX[c] != nullptr): H2D this block's slice
        //     [tile_start, tile_start+this_tile) of the committed host bytes into the
        //     reused tile buffer, then bias by -tile_start so the kernel's GLOBAL-row
        //     read trace_evaluations[c][row] lands at tile-local slot row-tile_start.
        //     The slice is contiguous because tree0/tree1 reads are pure pointwise
        //     [row]. Its trace_evaluations[c] device pointer is a FREED stash key and
        //     is NEVER dereferenced.
        //   * RESIDENT (host_traceX[c] == nullptr, or a wholly-null table): use the
        //     LIVE full-eval-domain device buffer trace_evaluations[c] with NO bias.
        //     The kernel reads col[global_row] and the resident buffer is already the
        //     full domain indexed by global row, so it must be passed unbiased (a
        //     -tile_start bias would read col[global-tile_start] = the WRONG row on
        //     every tile after the first). This matches the resident tree2 (d_trace2)
        //     pointers, which are likewise passed whole/unbiased. Only a STAGED tile
        //     buffer (local rows [0,this_tile)) needs -tile_start. No H2D, no re-staging.
        // Same biased-pointer trick as d_fractions above and the quotient Site-1 fix —
        // no eval_at_row.cuh change, byte-identical. FAIL LOUD if a staged column has a
        // tile buffer mismatch (never H2D into a null buffer / read a bad address).
        m31 **pre_trace0 = d_trace0;
        m31 **pre_trace1 = d_trace1;
        if (tiled_input) {
            std::vector<m31 *> biased0(trace0_evaluations_len);
            std::vector<m31 *> biased1(trace1_evaluations_len);
            for (unsigned c = 0; c < trace0_evaluations_len; ++c) {
                bool staged = (host_trace0 != nullptr) && (host_trace0[c] != nullptr);
                if (staged) {
                    if (tile0_bufs[c] == nullptr) {
                        printf("evaluate_gate_air: tree0 col %u staged but no tile buffer\n", c);
                        assert(false && "staged tree0 col missing tile buffer");
                    }
                    cuda_mem_copy_host_to_device<uint32_t>(
                        (uint32_t *)host_trace0[c] + tile_start,
                        (uint32_t *)tile0_bufs[c], this_tile);
                    biased0[c] = tile0_bufs[c] - (size_t)tile_start;
                } else {
                    // Resident: live full-eval-domain buffer, indexed by the kernel's
                    // GLOBAL row directly — NO bias (unlike a staged tile buffer, which
                    // holds only local rows [0,this_tile) and therefore needs -tile_start).
                    // Same convention as the resident tree2 (d_trace2) pointers below.
                    biased0[c] = trace0_evaluations[c];
                }
            }
            for (unsigned c = 0; c < trace1_evaluations_len; ++c) {
                bool staged = (host_trace1 != nullptr) && (host_trace1[c] != nullptr);
                if (staged) {
                    if (tile1_bufs[c] == nullptr) {
                        printf("evaluate_gate_air: tree1 col %u staged but no tile buffer\n", c);
                        assert(false && "staged tree1 col missing tile buffer");
                    }
                    cuda_mem_copy_host_to_device<uint32_t>(
                        (uint32_t *)host_trace1[c] + tile_start,
                        (uint32_t *)tile1_bufs[c], this_tile);
                    biased1[c] = tile1_bufs[c] - (size_t)tile_start;
                } else {
                    // Resident: live full-eval-domain buffer, indexed by the kernel's
                    // GLOBAL row directly — NO bias (unlike a staged tile buffer, which
                    // holds only local rows [0,this_tile) and therefore needs -tile_start).
                    // Same convention as the resident tree2 (d_trace2) pointers below.
                    biased1[c] = trace1_evaluations[c];
                }
            }
            cuda_mem_copy_host_to_device<m31 *>(biased0.data(), d_tile0_ptrs, trace0_evaluations_len);
            cuda_mem_copy_host_to_device<m31 *>(biased1.data(), d_tile1_ptrs, trace1_evaluations_len);
            pre_trace0 = d_tile0_ptrs;
            pre_trace1 = d_tile1_ptrs;
        }

        int block_dim = this_tile < GATE_AIR_THREAD_COUNT_MAX
            ? (int)this_tile : GATE_AIR_THREAD_COUNT_MAX;
        int num_blocks = (this_tile + block_dim - 1) / block_dim;

        // ----- pre_kernel (PHASE 1: algebraic + relation emits) -----
        if (use_assert_evaluator) {
            evaluate_gate_air_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, pre_trace0, pre_trace1, random_coeff_powers,
                domain_log_size, eval_domain_log_size, d_gate_eval, cumsum_shift,
                frac_biased, logup_counts, constraint_index_array,
                tile_start, this_tile);
        } else {
            evaluate_gate_air_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, pre_trace0, pre_trace1, random_coeff_powers,
                domain_log_size, eval_domain_log_size, d_gate_eval, cumsum_shift,
                frac_biased, logup_counts, constraint_index_array,
                tile_start, this_tile);
        }
        // STAGED path keeps the per-tile sync (its ping-pong H2D depends on it).
        // RESIDENT path drops the sync (same-stream ordering suffices) but keeps the
        // non-blocking launch-error check.
        if (!resident_path) {
            ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
        }
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        // ----- Under tree2_tiled: H2D this block's slice of every tree2 column (20
        // source + 4 shifted) into the reused tile buffers, then build the biased
        // pointer table. Same convention as tree0/1: a STAGED source col H2Ds from
        // host_trace2[c]; a RESIDENT source col H2Ds from its live device buffer via
        // D2D (kept contiguous so the tile slice is the row-block). The 4 shifted cols
        // always H2D from their host stash (shift_host). Every column read here is a
        // pure offset-0 [row] read, so the physical slice [tile_start, +this_tile) is
        // exactly the logical row-block and biasing by -tile_start is correct. -----
        m31 **post_trace2 = d_trace2;
        if (tree2_tiled) {
            std::vector<m31 *> biased2(len2 + N_SHIFT);
            const unsigned shift_first = (len2 >= N_SHIFT) ? (len2 - N_SHIFT) : 0;
            for (unsigned c = 0; c < len2; ++c) {
                if (host_trace2[c] != nullptr) {
                    cuda_mem_copy_host_to_device<uint32_t>(
                        (uint32_t *)host_trace2[c] + tile_start,
                        (uint32_t *)tile2_bufs[c], this_tile);
                } else {
                    // Resident source col: copy this block's slice from the live buffer.
                    cuda_mem_copy_device_to_device<m31>(
                        trace2_evaluations[c] + tile_start, tile2_bufs[c], this_tile);
                }
                biased2[c] = tile2_bufs[c] - (size_t)tile_start;
            }
            // 4 shifted cols appended at [len2, len2+N_SHIFT); source coords are
            // [shift_first, len2). shift_host[k] is the whole shifted column k.
            (void)shift_first;
            for (unsigned k = 0; k < N_SHIFT; ++k) {
                cuda_mem_copy_host_to_device<uint32_t>(
                    shift_host[k] + tile_start,
                    (uint32_t *)tile2_bufs[len2 + k], this_tile);
                biased2[len2 + k] = tile2_bufs[len2 + k] - (size_t)tile_start;
            }
            cuda_mem_copy_host_to_device<m31 *>(biased2.data(), d_tile2_ptrs, len2 + N_SHIFT);
            post_trace2 = d_tile2_ptrs;
        }

        // ----- post_kernel (PHASE 2 SOUNDNESS GATE: LogUp pair-batches) -----
        // Under tree2_tiled the post_kernel reads prev_row_cumsum from the appended
        // shifted coords at offset 0 (tree2_shifted=true); else tree2 is resident and
        // the scattered `-1` read runs (tree2_shifted=false) — byte-for-byte unchanged.
        if (use_assert_evaluator) {
            evaluate_gate_air_post_kernel_tiled<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, frac_biased, constraint_index_array, post_trace2,
                random_coeff_powers, domain_log_size, eval_domain_log_size,
                logup_counts, last_batch, cumsum_shift,
                tile_start, this_tile, tree2_tiled);
        } else {
            evaluate_gate_air_post_kernel_tiled<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, frac_biased, constraint_index_array, post_trace2,
                random_coeff_powers, domain_log_size, eval_domain_log_size,
                logup_counts, last_batch, cumsum_shift,
                tile_start, this_tile, tree2_tiled);
        }
        // STAGED path keeps the per-tile sync (its ping-pong H2D depends on it).
        // RESIDENT path drops the sync (same-stream ordering serializes this tile's
        // post before the next tile's pre / d_fractions reuse) but keeps the
        // non-blocking launch-error check. The single post-loop sync below covers all
        // resident-path kernels before any host read.
        if (!resident_path) {
            ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
        }
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }

    // DEFERRED ASYNC (correctness-first, scope §2): the per-block H2D above is
    // SYNCHRONOUS on the compute stream, so block b's transfer serializes before
    // block b's kernels. To overlap, wire the pinned + async primitives
    // (cuda_alloc_pinned_host_uint32_t / copy_uint32_t_vec_from_host_to_device_async
    // in utils.cu, exposed in bindings.rs) as a two-buffer ping-pong: while the
    // kernel computes block b from tile-buffer set A, H2D block b+1 into set B on a
    // dedicated copy stream, then swap (device cost doubles the tile-buffer term).
    // The stash host bytes must be pinned at dehydrate time for the async H2D to
    // overlap. Correctness and the residency win hold WITHOUT this (the ceiling
    // lift is the point); the overlap is a PCIe-latency optimization only.

    // ----- finalize: quotient = numerators[row] * denom_inv[row>>trace_log] -----
    // Pointwise per row over the FULL eval domain (the tiling above only bounded
    // the d_fractions transient and the tree0/tree1 input residency; numerators[]
    // is now fully populated). Identical to accumulate_pointwise_cpu
    // (component_prover.rs:288-289).
    int finalize_block_dim = eval_domain_size < GATE_AIR_THREAD_COUNT_MAX
        ? (int)eval_domain_size : GATE_AIR_THREAD_COUNT_MAX;
    int finalize_num_blocks = (eval_domain_size + finalize_block_dim - 1) / finalize_block_dim;
    generic_constraint_quotients_finalize_kernel<<<finalize_num_blocks, finalize_block_dim, 0, stream>>>(
        quotients_0, quotients_1, quotients_2, quotients_3,
        numerators, denominator_inverses,
        domain_log_size, eval_domain_log_size, should_accumulate);

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_gate_air");

    if (tiled_input) {
        // Only staged columns allocated a tile buffer; resident columns are null.
        for (unsigned c = 0; c < trace0_evaluations_len; ++c)
            if (tile0_bufs[c] != nullptr) cuda_free_memory(tile0_bufs[c]);
        for (unsigned c = 0; c < trace1_evaluations_len; ++c)
            if (tile1_bufs[c] != nullptr) cuda_free_memory(tile1_bufs[c]);
        std::free(tile0_bufs);
        std::free(tile1_bufs);
        cuda_free_memory(d_tile0_ptrs);
        cuda_free_memory(d_tile1_ptrs);
    } else {
        cuda_free_memory(d_trace0);
        cuda_free_memory(d_trace1);
    }
    if (tree2_tiled) {
        for (unsigned c = 0; c < len2 + N_SHIFT; ++c)
            if (tile2_bufs[c] != nullptr) cuda_free_memory(tile2_bufs[c]);
        std::free(tile2_bufs);
        cuda_free_memory(d_tile2_ptrs);
        for (unsigned k = 0; k < N_SHIFT; ++k)
            if (shift_host[k] != nullptr) std::free(shift_host[k]);
    } else {
        cuda_free_memory(d_trace2);
    }
    cuda_free_memory(numerators);
    cuda_free_memory(d_gate_eval);
    cuda_free_memory(d_fractions);
    cuda_free_memory(constraint_index_array);
}
