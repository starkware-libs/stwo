// ============================================================================
// gate_air MAIN component constraint kernel — PHASE 1 (algebraic core).
//
// !!! BOX-UNVALIDATED CUDA — cannot be compiled on the laptop (no nvcc). !!!
// Transcribed line-for-line from `impl FrameworkEval for GateEval`
// (gate-air-leaf/src/main.rs). Build + validate on the GPU box before trusting.
//
// WITNESS-SHRINK LAYOUT (matches main.rs ~lines 379-401, 778-797): the main
// (witness) trace is 188 columns. `enabler`, `shot_id`, `pc` (and the
// pre-existing `pc_in_prog`) live in the PREPROCESSED tree (tree0) and are read
// via eval0.get_preprocessed_column() in call order enabler/shot_id/pc/pc_in_prog.
// The main trace header is just the 4 opcode masks; all later main reads shift
// down by 3 vs the old 191-col layout. The main.rs line citations below were
// re-derived against the witness-shrunk source.
//
// Pipeline (mirror of evaluate_memory_address_to_id.cu):
//   1. pre_kernel  : per eval-domain row, run the 151 ALGEBRAIC add_constraints
//                    (-> numerators[row] = row_res) and emit the 12 LogUp
//                    relation entries (-> intermediate_fractions). PHASE 1.
//   2. post_kernel : generic_constraint_post_kernel (evaluate_common.cuh) folds
//                    the 12 fractions into 6 pair-batches and adds the 6 LogUp
//                    cumsum constraints. PHASE 2 SOUNDNESS GATE — wired here but
//                    NOT trusted until box accumulator-diff is zero.
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
// Relation slicing helper.
//
// gate_air draws ONE width-35 relation (`relation!(GateRel, 35)`, main.rs:110)
// shared by all five logical relations (state/qdecode/rc_lo/rc_hi/program); they
// are kept distinct only by the integer TAG prepended as values[0]
// (main.rs:120-142). Every emit therefore combines against the SAME (z, alpha,
// alpha_powers), using only the first N alpha_powers for an N-wide tuple — see
// the Rust `combine` (constraint-framework logup.rs:84-99), which folds
// `alpha_powers[0..values.len()]` and subtracts z.
//
// The `CudaEvaluator::add_to_relation<N>(RelationEntry<N>)` overload
// (eval_at_row.cuh:320) requires a `RelationEntry<N>`, whose `relation` field is
// a `LookupElementsBasic<N>`. The gate relation is `LookupElementsBasic<35>`, so
// for the N<35 entries (qdecode N=5, rc N=3, program N=6) we build the matching
// `LookupElementsBasic<N>` by copying z, alpha, and the first N alpha_powers.
// This yields bit-identical `combine` to the width-35 relation (the math only
// touches alpha_powers[0..N]) while making the template overload match for both
// CudaAssertEvaluator and CudaEvaluator. N==35 needs no slice.
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
// Per-read masks, in the EXACT order `read_masks` (main.rs:954-973) consumes
// columns from the main trace: q, limb_idx, bit_pos, mask, lsel[0..32], lo, hi,
// bit. 39 columns per read.
// ----------------------------------------------------------------------------
struct GateReadMasks {
    m31 q;
    m31 limb_idx;
    m31 bit_pos;
    m31 mask;
    m31 lsel[GATE_AIR_N_LIMBS];
    m31 lo;
    m31 hi;
    m31 bit;
};

// Mirror of `read_masks` (main.rs:954-973): pull 39 consecutive main-trace masks.
template<typename EvaluatorT>
DEVICE_FORCEINLINE GateReadMasks gate_read_masks(EvaluatorT &eval) {
    GateReadMasks r;
    r.q = eval.next_trace_mask();
    r.limb_idx = eval.next_trace_mask();
    r.bit_pos = eval.next_trace_mask();
    r.mask = eval.next_trace_mask();
    for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) {
        r.lsel[j] = eval.next_trace_mask();
    }
    r.lo = eval.next_trace_mask();
    r.hi = eval.next_trace_mask();
    r.bit = eval.next_trace_mask();
    return r;
}

// Mirror of `read_constraints` (main.rs:916-952). Emits, IN ORDER:
//   * 32 lsel booleanity constraints  s*(s-1)              (main.rs:924-926)
//   * sum(lsel) - active                                   (main.rs:934)
//   * sum(j*lsel_j) - limb_idx                             (main.rs:935)
//   * L - hi*mask*2 - bit*mask - lo, L = sum lsel_j*in_limb_j (main.rs:937-947)
//   * bit*(bit-1)                                          (main.rs:949)
//   * (1-active)*bit                                       (main.rs:951)
// => 37 add_constraints per read, matching the Rust order exactly.
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_read_constraints(
    EvaluatorT &eval,
    const GateReadMasks &r,
    m31 active,
    const m31 *in_limb
) {
    // lsel booleanity (32).
    for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) {
        m31 s = r.lsel[j];
        eval.add_constraint(mul(s, sub(s, m31(1))));
    }
    // sum lsel = active ; sum j*lsel_j = limb_idx.
    m31 sum = 0;
    m31 weighted = 0;
    for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) {
        sum = add(sum, r.lsel[j]);
        weighted = add(weighted, mul(r.lsel[j], m31((unsigned)j)));
    }
    eval.add_constraint(sub(sum, active));
    eval.add_constraint(sub(weighted, r.limb_idx));
    // selected limb L = sum lsel_j * in_limb_j.
    m31 l = 0;
    for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) {
        l = add(l, mul(r.lsel[j], in_limb[j]));
    }
    // L = hi*2*mask + bit*mask + lo  ->  L - hi*mask*2 - bit*mask - lo = 0.
    m31 hi_mask_two = mul(mul(r.hi, r.mask), m31(2));
    m31 bit_mask = mul(r.bit, r.mask);
    eval.add_constraint(sub(sub(sub(l, hi_mask_two), bit_mask), r.lo));
    // bit booleanity.
    eval.add_constraint(mul(r.bit, sub(r.bit, m31(1))));
    // inactive => bit forced 0:  (1 - active) * bit.
    eval.add_constraint(mul(sub(m31(1), active), r.bit));
}

// Mirror of `add_qdecode_lookup` (main.rs:975-995): emit relation entry
//   (TAG_QDECODE, q, limb_idx, bit_pos, mask) with multiplicity = active.
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_add_qdecode_lookup(
    EvaluatorT &eval,
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &relation,
    const GateReadMasks &r,
    m31 active
) {
    m31 values[5] = { m31(GATE_AIR_TAG_QDECODE), r.q, r.limb_idx, r.bit_pos, r.mask };
    qm31 mult = { { active, 0 }, { 0, 0 } };  // E::EF::from(active)
    RelationEntry<5> entry(gate_relation_slice<5>(relation), mult, values);
    eval.template add_to_relation<5>(entry);
}

// Mirror of `add_rc_lookup` (main.rs:997-1024): emit lo entry THEN hi entry
//   (TAG_RC_LO, bit_pos, lo) and (TAG_RC_HI, bit_pos, hi), multiplicity=active.
// Order is lo, then hi — this ordering is load-bearing (fraction index order).
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_add_rc_lookup(
    EvaluatorT &eval,
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &relation,
    const GateReadMasks &r,
    m31 active
) {
    qm31 mult = { { active, 0 }, { 0, 0 } };
    m31 lo_values[3] = { m31(GATE_AIR_TAG_RC_LO), r.bit_pos, r.lo };
    RelationEntry<3> lo_entry(gate_relation_slice<3>(relation), mult, lo_values);
    eval.template add_to_relation<3>(lo_entry);

    m31 hi_values[3] = { m31(GATE_AIR_TAG_RC_HI), r.bit_pos, r.hi };
    RelationEntry<3> hi_entry(gate_relation_slice<3>(relation), mult, hi_values);
    eval.template add_to_relation<3>(hi_entry);
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
    // WITNESS-SHRINK (main.rs:778-797). The Rust `evaluate()` now reads FOUR
    // preprocessed columns up front, in this EXACT call order:
    //     enabler, shot_id, pc, pc_in_prog
    // (= GateEval's `preprocessed_column_indices` order, which is the order the
    // InfoEvaluator records `get_preprocessed_column` calls; the CUDA component
    // prover gathers `trace0_evaluations` in precisely that index order — see
    // constraint-framework component.rs / component_prover.rs). eval0 therefore
    // reads them sequentially via col_index[0] in the SAME order. Only the 4
    // opcode masks (is_nop/is_not/is_cnot/is_toffoli) remain in the main trace
    // header; everything else shifts down by 3 vs the old 191-col layout.

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

    // --- Main-trace masks (188 cols), in declaration order (main.rs:783-797). ---
    // Header is now ONLY the 4 opcode masks (enabler/shot_id/pc moved to tree0).
    m31 is_nop     = eval.next_trace_mask();
    m31 is_not     = eval.next_trace_mask();
    m31 is_cnot    = eval.next_trace_mask();
    m31 is_toffoli = eval.next_trace_mask();

    m31 in_limb[GATE_AIR_N_LIMBS];
    for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) in_limb[j] = eval.next_trace_mask();
    m31 out_limb[GATE_AIR_N_LIMBS];
    for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) out_limb[j] = eval.next_trace_mask();

    GateReadMasks target = gate_read_masks(eval);
    GateReadMasks ctrl_a = gate_read_masks(eval);
    GateReadMasks ctrl_b = gate_read_masks(eval);

    m31 ab    = eval.next_trace_mask();
    m31 fire  = eval.next_trace_mask();
    m31 delta = eval.next_trace_mask();

    // ===================== ALGEBRAIC CONSTRAINTS (151) =====================
    // [1-4] opcode booleanity op*(op-1) for is_nop/is_not/is_cnot/is_toffoli (main.rs:790-792).
    eval.add_constraint(mul(is_nop,     sub(is_nop,     m31(1))));
    eval.add_constraint(mul(is_not,     sub(is_not,     m31(1))));
    eval.add_constraint(mul(is_cnot,    sub(is_cnot,    m31(1))));
    eval.add_constraint(mul(is_toffoli, sub(is_toffoli, m31(1))));
    // [5] one-hot sum = enabler: enabler - is_nop - is_not - is_cnot - is_toffoli (main.rs:793-799).
    eval.add_constraint(
        sub(sub(sub(sub(enabler, is_nop), is_not), is_cnot), is_toffoli)
    );

    // active sub-expressions (main.rs:801-802). NOT columns — computed inline.
    m31 a_active = add(is_cnot, is_toffoli);
    m31 b_active = is_toffoli;

    // [6-42]  target read block (active = enabler)        (main.rs:806).
    // [43-79] ctrl_a read block (active = a_active)        (main.rs:807).
    // [80-116] ctrl_b read block (active = b_active)       (main.rs:808).
    gate_read_constraints(eval, target, enabler,  in_limb);
    gate_read_constraints(eval, ctrl_a, a_active, in_limb);
    gate_read_constraints(eval, ctrl_b, b_active, in_limb);

    // --- Fire / delta logic. ---
    m31 a_bit = ctrl_a.bit;  // main.rs:811
    m31 b_bit = ctrl_b.bit;  // main.rs:812
    m31 t_bit = target.bit;  // main.rs:813

    // [117] ab - a_bit*b_bit (main.rs:815).
    eval.add_constraint(sub(ab, mul(a_bit, b_bit)));
    // [118] fire - is_not - is_cnot*a_bit - is_toffoli*ab (main.rs:817-822).
    eval.add_constraint(
        sub(sub(sub(fire, is_not), mul(is_cnot, a_bit)), mul(is_toffoli, ab))
    );
    // [119] delta - fire + 2*t_bit*fire (main.rs:825-827).
    //   delta - fire + t_bit*fire*2.
    eval.add_constraint(
        add(sub(delta, fire), mul(mul(t_bit, fire), m31(2)))
    );

    // [120-151] out_limb[j] - in_limb[j] - lsel_t[j]*delta*mask_t  (main.rs:830-836).
    for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) {
        m31 sel_term = mul(mul(target.lsel[j], delta), target.mask);
        eval.add_constraint(sub(sub(out_limb[j], in_limb[j]), sel_term));
    }

    // ===================== LOGUP RELATION ENTRIES (12) =====================
    // PHASE 1 emits the 12 entries (the post_kernel turns them into the 6 LogUp
    // constraints in PHASE 2). Order MUST match `evaluate()` exactly.
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &rel = gate_eval->relation;

    // 1. state input  (+enabler), 35-wide  (main.rs:839-856).
    //    [TAG_STATE, shot_id, pc, in_limb[0..32]]
    {
        m31 in_state[GATE_AIR_REL_WIDTH];
        in_state[0] = m31(GATE_AIR_TAG_STATE);
        in_state[1] = shot_id;
        in_state[2] = pc;
        for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) in_state[3 + j] = in_limb[j];
        qm31 mult = { { enabler, 0 }, { 0, 0 } };            // E::EF::from(enabler)
        RelationEntry<GATE_AIR_REL_WIDTH> entry(rel, mult, in_state);
        eval.template add_to_relation<GATE_AIR_REL_WIDTH>(entry);
    }
    // 2. state output (-enabler), 35-wide  (main.rs:845-861).
    //    [TAG_STATE, shot_id, pc+1, out_limb[0..32]]
    {
        m31 out_state[GATE_AIR_REL_WIDTH];
        out_state[0] = m31(GATE_AIR_TAG_STATE);
        out_state[1] = shot_id;
        out_state[2] = add(pc, m31(1));
        for (int j = 0; j < GATE_AIR_N_LIMBS; ++j) out_state[3 + j] = out_limb[j];
        m31 neg_enabler = neg(enabler);                       // -mult
        qm31 mult = { { neg_enabler, 0 }, { 0, 0 } };
        RelationEntry<GATE_AIR_REL_WIDTH> entry(rel, mult, out_state);
        eval.template add_to_relation<GATE_AIR_REL_WIDTH>(entry);
    }

    // 3,4,5. qdecode for target / ctrl_a / ctrl_b  (main.rs:864-866).
    gate_add_qdecode_lookup(eval, rel, target, enabler);
    gate_add_qdecode_lookup(eval, rel, ctrl_a, a_active);
    gate_add_qdecode_lookup(eval, rel, ctrl_b, b_active);

    // 6..11. rc (lo,hi) for target / ctrl_a / ctrl_b  (main.rs:869-871).
    // Each call emits TWO entries (lo then hi) -> 6 entries.
    gate_add_rc_lookup(eval, rel, target, enabler);
    gate_add_rc_lookup(eval, rel, ctrl_a, a_active);
    gate_add_rc_lookup(eval, rel, ctrl_b, b_active);

    // 12. program (+enabler)  (main.rs:882-897).
    //   opcode_scalar = is_not*1 + is_cnot*2 + is_toffoli*3.
    //   [TAG_PROGRAM, pc_in_prog, opcode_scalar, target.q, ctrl_a.q, ctrl_b.q]
    {
        m31 opcode_scalar = add(add(is_not, mul(is_cnot, m31(2))), mul(is_toffoli, m31(3)));
        m31 prog[6] = {
            m31(GATE_AIR_TAG_PROGRAM), pc_in_prog, opcode_scalar,
            target.q, ctrl_a.q, ctrl_b.q
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
    unsigned tile_rows
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

        int offsets2[2] = { 0, -1 };
        qm31 cumsum2[2] = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} } };
        evaluator.next_extension_interaction_mask(logup_interaction, offsets2, 2, cumsum2);

        const qm31 prev_row_cumsum = cumsum2[1];
        const qm31 cur_cumsum = cumsum2[0];
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
static unsigned gate_air_resolve_tile_rows() {
    unsigned tile_rows = 1u << 20;  // 2^20 default
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
    // (the 188 large tree1 eval columns, dehydrated by GATE_AIR_STREAM_COMMIT — their
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

    // Resident device pointer tables. Under tiled_input, tree0/tree1 tables are
    // REBUILT per block into d_tile{0,1}_ptrs (staged cols -> biased tile buffer,
    // resident cols -> biased live buffer), so we do not clone the whole
    // trace{0,1}_evaluations here (staged entries are freed stash keys). tree2 is
    // always resident.
    m31 **d_trace0 = tiled_input ? nullptr
        : clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **d_trace1 = tiled_input ? nullptr
        : clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **d_trace2 = clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    GateAirEval *d_gate_eval = cuda_malloc<GateAirEval>(1);
    cuda_mem_copy_host_to_device<GateAirEval>(gate_eval, d_gate_eval, 1);

    // ----- Row-tiling (COMPOSITION_TILING_SCOPE route c). -----
    // Steps 1-2 already row-tiled the d_fractions INTERMEDIATE (a ~12 GB @2^24
    // transient) with the same tile_start/this_tile passed to the kernels; the
    // kernels index every trace read by GLOBAL row and only the fraction pointer
    // was biased. This completes route (c): under tiled_input we ALSO row-tile the
    // 188 main + 4 preprocessed INPUT columns so their residency is
    // O(this_tile * (trace0_len + trace1_len)) instead of the full ~47 GB eval set.
    // The eval is pure pointwise for tree0/tree1 (offset 0, no next-row mask), so a
    // row-block is a contiguous committed-byte slice and biased-pointer indexing is
    // byte-trivially correct.
    unsigned tile_rows = gate_air_resolve_tile_rows();
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

    timer global_timer;
    global_timer.start("evaluate_gate_air");

    // gate_air uses `finalize_logup_in_pairs` => batching[i] = i/2 (6 pairs).
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
        ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        // ----- post_kernel (PHASE 2 SOUNDNESS GATE: LogUp pair-batches) -----
        // tree2 (d_trace2) is FULLY RESIDENT in both paths — byte-unchanged.
        if (use_assert_evaluator) {
            evaluate_gate_air_post_kernel_tiled<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, frac_biased, constraint_index_array, d_trace2,
                random_coeff_powers, domain_log_size, eval_domain_log_size,
                logup_counts, last_batch, cumsum_shift,
                tile_start, this_tile);
        } else {
            evaluate_gate_air_post_kernel_tiled<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, frac_biased, constraint_index_array, d_trace2,
                random_coeff_powers, domain_log_size, eval_domain_log_size,
                logup_counts, last_batch, cumsum_shift,
                tile_start, this_tile);
        }
        ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
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
    cuda_free_memory(d_trace2);
    cuda_free_memory(numerators);
    cuda_free_memory(d_gate_eval);
    cuda_free_memory(d_fractions);
    cuda_free_memory(constraint_index_array);
}
