#ifndef EVALUATE_GATE_AIR_H
#define EVALUATE_GATE_AIR_H

// ============================================================================
// gate_air MAIN component constraint-evaluation kernel — PHASE 1 (algebraic).
//
// !!! BOX-UNVALIDATED CUDA !!!
// This file CANNOT be compiled on the laptop (no nvcc). It has been written by
// transcribing `impl FrameworkEval for GateEval` (gate-air-leaf/src/main.rs,
// `evaluate()`, `access_masks`, `add_qubitmem_pair`, `add_rc_lookup`,
// `add_ts_range`) line-for-line (qubit-memory + ts=pc+1-inlined + single-`d`
// rc-table encoding) into the NitrooZK `CudaEvaluator` mirror (eval_at_row.cuh).
// It MUST be built + validated on the GPU box (nvcc / libstwo_cuda) before it is
// trusted. See the report for the exact box validation plan.
//
// SCOPE (Phase 1): emit GateEval's 15 ALGEBRAIC constraints (degree <= 3) in
// the EXACT order `evaluate()` pushes them, accumulating
//   row_res += random_coeff_powers[constraint_index++] * constraint_i
// (mirror of CpuDomainEvaluator::add_constraint, cpu_domain.rs:88-94) and emit
// the 10 LogUp relation entries into `intermediate_fractions` (so the Phase-2
// post/finalize kernels can consume them). The generic post_kernel
// (evaluate_common.cuh) then adds the 5 LogUp batch constraints (all pairs) —
// that part is the PHASE 2 soundness gate (see report); Phase 1
// validates the algebraic core + column order via `use_assert_evaluator=true`.
// (ts = pc+1 inlined + single-`d` rc-table range-check: per active access ONE ts
// constraint, the RANGE recon `active*((pc+1)-prev_ts-1 - d)`, plus 1 rc LOOKUP
// (TAG_RC, d). The old PIN `active*(ts-(pc*3+slot))` is GONE (ts is structurally
// pc+1), and the target `v_after` equality is GONE (v_after = v_before+delta
// inlined). So algebraic 19->15 (-3 PIN, -1 v_after equality); LogUp entries are 10
// (3 qubitmem pairs + 3 rc-d + 1 program).)
//
// The accumulation tail (numerators[row] -> quotient via denom_inv, written into
// the 4 accumulator coord columns) is the generic finalize kernel, identical to
// every other component and to `accumulate_pointwise_cpu`
// (component_prover.rs:264-292):
//   res[row] = accum[row] + row_res * denom_inv[row >> trace_log_size].
// ============================================================================

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"

// gate_air structural constants — QUBIT-MEMORY + ts=pc+1 (inlined) + single-`d` rc-table
// (branch anatg/gate-air-qubit-mem). Must equal the Rust constants in gate-air-leaf src/main.rs:
//   ACCESS_COLS=3, ACCESS_BLOCK=4, TRACE_COLUMNS=19, GATE_REL_WIDTH=6, and the TAG_* ids.
//   (ts = pc+1, no per-gate slot; not a column. The two rc limbs collapsed to a single diff col `d`.)
//
// The old whole-state (188-col TAG_STATE) encoding is replaced by a per-qubit chain-lookup
// qubit-memory. Each access block is ACCESS_BLOCK = ACCESS_COLS(3) + 1 rc diff col `d` = 4. Main
// (witness) trace = 19 columns (cell_at order). ts (= pc+1) and the target's v_after (= v_before+delta)
// are INLINED, NOT columns:
//   [0..4)   is_nop, is_not, is_cnot, is_toffoli
//   [4..8)   target access: addr, prev_ts, v_before, d   (ACCESS_BLOCK)
//   [8..12)  ctrl_a access:  addr, prev_ts, v, d          (ACCESS_BLOCK)
//   [12..16) ctrl_b access:  addr, prev_ts, v, d          (ACCESS_BLOCK)
//   [16..19) ab, fire, delta
// enabler/shot_id/pc/pc_in_prog stay in the PREPROCESSED tree (tree0), read up front in that
// call order — GATE_AIR_N_PREPROCESSED = 4, unchanged. `pc` now FEEDS the inlined ts = pc+1 (Yield
// tuple + RANGE recon). The relation is width 6 (widest tuple = program = tag + 5 payload). TAG_RC.
#define GATE_AIR_ACCESS_COLS 3         // addr, prev_ts, v (core access cols read by gate_access_masks; ts inlined=pc+1)
#define GATE_AIR_ACCESS_BLOCK 4        // ACCESS_COLS + 1 rc diff col `d`
#define GATE_AIR_TRACE_COLUMNS 19      // 4 + ACCESS_BLOCK + ACCESS_BLOCK + ACCESS_BLOCK + 3
#define GATE_AIR_N_PREPROCESSED 4      // enabler, shot_id, pc, pc_in_prog (tree0, call order)
#define GATE_AIR_REL_WIDTH 6           // tag + widest payload (program: pc_in_prog,op,3 addrs = 5)
// ts = pc+1 (inlined) + single-`d` rc-table range-check: per active access the ts-ordering is ONE
// RANGE reconstruction `active*((pc+1)-prev_ts-1 - d)` (1 algebraic constraint/access) plus 1 rc
// LOOKUP (TAG_RC, d). The old PIN is gone (ts structural) and the v_after equality is gone
// (v_after=v_before+delta inlined). So: LOGUP_COUNTS 10 (3 qubitmem pairs + 3 rc-d + 1 program)
// => 5 batches (all pairs), and N_ALGEBRAIC 19->15 (-3 PIN, -1 v_after eq).
#define GATE_AIR_LOGUP_COUNTS 10       // 10 relation entries -> 5 batches (all pairs)
#define GATE_AIR_N_ALGEBRAIC 15        // algebraic add_constraint count (sanity): 4+1+4+1+1+1 + 1*3
// ts-ordering constant (main.rs): ts = pc + 1 (inlined); d = (pc+1) - prev_ts - 1 = pc - prev_ts.

#define GATE_AIR_TAG_QUBITMEM 1
#define GATE_AIR_TAG_RC 2
#define GATE_AIR_TAG_PROGRAM 5

// Host-side `eval` struct passed through the generic FFI `void *eval` arg. The
// first field MUST be `eval_id` (see CommonEval in evaluate_constraints.cuh): the
// dispatcher reads `((CommonEval*)eval)->eval_id` to select this kernel. The
// remaining fields carry the single drawn LogUp relation `(z, alpha, alpha^i)`,
// shared by the three logical relations (qubitmem/rc/program), which
// are separated only by the integer TAG as tuple element 0 — see main.rs:120-142
// (`LookupElements::draw` clones one `GateRel`). The Rust side populates this via
// `extract_z_alpha` (gpu_tracegen.rs:1060), guaranteeing identical challenges to
// the trace-gen / interaction kernels.
struct GateAirEval {
    unsigned eval_id;
    unsigned log_n_rows;
    LookupElementsBasic<GATE_AIR_REL_WIDTH> relation;
};

extern "C"
void evaluate_gate_air(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    // COMPOSITION_TILING_SCOPE (route c) host-tile-source pointers for tree0/tree1
    // (GATE_AIR_STREAM_COMMIT). Both non-null => row-tile the 19 main + 4
    // preprocessed INPUT columns, H2D per block from these host bases into reused
    // tile buffers (residency O(tile_rows*(len0+len1)) instead of the full eval
    // set). Both null => legacy resident path, BYTE-FOR-BYTE unchanged. tree2 is
    // always resident (its post_kernel `-1` offset is a scattered index — no tile).
    const uint32_t * const *host_trace0,
    const uint32_t * const *host_trace1,
    // F2-b / Option B: per-column host-tile-source table for tree2 (interaction).
    // Non-null => row-tile tree2 like tree0/1 AND precompute the 4 shifted last-LogUp
    // cumsum coords (interaction_shift_neg1) so the composition post_kernel reads
    // prev_row_cumsum at offset 0 (no scattered `-1`, tree2 no longer held whole).
    // Null => tree2 resident + legacy scattered `-1` read (BYTE-FOR-BYTE).
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
);

#endif // EVALUATE_GATE_AIR_H
