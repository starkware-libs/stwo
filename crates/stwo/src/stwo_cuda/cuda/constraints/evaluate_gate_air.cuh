#ifndef EVALUATE_GATE_AIR_H
#define EVALUATE_GATE_AIR_H

// ============================================================================
// gate_air MAIN component constraint-evaluation kernel — PHASE 1 (algebraic).
//
// !!! BOX-UNVALIDATED CUDA !!!
// This file CANNOT be compiled on the laptop (no nvcc). It has been written by
// transcribing `impl FrameworkEval for GateEval` (gate-air-leaf/src/main.rs,
// `evaluate()` ~lines 762-901, `read_constraints` 916-952, `add_qdecode_lookup`
// 975-995, `add_rc_lookup` 997-1024) line-for-line into the NitrooZK
// `CudaEvaluator` mirror (eval_at_row.cuh). It MUST be built + validated on the
// GPU box (nvcc / libstwo_cuda) before it is trusted. See the report for the
// exact box validation plan.
//
// SCOPE (Phase 1): emit GateEval's ~151 ALGEBRAIC constraints (degree <= 3) in
// the EXACT order `evaluate()` pushes them, accumulating
//   row_res += random_coeff_powers[constraint_index++] * constraint_i
// (mirror of CpuDomainEvaluator::add_constraint, cpu_domain.rs:88-94) and emit
// the 12 LogUp relation entries into `intermediate_fractions` (so the Phase-2
// post/finalize kernels can consume them). The generic post_kernel
// (evaluate_common.cuh) then adds the 6 LogUp pair-batch constraints — that part
// is the PHASE 2 soundness gate (see report); Phase 1 validates the algebraic
// core + column order via `use_assert_evaluator=true`.
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

// gate_air structural constants (must equal the Rust constants in gate-air-leaf
// src/main.rs: N_LIMBS=32, READ_COLS=39, TRACE_COLUMNS=188, GATE_REL_WIDTH=35,
// and the TAG_* relation ids).
//
// WITNESS-SHRINK (matches main.rs ~lines 379-401 + 778-797): enabler, shot_id, pc
// were MOVED OUT of the main (witness) trace into the PREPROCESSED tree (tree0),
// alongside pc_in_prog. The main trace header is now just the 4 opcode masks
// (is_nop/is_not/is_cnot/is_toffoli). So TRACE_COLUMNS = 4 + 32 + 32 + 3*39 + 3
// = 188 (was 191), and GateEval reads FOUR preprocessed columns up front.
#define GATE_AIR_N_LIMBS 32
#define GATE_AIR_READ_COLS 39          // 4 + N_LIMBS + 3
#define GATE_AIR_TRACE_COLUMNS 188     // 4 + 32 + 32 + 3*39 + 3 (header = 4 opcode masks)
#define GATE_AIR_N_PREPROCESSED 4      // enabler, shot_id, pc, pc_in_prog (tree0, call order)
#define GATE_AIR_REL_WIDTH 35          // 1 + STATE_WIDTH (STATE_WIDTH = 2 + N_LIMBS)
#define GATE_AIR_LOGUP_COUNTS 12       // 12 relation entries -> 6 pairs
#define GATE_AIR_N_ALGEBRAIC 151       // algebraic add_constraint count (sanity)

#define GATE_AIR_TAG_STATE 1
#define GATE_AIR_TAG_QDECODE 2
#define GATE_AIR_TAG_RC_LO 3
#define GATE_AIR_TAG_RC_HI 4
#define GATE_AIR_TAG_PROGRAM 5

// Host-side `eval` struct passed through the generic FFI `void *eval` arg. The
// first field MUST be `eval_id` (see CommonEval in evaluate_constraints.cuh): the
// dispatcher reads `((CommonEval*)eval)->eval_id` to select this kernel. The
// remaining fields carry the single drawn LogUp relation `(z, alpha, alpha^i)`,
// shared by all five logical relations (state/qdecode/rc_lo/rc_hi/program), which
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
    // (GATE_AIR_STREAM_COMMIT). Both non-null => row-tile the 188 main + 4
    // preprocessed INPUT columns, H2D per block from these host bases into reused
    // tile buffers (residency O(tile_rows*(len0+len1)) instead of the full eval
    // set). Both null => legacy resident path, BYTE-FOR-BYTE unchanged. tree2 is
    // always resident (its post_kernel `-1` offset is a scattered index — no tile).
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
);

#endif // EVALUATE_GATE_AIR_H
