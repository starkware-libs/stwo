#include "evaluate_constraints.cuh"
#include "evaluate_common.cuh"
#include "evaluate_wide_fibonacci.cuh"
#include "evaluate_poseidon_constraint.cuh"
#include "timer.cuh"

#include <vector>
#include <algorithm>

// Include stwo-cairo components
// Global variable to control accumulation behavior (declared extern in evaluate_common.cuh)
// This is set by the dispatcher before each component evaluation
// Note: Not thread-safe across streams, but each stream writes to its own buffers
#include "evaluate_blake_compress_opcode.cuh"
#include "evaluate_blake_g.cuh"
#include "evaluate_triple_xor_32.cuh"
// Verify / bitwise XOR components
#include "evaluate_verify_bitwise_xor_4.cuh"
#include "evaluate_verify_bitwise_xor_7.cuh"
#include "evaluate_verify_bitwise_xor_8.cuh"
#include "evaluate_verify_bitwise_xor_8_b.cuh"
#include "evaluate_verify_bitwise_xor_9.cuh"
#include "evaluate_verify_bitwise_xor_12.cuh"
#include "evaluate_verify_instruction.cuh"
#include "evaluate_add_mod_builtin.cuh"
#include "evaluate_mul_mod_builtin.cuh"
#include "evaluate_cube_252.cuh"
#include "evaluate_poseidon_builtin.cuh"
#include "evaluate_pedersen_builtin.cuh"
#include "evaluate_pedersen_builtin_narrow_windows.cuh"
#include "evaluate_bitwise_builtin.cuh"
// Pedersen context components
#include "constraints/pedersen/evaluate_partial_ec_mul.cuh"
#include "constraints/pedersen/evaluate_pedersen_points_table.cuh"
#include "constraints/pedersen/evaluate_pedersen_aggregator.cuh"
#include "constraints/pedersen/evaluate_partial_ec_mul_window_bits_18.cuh"
#include "constraints/pedersen/evaluate_pedersen_points_table_window_bits_18.cuh"
#include "constraints/pedersen/evaluate_pedersen_aggregator_window_bits_9.cuh"
#include "constraints/pedersen/evaluate_partial_ec_mul_window_bits_9.cuh"
#include "constraints/pedersen/evaluate_pedersen_points_table_window_bits_9.cuh"
// Poseidon context components
#include "constraints/poseidon/evaluate_poseidon_aggregator.cuh"
#include "constraints/poseidon/evaluate_poseidon_3_partial_rounds_chain.cuh"
#include "constraints/poseidon/evaluate_poseidon_full_round_chain.cuh"
#include "constraints/poseidon/evaluate_poseidon_round_keys.cuh"
// Range-check components
#include "evaluate_range_check_3_3_3_3_3.cuh"
#include "evaluate_range_check_3_6_6_3.cuh"
#include "evaluate_range_check_4_3.cuh"
#include "evaluate_range_check_4_4.cuh"
#include "evaluate_range_check_4_4_4_4.cuh"
#include "evaluate_range_check_6.cuh"
#include "evaluate_range_check_8.cuh"
#include "evaluate_range_check_9_9.cuh"
#include "evaluate_range_check_9_9_b.cuh"
#include "evaluate_range_check_9_9_c.cuh"
#include "evaluate_range_check_9_9_d.cuh"
#include "evaluate_range_check_9_9_e.cuh"
#include "evaluate_range_check_9_9_f.cuh"
#include "evaluate_range_check_9_9_g.cuh"
#include "evaluate_range_check_9_9_h.cuh"
#include "evaluate_range_check_11.cuh"
#include "evaluate_range_check_12.cuh"
#include "evaluate_range_check_18.cuh"
#include "evaluate_range_check_18_b.cuh"
// 19-bit ranges (legacy)
#include "evaluate_range_check_19.cuh"
#include "evaluate_range_check_19_b.cuh"
#include "evaluate_range_check_19_c.cuh"
#include "evaluate_range_check_19_d.cuh"
#include "evaluate_range_check_19_e.cuh"
#include "evaluate_range_check_19_f.cuh"
#include "evaluate_range_check_19_g.cuh"
#include "evaluate_range_check_19_h.cuh"
// 20-bit range (replaces 19-bit in "now")
#include "evaluate_range_check_20.cuh"
// Builtin range-checks
#include "evaluate_range_check_builtin_bits_96.cuh"
#include "evaluate_range_check_builtin_bits_128.cuh"
#include "evaluate_range_check_felt_252_width_27.cuh"
#include "evaluate_range_check_7_2_5.cuh"
#include "evaluate_blake_round_sigma.cuh"
#include "evaluate_memory_address_to_id.cuh"
#include "evaluate_memory_id_to_big.cuh"
#include "evaluate_blake_round.cuh"
#include "evaluate_add_opcode.cuh"
#include "evaluate_add_ap_opcode.cuh"
#include "evaluate_add_opcode_small.cuh"
#include "evaluate_assert_eq_opcode.cuh"
#include "evaluate_assert_eq_opcode_double_deref.cuh"
#include "evaluate_assert_eq_opcode_imm.cuh"
#include "evaluate_call_opcode.cuh"
#include "evaluate_call_opcode_rel_imm.cuh"
#include "evaluate_ret_opcode.cuh"
#include "evaluate_jump_opcode.cuh"
#include "evaluate_jump_opcode_double_deref.cuh"
#include "evaluate_jump_opcode_rel_imm.cuh"
#include "evaluate_jump_opcode_rel.cuh"
#include "evaluate_jnz_opcode.cuh"
#include "evaluate_jnz_opcode_non_taken.cuh"
#include "evaluate_jnz_opcode_taken.cuh"
#include "evaluate_mul_opcode.cuh"
#include "evaluate_mul_opcode_small.cuh"
#include "evaluate_generic_opcode.cuh"
#include "evaluate_qm_31_add_mul_opcode.cuh"

bool g_should_accumulate_host = true;

// Internal dispatch: evaluates a single component, writing to the given quotient buffers
// on the given stream. The should_accumulate flag controls overwrite vs accumulate.
// Returns true if the component was handled, false if unsupported (caller should fall back to CPU).
static bool dispatch_single_eval(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
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
    // Set global flag for this evaluation
    g_should_accumulate_host = should_accumulate;

    CommonEval *common_eval = (CommonEval *) eval;
    const unsigned eval_id = common_eval->eval_id;

#define DISPATCH_EVAL_SIMPLE(TAG, MSG, FN) \
    case fnv1a_eval_id_gen(TAG): { \
        FN( \
            quotients_0, quotients_1, quotients_2, quotients_3, \
            trace0_evaluations, \
            trace0_evaluations_len, \
            trace1_evaluations, \
            trace1_evaluations_len, \
            random_coeff_powers, \
            denominator_inverses, \
            domain_log_size, \
            eval_domain_log_size, \
            number_of_columns, \
            logup_counts \
        ); \
        return true; \
    }

#define DISPATCH_EVAL_WITH_BOOLS(TAG, MSG, FN) \
    case fnv1a_eval_id_gen(TAG): { \
        FN( \
            quotients_0, quotients_1, quotients_2, quotients_3, \
            trace0_evaluations, \
            trace0_evaluations_len, \
            trace1_evaluations, \
            trace1_evaluations_len, \
            trace2_evaluations, \
            trace2_evaluations_len, \
            random_coeff_powers, \
            denominator_inverses, \
            domain_log_size, \
            eval_domain_log_size, \
            number_of_columns, \
            logup_counts, \
            eval, \
            cumsum_shift, \
            should_accumulate, \
            use_assert_evaluator, \
            stream \
        ); \
        return true; \
    }

#define DISPATCH_EVAL_WITH_CUMSUM(TAG, MSG, FN) \
    case fnv1a_eval_id_gen(TAG): { \
        FN( \
            quotients_0, quotients_1, quotients_2, quotients_3, \
            trace0_evaluations, \
            trace0_evaluations_len, \
            trace1_evaluations, \
            trace1_evaluations_len, \
            trace2_evaluations, \
            trace2_evaluations_len, \
            random_coeff_powers, \
            denominator_inverses, \
            domain_log_size, \
            eval_domain_log_size, \
            number_of_columns, \
            logup_counts, \
            eval, \
            cumsum_shift \
        ); \
        return true; \
    }

    switch (eval_id) {
        DISPATCH_EVAL_SIMPLE("fibonacci_example", "call cuda eval wide fib example", evaluate_wide_fibonacci_constraint_quotients_on_domain);
        DISPATCH_EVAL_WITH_BOOLS("poseidon_example", "call cuda eval poseidon example", evaluate_poseidon_constraint_quotients_on_domain);

        // stwo-cairo components
        DISPATCH_EVAL_WITH_BOOLS("blake_compress_opcode", "call cuda eval stwo-cairo blake_compress_opcode", evaluate_blake_compress_opcode);
        DISPATCH_EVAL_WITH_BOOLS("blake_g", "call cuda eval stwo-cairo blake_g", evaluate_blake_g);
        DISPATCH_EVAL_WITH_BOOLS("triple_xor_32", "call cuda eval stwo-cairo triple_xor_32", evaluate_triple_xor_32);
        DISPATCH_EVAL_WITH_BOOLS("verify_bitwise_xor_4", "call cuda eval stwo-cairo verify_bitwise_xor_4", evaluate_verify_bitwise_xor_4);
        DISPATCH_EVAL_WITH_BOOLS("verify_bitwise_xor_7", "call cuda eval stwo-cairo verify_bitwise_xor_7", evaluate_verify_bitwise_xor_7);
        DISPATCH_EVAL_WITH_BOOLS("verify_bitwise_xor_8", "call cuda eval stwo-cairo verify_bitwise_xor_8", evaluate_verify_bitwise_xor_8);
        DISPATCH_EVAL_WITH_BOOLS("verify_bitwise_xor_8_b", "call cuda eval stwo-cairo verify_bitwise_xor_8_b", evaluate_verify_bitwise_xor_8_b);
        DISPATCH_EVAL_WITH_BOOLS("verify_bitwise_xor_9", "call cuda eval stwo-cairo verify_bitwise_xor_9", evaluate_verify_bitwise_xor_9);
        DISPATCH_EVAL_WITH_BOOLS("verify_bitwise_xor_12", "call cuda eval stwo-cairo verify_bitwise_xor_12", evaluate_verify_bitwise_xor_12);

        // Simple range checks
        DISPATCH_EVAL_WITH_BOOLS("range_check_3_3_3_3_3", "call cuda eval stwo-cairo range_check_3_3_3_3_3", evaluate_range_check_3_3_3_3_3);
        DISPATCH_EVAL_WITH_BOOLS("range_check_3_6_6_3", "call cuda eval stwo-cairo range_check_3_6_6_3", evaluate_range_check_3_6_6_3);
        DISPATCH_EVAL_WITH_BOOLS("range_check_4_3", "call cuda eval stwo-cairo range_check_4_3", evaluate_range_check_4_3);
        DISPATCH_EVAL_WITH_BOOLS("range_check_4_4", "call cuda eval stwo-cairo range_check_4_4", evaluate_range_check_4_4);
        DISPATCH_EVAL_WITH_BOOLS("range_check_4_4_4_4", "call cuda eval stwo-cairo range_check_4_4_4_4", evaluate_range_check_4_4_4_4);
        DISPATCH_EVAL_WITH_BOOLS("range_check_6", "call cuda eval stwo-cairo range_check_6", evaluate_range_check_6);
        DISPATCH_EVAL_WITH_BOOLS("range_check_8", "call cuda eval stwo-cairo range_check_8", evaluate_range_check_8);
        DISPATCH_EVAL_WITH_BOOLS("range_check_9_9", "call cuda eval stwo-cairo range_check_9_9", evaluate_range_check_9_9);
        DISPATCH_EVAL_WITH_BOOLS("range_check_9_9_b", "call cuda eval stwo-cairo range_check_9_9_b", evaluate_range_check_9_9_b);
        DISPATCH_EVAL_WITH_BOOLS("range_check_9_9_c", "call cuda eval stwo-cairo range_check_9_9_c", evaluate_range_check_9_9_c);
        DISPATCH_EVAL_WITH_BOOLS("range_check_9_9_d", "call cuda eval stwo-cairo range_check_9_9_d", evaluate_range_check_9_9_d);
        DISPATCH_EVAL_WITH_BOOLS("range_check_9_9_e", "call cuda eval stwo-cairo range_check_9_9_e", evaluate_range_check_9_9_e);
        DISPATCH_EVAL_WITH_BOOLS("range_check_9_9_f", "call cuda eval stwo-cairo range_check_9_9_f", evaluate_range_check_9_9_f);
        DISPATCH_EVAL_WITH_BOOLS("range_check_9_9_g", "call cuda eval stwo-cairo range_check_9_9_g", evaluate_range_check_9_9_g);
        DISPATCH_EVAL_WITH_BOOLS("range_check_9_9_h", "call cuda eval stwo-cairo range_check_9_9_h", evaluate_range_check_9_9_h);
        DISPATCH_EVAL_WITH_BOOLS("range_check_11", "call cuda eval stwo-cairo range_check_11", evaluate_range_check_11);
        DISPATCH_EVAL_WITH_BOOLS("range_check_12", "call cuda eval stwo-cairo range_check_12", evaluate_range_check_12);
        DISPATCH_EVAL_WITH_BOOLS("range_check_18", "call cuda eval stwo-cairo range_check_18", evaluate_range_check_18);
        DISPATCH_EVAL_WITH_BOOLS("range_check_18_b", "call cuda eval stwo-cairo range_check_18_b", evaluate_range_check_18_b);
        DISPATCH_EVAL_WITH_BOOLS("range_check_19", "call cuda eval stwo-cairo range_check_19", evaluate_range_check_19);
        DISPATCH_EVAL_WITH_BOOLS("range_check_19_b", "call cuda eval stwo-cairo range_check_19_b", evaluate_range_check_19_b);
        DISPATCH_EVAL_WITH_BOOLS("range_check_19_c", "call cuda eval stwo-cairo range_check_19_c", evaluate_range_check_19_c);
        DISPATCH_EVAL_WITH_BOOLS("range_check_19_d", "call cuda eval stwo-cairo range_check_19_d", evaluate_range_check_19_d);
        DISPATCH_EVAL_WITH_BOOLS("range_check_19_e", "call cuda eval stwo-cairo range_check_19_e", evaluate_range_check_19_e);
        DISPATCH_EVAL_WITH_BOOLS("range_check_19_f", "call cuda eval stwo-cairo range_check_19_f", evaluate_range_check_19_f);
        DISPATCH_EVAL_WITH_BOOLS("range_check_19_g", "call cuda eval stwo-cairo range_check_19_g", evaluate_range_check_19_g);
        DISPATCH_EVAL_WITH_BOOLS("range_check_19_h", "call cuda eval stwo-cairo range_check_19_h", evaluate_range_check_19_h);
        // range_check_20: 8 variants (replaces range_check_19 in "now")
        // All range_check_20 variants use the same kernel (evaluate_range_check_20) since the
        // constraint logic is identical for single-column range checks - only the domain size differs.
        DISPATCH_EVAL_WITH_BOOLS("range_check_20", "call cuda eval stwo-cairo range_check_20", evaluate_range_check_20);
        DISPATCH_EVAL_WITH_BOOLS("range_check_20_b", "call cuda eval stwo-cairo range_check_20_b", evaluate_range_check_20);
        DISPATCH_EVAL_WITH_BOOLS("range_check_20_c", "call cuda eval stwo-cairo range_check_20_c", evaluate_range_check_20);
        DISPATCH_EVAL_WITH_BOOLS("range_check_20_d", "call cuda eval stwo-cairo range_check_20_d", evaluate_range_check_20);
        DISPATCH_EVAL_WITH_BOOLS("range_check_20_e", "call cuda eval stwo-cairo range_check_20_e", evaluate_range_check_20);
        DISPATCH_EVAL_WITH_BOOLS("range_check_20_f", "call cuda eval stwo-cairo range_check_20_f", evaluate_range_check_20);
        DISPATCH_EVAL_WITH_BOOLS("range_check_20_g", "call cuda eval stwo-cairo range_check_20_g", evaluate_range_check_20);
        DISPATCH_EVAL_WITH_BOOLS("range_check_20_h", "call cuda eval stwo-cairo range_check_20_h", evaluate_range_check_20);
        DISPATCH_EVAL_WITH_BOOLS("range_check_builtin_bits_96", "call cuda eval stwo-cairo range_check_builtin_bits_96", evaluate_range_check_builtin_bits_96);
        DISPATCH_EVAL_WITH_BOOLS("range_check_builtin_bits_128", "call cuda eval stwo-cairo range_check_builtin_bits_128", evaluate_range_check_builtin_bits_128);
        DISPATCH_EVAL_WITH_BOOLS("range_check_builtin", "call cuda eval stwo-cairo range_check_builtin", evaluate_range_check_builtin_bits_128);
        DISPATCH_EVAL_WITH_BOOLS("range_check_felt_252_width_27", "call cuda eval stwo-cairo range_check_felt_252_width_27", evaluate_range_check_felt_252_width_27);
        DISPATCH_EVAL_WITH_BOOLS("range_check_252_width_27", "call cuda eval stwo-cairo range_check_252_width_27", evaluate_range_check_felt_252_width_27);
        DISPATCH_EVAL_WITH_BOOLS("range_check_7_2_5", "call cuda eval stwo-cairo range_check_7_2_5", evaluate_range_check_7_2_5);

        DISPATCH_EVAL_WITH_BOOLS("blake_round_sigma", "call cuda eval stwo-cairo blake_round_sigma", evaluate_blake_round_sigma);
        DISPATCH_EVAL_WITH_BOOLS("memory_address_to_id", "call cuda eval stwo-cairo memory_address_to_id", evaluate_memory_address_to_id);
        DISPATCH_EVAL_WITH_BOOLS("memory_id_to_big_big_eval", "call cuda eval stwo-cairo memory_id_to_big_big_eval", evaluate_memory_id_to_big_big);
        DISPATCH_EVAL_WITH_BOOLS("memory_id_to_big_small_eval", "call cuda eval stwo-cairo memory_id_to_big_small_eval", evaluate_memory_id_to_big_small);
        DISPATCH_EVAL_WITH_BOOLS("blake_round", "call cuda eval stwo-cairo blake_round", evaluate_blake_round);
        DISPATCH_EVAL_WITH_BOOLS("add_opcode", "call cuda eval stwo-cairo add_opcode", evaluate_add_opcode);
        DISPATCH_EVAL_WITH_BOOLS("add_ap_opcode", "call cuda eval stwo-cairo add_ap_opcode", evaluate_add_ap_opcode);
        DISPATCH_EVAL_WITH_BOOLS("add_opcode_small", "call cuda eval stwo-cairo add_opcode_small", evaluate_add_opcode_small);
        DISPATCH_EVAL_WITH_BOOLS("assert_eq_opcode", "call cuda eval stwo-cairo assert_eq_opcode", evaluate_assert_eq_opcode);
        DISPATCH_EVAL_WITH_BOOLS("assert_eq_opcode_double_deref", "call cuda eval stwo-cairo assert_eq_opcode_double_deref", evaluate_assert_eq_opcode_double_deref);
        DISPATCH_EVAL_WITH_BOOLS("assert_eq_opcode_imm", "call cuda eval stwo-cairo assert_eq_opcode_imm", evaluate_assert_eq_opcode_imm);
        DISPATCH_EVAL_WITH_BOOLS("call_opcode", "call cuda eval stwo-cairo call_opcode", evaluate_call_opcode);
        DISPATCH_EVAL_WITH_BOOLS("call_opcode_abs", "call cuda eval stwo-cairo call_opcode_abs", evaluate_call_opcode);
        DISPATCH_EVAL_WITH_BOOLS("call_opcode_rel_imm", "call cuda eval stwo-cairo call_opcode_rel_imm", evaluate_call_opcode_rel_imm);
        DISPATCH_EVAL_WITH_BOOLS("ret_opcode", "call cuda eval stwo-cairo ret_opcode", evaluate_ret_opcode);
        DISPATCH_EVAL_WITH_BOOLS("jump_opcode", "call cuda eval stwo-cairo jump_opcode", evaluate_jump_opcode);
        DISPATCH_EVAL_WITH_BOOLS("jump_opcode_abs", "call cuda eval stwo-cairo jump_opcode_abs", evaluate_jump_opcode);
        DISPATCH_EVAL_WITH_BOOLS("jump_opcode_double_deref", "call cuda eval stwo-cairo jump_opcode_double_deref", evaluate_jump_opcode_double_deref);
        DISPATCH_EVAL_WITH_BOOLS("jump_opcode_rel_imm", "call cuda eval stwo-cairo jump_opcode_rel_imm", evaluate_jump_opcode_rel_imm);
        DISPATCH_EVAL_WITH_BOOLS("jump_opcode_rel", "call cuda eval stwo-cairo jump_opcode_rel", evaluate_jump_opcode_rel);
        DISPATCH_EVAL_WITH_BOOLS("jnz_opcode", "call cuda eval stwo-cairo jnz_opcode", evaluate_jnz_opcode);
        DISPATCH_EVAL_WITH_BOOLS("jnz_opcode_non_taken", "call cuda eval stwo-cairo jnz_opcode_non_taken", evaluate_jnz_opcode_non_taken);
        DISPATCH_EVAL_WITH_BOOLS("jnz_opcode_taken", "call cuda eval stwo-cairo jnz_opcode_taken", evaluate_jnz_opcode_taken);
        DISPATCH_EVAL_WITH_BOOLS("mul_opcode", "call cuda eval stwo-cairo mul_opcode", evaluate_mul_opcode);
        DISPATCH_EVAL_WITH_BOOLS("mul_opcode_small", "call cuda eval stwo-cairo mul_opcode_small", evaluate_mul_opcode_small);
        DISPATCH_EVAL_WITH_BOOLS("generic_opcode", "call cuda eval stwo-cairo generic_opcode", evaluate_generic_opcode);
        DISPATCH_EVAL_WITH_BOOLS("qm_31_add_mul_opcode", "call cuda eval stwo-cairo qm_31_add_mul_opcode", evaluate_qm_31_add_mul_opcode);
        DISPATCH_EVAL_WITH_BOOLS("verify_instruction", "call cuda eval stwo-cairo verify_instruction", evaluate_verify_instruction);
        DISPATCH_EVAL_WITH_BOOLS("add_mod_builtin", "call cuda eval stwo-cairo add_mod_builtin", evaluate_add_mod_builtin);
        DISPATCH_EVAL_WITH_BOOLS("mul_mod_builtin", "call cuda eval stwo-cairo mul_mod_builtin", evaluate_mul_mod_builtin);
        DISPATCH_EVAL_WITH_BOOLS("cube_252", "call cuda eval stwo-cairo cube_252", evaluate_cube_252);
        DISPATCH_EVAL_WITH_BOOLS("poseidon_builtin", "call cuda eval stwo-cairo poseidon_builtin", evaluate_poseidon_builtin);
        DISPATCH_EVAL_WITH_BOOLS("pedersen_builtin", "call cuda eval stwo-cairo pedersen_builtin", evaluate_pedersen_builtin);
        DISPATCH_EVAL_WITH_BOOLS("pedersen_builtin_narrow_windows", "call cuda eval stwo-cairo pedersen_builtin_narrow_windows", evaluate_pedersen_builtin_narrow_windows);
        DISPATCH_EVAL_WITH_BOOLS("bitwise_builtin", "call cuda eval stwo-cairo bitwise_builtin", evaluate_bitwise_builtin);
        // Pedersen context components
        DISPATCH_EVAL_WITH_BOOLS("partial_ec_mul", "call cuda eval stwo-cairo partial_ec_mul", evaluate_partial_ec_mul);
        DISPATCH_EVAL_WITH_BOOLS("pedersen_points_table", "call cuda eval stwo-cairo pedersen_points_table", evaluate_pedersen_points_table);
        DISPATCH_EVAL_WITH_BOOLS("pedersen_aggregator_window_bits_18", "call cuda eval stwo-cairo pedersen_aggregator_window_bits_18", evaluate_pedersen_aggregator_window_bits_18);
        DISPATCH_EVAL_WITH_BOOLS("partial_ec_mul_window_bits_18", "call cuda eval stwo-cairo partial_ec_mul_window_bits_18", evaluate_partial_ec_mul_window_bits_18);
        DISPATCH_EVAL_WITH_BOOLS("pedersen_points_table_window_bits_18", "call cuda eval stwo-cairo pedersen_points_table_window_bits_18", evaluate_pedersen_points_table_window_bits_18);
        // Pedersen window_bits_9 context components
        DISPATCH_EVAL_WITH_BOOLS("pedersen_aggregator_window_bits_9", "call cuda eval stwo-cairo pedersen_aggregator_window_bits_9", evaluate_pedersen_aggregator_window_bits_9);
        DISPATCH_EVAL_WITH_BOOLS("partial_ec_mul_window_bits_9", "call cuda eval stwo-cairo partial_ec_mul_window_bits_9", evaluate_partial_ec_mul_window_bits_9);
        DISPATCH_EVAL_WITH_BOOLS("pedersen_points_table_window_bits_9", "call cuda eval stwo-cairo pedersen_points_table_window_bits_9", evaluate_pedersen_points_table_window_bits_9);
        // Poseidon context components
        DISPATCH_EVAL_WITH_BOOLS("poseidon_3_partial_rounds_chain", "call cuda eval stwo-cairo poseidon_3_partial_rounds_chain", evaluate_poseidon_3_partial_rounds_chain);
        DISPATCH_EVAL_WITH_BOOLS("poseidon_full_round_chain", "call cuda eval stwo-cairo poseidon_full_round_chain", evaluate_poseidon_full_round_chain);
        DISPATCH_EVAL_WITH_BOOLS("poseidon_round_keys", "call cuda eval stwo-cairo poseidon_round_keys", evaluate_poseidon_round_keys);
        DISPATCH_EVAL_WITH_BOOLS("poseidon_aggregator", "call cuda eval stwo-cairo poseidon_aggregator", evaluate_poseidon_aggregator);

        default:
            fprintf(stderr, "CUDA dispatch: eval id:%u not supported, falling back to CPU\n", eval_id);
            fflush(stderr);
            return false;
    }

#undef DISPATCH_EVAL_SIMPLE
#undef DISPATCH_EVAL_WITH_BOOLS
#undef DISPATCH_EVAL_WITH_CUMSUM
    return true;
}

// Original sequential entry point (backward compatible)
// Returns true if the component was handled by CUDA, false if unsupported (caller should
// fall back to CPU evaluation).
bool evaluate_constraint_quotients_on_domain(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,
    bool use_assert_evaluator
) {
    // Delegate to internal dispatch with default stream (0)
    return dispatch_single_eval(
        quotients_0, quotients_1, quotients_2, quotients_3,
        trace0_evaluations, trace0_evaluations_len,
        trace1_evaluations, trace1_evaluations_len,
        trace2_evaluations, trace2_evaluations_len,
        random_coeff_powers, denominator_inverses,
        domain_log_size, eval_domain_log_size,
        number_of_columns, logup_counts,
        eval, cumsum_shift,
        should_accumulate, use_assert_evaluator,
        0  // default stream
    );
}
