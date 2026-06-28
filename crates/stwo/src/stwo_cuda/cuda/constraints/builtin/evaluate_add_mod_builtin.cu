// ============================================================================
// AddMod Builtin CUDA Evaluator
// ============================================================================
//
// CUDA version of AddMod builtin AIR constraint evaluator
// translated from cairo-air/src/components/add_mod_builtin.rs
//
// ## Functionality Overview
// AddMod builtin verifies 256-bit modular addition operations: a + b ≡ c (mod p)
//
// ## Data Structure
// - 267 trace columns:
// - p0-p3: modulus p as 4 64-bit words (each with 11 limbs)
// - a0-a3: operand a as 4 64-bit words
// - b0-b3: operand b as 4 64-bit words
// - c0-c3: result c as 4 64-bit words
// - sub_p_bit: flag indicating whether p needs to be subtracted (0 or 1)
// - carry_0 to carry_13: 14 carry values (each ∈ {-1, 0, 1})
// - other: memory address, offset and other auxiliary columns
//
// ## Constraint Logic
// 1. ModUtils constraints: verify memory consistency and structure of p/a/b/c
// 2. sub_p_bit constraint: verify sub_p_bit ∈ {0, 1}
// 3. 14 carry constraints: verify correctness of limb-by-limb addition
// 4. final constraint: verify last carry is 0
//
// ## Relation Lookups
// - MemoryAddressToId: 29 lookups
// - MemoryIdToBig: 24 lookups
//
// ============================================================================

#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_add_mod_builtin.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_read_small.cuh"
#include "evaluate_mem_verify.cuh"
#include "evaluate_common.cuh"
#include "mod_utils_common.cuh"

#define ADD_MOD_BUILTIN_THREAD_COUNT_MAX 256

// ============================================================================
// AddMod Pre-Kernel: main constraint evaluation
// ============================================================================
// This kernel is responsible for:
// 1. Reading all 267 trace columns
// 2. Calling ModUtils to verify per-row memory and structure
// 3. Evaluating sub_p_bit and 14 carry constraints
// 4. Accumulating all constraints into numerators
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_add_mod_builtin_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    AddModBuiltin_Eval *add_mod_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) {
        return;
    }

    // Evaluator for preprocessed trace (trace0)
    EvaluatorT cuda_evaluator0(
        trace0_evaluations,
        random_coeff_powers,
        0,
        row,
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // Read preprocessed column (Seq)
    m31 seq = cuda_evaluator0.next_trace_mask();

    // Evaluator for base trace (trace1)
    EvaluatorT cuda_evaluator1(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // Read base trace columns (267 columns)
    m31 is_instance_0_col0 = cuda_evaluator1.next_trace_mask();
    m31 p0_id_col1 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_0_col2 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_1_col3 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_2_col4 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_3_col5 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_4_col6 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_5_col7 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_6_col8 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_7_col9 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_8_col10 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_9_col11 = cuda_evaluator1.next_trace_mask();
    m31 p0_limb_10_col12 = cuda_evaluator1.next_trace_mask();
    m31 p1_id_col13 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_0_col14 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_1_col15 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_2_col16 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_3_col17 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_4_col18 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_5_col19 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_6_col20 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_7_col21 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_8_col22 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_9_col23 = cuda_evaluator1.next_trace_mask();
    m31 p1_limb_10_col24 = cuda_evaluator1.next_trace_mask();
    m31 p2_id_col25 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_0_col26 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_1_col27 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_2_col28 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_3_col29 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_4_col30 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_5_col31 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_6_col32 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_7_col33 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_8_col34 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_9_col35 = cuda_evaluator1.next_trace_mask();
    m31 p2_limb_10_col36 = cuda_evaluator1.next_trace_mask();
    m31 p3_id_col37 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_0_col38 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_1_col39 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_2_col40 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_3_col41 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_4_col42 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_5_col43 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_6_col44 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_7_col45 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_8_col46 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_9_col47 = cuda_evaluator1.next_trace_mask();
    m31 p3_limb_10_col48 = cuda_evaluator1.next_trace_mask();
    m31 values_ptr_id_col49 = cuda_evaluator1.next_trace_mask();
    m31 values_ptr_limb_0_col50 = cuda_evaluator1.next_trace_mask();
    m31 values_ptr_limb_1_col51 = cuda_evaluator1.next_trace_mask();
    m31 values_ptr_limb_2_col52 = cuda_evaluator1.next_trace_mask();
    m31 values_ptr_limb_3_col53 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col54 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_id_col55 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_limb_0_col56 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_limb_1_col57 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_limb_2_col58 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_limb_3_col59 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col60 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_prev_id_col61 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_prev_limb_0_col62 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_prev_limb_1_col63 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_prev_limb_2_col64 = cuda_evaluator1.next_trace_mask();
    m31 offsets_ptr_prev_limb_3_col65 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col66 = cuda_evaluator1.next_trace_mask();
    m31 n_id_col67 = cuda_evaluator1.next_trace_mask();
    m31 n_limb_0_col68 = cuda_evaluator1.next_trace_mask();
    m31 n_limb_1_col69 = cuda_evaluator1.next_trace_mask();
    m31 n_limb_2_col70 = cuda_evaluator1.next_trace_mask();
    m31 n_limb_3_col71 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col72 = cuda_evaluator1.next_trace_mask();
    m31 n_prev_id_col73 = cuda_evaluator1.next_trace_mask();
    m31 n_prev_limb_0_col74 = cuda_evaluator1.next_trace_mask();
    m31 n_prev_limb_1_col75 = cuda_evaluator1.next_trace_mask();
    m31 n_prev_limb_2_col76 = cuda_evaluator1.next_trace_mask();
    m31 n_prev_limb_3_col77 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col78 = cuda_evaluator1.next_trace_mask();
    m31 values_ptr_prev_id_col79 = cuda_evaluator1.next_trace_mask();
    m31 p_prev0_id_col80 = cuda_evaluator1.next_trace_mask();
    m31 p_prev1_id_col81 = cuda_evaluator1.next_trace_mask();
    m31 p_prev2_id_col82 = cuda_evaluator1.next_trace_mask();
    m31 p_prev3_id_col83 = cuda_evaluator1.next_trace_mask();
    m31 offsets_a_id_col84 = cuda_evaluator1.next_trace_mask();
    m31 msb_col85 = cuda_evaluator1.next_trace_mask();
    m31 mid_limbs_set_col86 = cuda_evaluator1.next_trace_mask();
    m31 offsets_a_limb_0_col87 = cuda_evaluator1.next_trace_mask();
    m31 offsets_a_limb_1_col88 = cuda_evaluator1.next_trace_mask();
    m31 offsets_a_limb_2_col89 = cuda_evaluator1.next_trace_mask();
    m31 remainder_bits_col90 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col91 = cuda_evaluator1.next_trace_mask();
    m31 offsets_b_id_col92 = cuda_evaluator1.next_trace_mask();
    m31 msb_col93 = cuda_evaluator1.next_trace_mask();
    m31 mid_limbs_set_col94 = cuda_evaluator1.next_trace_mask();
    m31 offsets_b_limb_0_col95 = cuda_evaluator1.next_trace_mask();
    m31 offsets_b_limb_1_col96 = cuda_evaluator1.next_trace_mask();
    m31 offsets_b_limb_2_col97 = cuda_evaluator1.next_trace_mask();
    m31 remainder_bits_col98 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col99 = cuda_evaluator1.next_trace_mask();
    m31 offsets_c_id_col100 = cuda_evaluator1.next_trace_mask();
    m31 msb_col101 = cuda_evaluator1.next_trace_mask();
    m31 mid_limbs_set_col102 = cuda_evaluator1.next_trace_mask();
    m31 offsets_c_limb_0_col103 = cuda_evaluator1.next_trace_mask();
    m31 offsets_c_limb_1_col104 = cuda_evaluator1.next_trace_mask();
    m31 offsets_c_limb_2_col105 = cuda_evaluator1.next_trace_mask();
    m31 remainder_bits_col106 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col107 = cuda_evaluator1.next_trace_mask();
    m31 a0_id_col108 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_0_col109 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_1_col110 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_2_col111 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_3_col112 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_4_col113 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_5_col114 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_6_col115 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_7_col116 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_8_col117 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_9_col118 = cuda_evaluator1.next_trace_mask();
    m31 a0_limb_10_col119 = cuda_evaluator1.next_trace_mask();
    m31 a1_id_col120 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_0_col121 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_1_col122 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_2_col123 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_3_col124 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_4_col125 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_5_col126 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_6_col127 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_7_col128 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_8_col129 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_9_col130 = cuda_evaluator1.next_trace_mask();
    m31 a1_limb_10_col131 = cuda_evaluator1.next_trace_mask();
    m31 a2_id_col132 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_0_col133 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_1_col134 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_2_col135 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_3_col136 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_4_col137 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_5_col138 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_6_col139 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_7_col140 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_8_col141 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_9_col142 = cuda_evaluator1.next_trace_mask();
    m31 a2_limb_10_col143 = cuda_evaluator1.next_trace_mask();
    m31 a3_id_col144 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_0_col145 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_1_col146 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_2_col147 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_3_col148 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_4_col149 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_5_col150 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_6_col151 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_7_col152 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_8_col153 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_9_col154 = cuda_evaluator1.next_trace_mask();
    m31 a3_limb_10_col155 = cuda_evaluator1.next_trace_mask();
    m31 b0_id_col156 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_0_col157 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_1_col158 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_2_col159 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_3_col160 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_4_col161 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_5_col162 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_6_col163 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_7_col164 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_8_col165 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_9_col166 = cuda_evaluator1.next_trace_mask();
    m31 b0_limb_10_col167 = cuda_evaluator1.next_trace_mask();
    m31 b1_id_col168 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_0_col169 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_1_col170 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_2_col171 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_3_col172 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_4_col173 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_5_col174 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_6_col175 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_7_col176 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_8_col177 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_9_col178 = cuda_evaluator1.next_trace_mask();
    m31 b1_limb_10_col179 = cuda_evaluator1.next_trace_mask();
    m31 b2_id_col180 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_0_col181 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_1_col182 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_2_col183 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_3_col184 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_4_col185 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_5_col186 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_6_col187 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_7_col188 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_8_col189 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_9_col190 = cuda_evaluator1.next_trace_mask();
    m31 b2_limb_10_col191 = cuda_evaluator1.next_trace_mask();
    m31 b3_id_col192 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_0_col193 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_1_col194 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_2_col195 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_3_col196 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_4_col197 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_5_col198 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_6_col199 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_7_col200 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_8_col201 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_9_col202 = cuda_evaluator1.next_trace_mask();
    m31 b3_limb_10_col203 = cuda_evaluator1.next_trace_mask();
    m31 c0_id_col204 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_0_col205 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_1_col206 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_2_col207 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_3_col208 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_4_col209 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_5_col210 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_6_col211 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_7_col212 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_8_col213 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_9_col214 = cuda_evaluator1.next_trace_mask();
    m31 c0_limb_10_col215 = cuda_evaluator1.next_trace_mask();
    m31 c1_id_col216 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_0_col217 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_1_col218 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_2_col219 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_3_col220 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_4_col221 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_5_col222 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_6_col223 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_7_col224 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_8_col225 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_9_col226 = cuda_evaluator1.next_trace_mask();
    m31 c1_limb_10_col227 = cuda_evaluator1.next_trace_mask();
    m31 c2_id_col228 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_0_col229 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_1_col230 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_2_col231 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_3_col232 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_4_col233 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_5_col234 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_6_col235 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_7_col236 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_8_col237 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_9_col238 = cuda_evaluator1.next_trace_mask();
    m31 c2_limb_10_col239 = cuda_evaluator1.next_trace_mask();
    m31 c3_id_col240 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_0_col241 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_1_col242 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_2_col243 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_3_col244 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_4_col245 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_5_col246 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_6_col247 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_7_col248 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_8_col249 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_9_col250 = cuda_evaluator1.next_trace_mask();
    m31 c3_limb_10_col251 = cuda_evaluator1.next_trace_mask();
    m31 sub_p_bit_col252 = cuda_evaluator1.next_trace_mask();
    m31 carry_0_col253 = cuda_evaluator1.next_trace_mask();
    m31 carry_1_col254 = cuda_evaluator1.next_trace_mask();
    m31 carry_2_col255 = cuda_evaluator1.next_trace_mask();
    m31 carry_3_col256 = cuda_evaluator1.next_trace_mask();
    m31 carry_4_col257 = cuda_evaluator1.next_trace_mask();
    m31 carry_5_col258 = cuda_evaluator1.next_trace_mask();
    m31 carry_6_col259 = cuda_evaluator1.next_trace_mask();
    m31 carry_7_col260 = cuda_evaluator1.next_trace_mask();
    m31 carry_8_col261 = cuda_evaluator1.next_trace_mask();
    m31 carry_9_col262 = cuda_evaluator1.next_trace_mask();
    m31 carry_10_col263 = cuda_evaluator1.next_trace_mask();
    m31 carry_11_col264 = cuda_evaluator1.next_trace_mask();
    m31 carry_12_col265 = cuda_evaluator1.next_trace_mask();
    m31 carry_13_col266 = cuda_evaluator1.next_trace_mask();

    // ===================== ModUtils call =====================
    // Call CUDA version ModUtils::evaluate, verify memory consistency and structure of p/a/b/c.
    // This part processes all constraints related to memory lookups and big number verification.
    mod_utils_evaluate<EvaluatorT>(
        m31(add_mod_eval->Claim.add_mod_builtin_segment_start),
        seq,
        is_instance_0_col0,
        p0_id_col1,
        p0_limb_0_col2,
        p0_limb_1_col3,
        p0_limb_2_col4,
        p0_limb_3_col5,
        p0_limb_4_col6,
        p0_limb_5_col7,
        p0_limb_6_col8,
        p0_limb_7_col9,
        p0_limb_8_col10,
        p0_limb_9_col11,
        p0_limb_10_col12,
        p1_id_col13,
        p1_limb_0_col14,
        p1_limb_1_col15,
        p1_limb_2_col16,
        p1_limb_3_col17,
        p1_limb_4_col18,
        p1_limb_5_col19,
        p1_limb_6_col20,
        p1_limb_7_col21,
        p1_limb_8_col22,
        p1_limb_9_col23,
        p1_limb_10_col24,
        p2_id_col25,
        p2_limb_0_col26,
        p2_limb_1_col27,
        p2_limb_2_col28,
        p2_limb_3_col29,
        p2_limb_4_col30,
        p2_limb_5_col31,
        p2_limb_6_col32,
        p2_limb_7_col33,
        p2_limb_8_col34,
        p2_limb_9_col35,
        p2_limb_10_col36,
        p3_id_col37,
        p3_limb_0_col38,
        p3_limb_1_col39,
        p3_limb_2_col40,
        p3_limb_3_col41,
        p3_limb_4_col42,
        p3_limb_5_col43,
        p3_limb_6_col44,
        p3_limb_7_col45,
        p3_limb_8_col46,
        p3_limb_9_col47,
        p3_limb_10_col48,
        values_ptr_id_col49,
        values_ptr_limb_0_col50,
        values_ptr_limb_1_col51,
        values_ptr_limb_2_col52,
        values_ptr_limb_3_col53,
        partial_limb_msb_col54,
        offsets_ptr_id_col55,
        offsets_ptr_limb_0_col56,
        offsets_ptr_limb_1_col57,
        offsets_ptr_limb_2_col58,
        offsets_ptr_limb_3_col59,
        partial_limb_msb_col60,
        offsets_ptr_prev_id_col61,
        offsets_ptr_prev_limb_0_col62,
        offsets_ptr_prev_limb_1_col63,
        offsets_ptr_prev_limb_2_col64,
        offsets_ptr_prev_limb_3_col65,
        partial_limb_msb_col66,
        n_id_col67,
        n_limb_0_col68,
        n_limb_1_col69,
        n_limb_2_col70,
        n_limb_3_col71,
        partial_limb_msb_col72,
        n_prev_id_col73,
        n_prev_limb_0_col74,
        n_prev_limb_1_col75,
        n_prev_limb_2_col76,
        n_prev_limb_3_col77,
        partial_limb_msb_col78,
        values_ptr_prev_id_col79,
        p_prev0_id_col80,
        p_prev1_id_col81,
        p_prev2_id_col82,
        p_prev3_id_col83,
        offsets_a_id_col84,
        msb_col85,
        mid_limbs_set_col86,
        offsets_a_limb_0_col87,
        offsets_a_limb_1_col88,
        offsets_a_limb_2_col89,
        remainder_bits_col90,
        partial_limb_msb_col91,
        offsets_b_id_col92,
        msb_col93,
        mid_limbs_set_col94,
        offsets_b_limb_0_col95,
        offsets_b_limb_1_col96,
        offsets_b_limb_2_col97,
        remainder_bits_col98,
        partial_limb_msb_col99,
        offsets_c_id_col100,
        msb_col101,
        mid_limbs_set_col102,
        offsets_c_limb_0_col103,
        offsets_c_limb_1_col104,
        offsets_c_limb_2_col105,
        remainder_bits_col106,
        partial_limb_msb_col107,
        a0_id_col108,
        a0_limb_0_col109,
        a0_limb_1_col110,
        a0_limb_2_col111,
        a0_limb_3_col112,
        a0_limb_4_col113,
        a0_limb_5_col114,
        a0_limb_6_col115,
        a0_limb_7_col116,
        a0_limb_8_col117,
        a0_limb_9_col118,
        a0_limb_10_col119,
        a1_id_col120,
        a1_limb_0_col121,
        a1_limb_1_col122,
        a1_limb_2_col123,
        a1_limb_3_col124,
        a1_limb_4_col125,
        a1_limb_5_col126,
        a1_limb_6_col127,
        a1_limb_7_col128,
        a1_limb_8_col129,
        a1_limb_9_col130,
        a1_limb_10_col131,
        a2_id_col132,
        a2_limb_0_col133,
        a2_limb_1_col134,
        a2_limb_2_col135,
        a2_limb_3_col136,
        a2_limb_4_col137,
        a2_limb_5_col138,
        a2_limb_6_col139,
        a2_limb_7_col140,
        a2_limb_8_col141,
        a2_limb_9_col142,
        a2_limb_10_col143,
        a3_id_col144,
        a3_limb_0_col145,
        a3_limb_1_col146,
        a3_limb_2_col147,
        a3_limb_3_col148,
        a3_limb_4_col149,
        a3_limb_5_col150,
        a3_limb_6_col151,
        a3_limb_7_col152,
        a3_limb_8_col153,
        a3_limb_9_col154,
        a3_limb_10_col155,
        b0_id_col156,
        b0_limb_0_col157,
        b0_limb_1_col158,
        b0_limb_2_col159,
        b0_limb_3_col160,
        b0_limb_4_col161,
        b0_limb_5_col162,
        b0_limb_6_col163,
        b0_limb_7_col164,
        b0_limb_8_col165,
        b0_limb_9_col166,
        b0_limb_10_col167,
        b1_id_col168,
        b1_limb_0_col169,
        b1_limb_1_col170,
        b1_limb_2_col171,
        b1_limb_3_col172,
        b1_limb_4_col173,
        b1_limb_5_col174,
        b1_limb_6_col175,
        b1_limb_7_col176,
        b1_limb_8_col177,
        b1_limb_9_col178,
        b1_limb_10_col179,
        b2_id_col180,
        b2_limb_0_col181,
        b2_limb_1_col182,
        b2_limb_2_col183,
        b2_limb_3_col184,
        b2_limb_4_col185,
        b2_limb_5_col186,
        b2_limb_6_col187,
        b2_limb_7_col188,
        b2_limb_8_col189,
        b2_limb_9_col190,
        b2_limb_10_col191,
        b3_id_col192,
        b3_limb_0_col193,
        b3_limb_1_col194,
        b3_limb_2_col195,
        b3_limb_3_col196,
        b3_limb_4_col197,
        b3_limb_5_col198,
        b3_limb_6_col199,
        b3_limb_7_col200,
        b3_limb_8_col201,
        b3_limb_9_col202,
        b3_limb_10_col203,
        c0_id_col204,
        c0_limb_0_col205,
        c0_limb_1_col206,
        c0_limb_2_col207,
        c0_limb_3_col208,
        c0_limb_4_col209,
        c0_limb_5_col210,
        c0_limb_6_col211,
        c0_limb_7_col212,
        c0_limb_8_col213,
        c0_limb_9_col214,
        c0_limb_10_col215,
        c1_id_col216,
        c1_limb_0_col217,
        c1_limb_1_col218,
        c1_limb_2_col219,
        c1_limb_3_col220,
        c1_limb_4_col221,
        c1_limb_5_col222,
        c1_limb_6_col223,
        c1_limb_7_col224,
        c1_limb_8_col225,
        c1_limb_9_col226,
        c1_limb_10_col227,
        c2_id_col228,
        c2_limb_0_col229,
        c2_limb_1_col230,
        c2_limb_2_col231,
        c2_limb_3_col232,
        c2_limb_4_col233,
        c2_limb_5_col234,
        c2_limb_6_col235,
        c2_limb_7_col236,
        c2_limb_8_col237,
        c2_limb_9_col238,
        c2_limb_10_col239,
        c3_id_col240,
        c3_limb_0_col241,
        c3_limb_1_col242,
        c3_limb_2_col243,
        c3_limb_3_col244,
        c3_limb_4_col245,
        c3_limb_5_col246,
        c3_limb_6_col247,
        c3_limb_7_col248,
        c3_limb_8_col249,
        c3_limb_9_col250,
        c3_limb_10_col251,
        add_mod_eval->common_lookup_elements,
        &cuda_evaluator1
    );

    // ===================== AddMod core constraint: carry propagation and sub_p_bit =====================
    //
    // AddMod constraints verify modular arithmetic: a + b ≡ c (mod p), optionally subtracting p.
    // Core idea:
    // - sub_p_bit ∈ {0, 1} indicates whether p needs to be subtracted
    // - Constraint verifies: a + b - c - sub_p_bit * p = 0 (expanded by limb)
    // - Uses 14 carry values to handle carries/borrows between limbs
    // - Each carry ∈ {-1, 0, 1}, guaranteed by constraint carry * (carry^2 - 1) = 0
    //
    // Constant explanations:
    // M31_16 = 2^4: for carry scaling in 9-bit limb groups (3+3+3 bits)
    // M31_128 = 2^7: for carry scaling across word boundaries
    // M31_512 = 2^9: for offset within 9-bit limb groups
    // M31_32768 = 2^15: for high-bit limb offset across words
    // M31_262144 = 2^18: for highest-bit offset in 9-bit limb groups (3rd limb)
    // M31_64 = 2^6: for special boundary situations

    const m31 M31_1      = m31(1);
    const m31 M31_16     = m31(16);
    const m31 M31_64     = m31(64);
    const m31 M31_128    = m31(128);
    const m31 M31_512    = m31(512);
    const m31 M31_32768  = m31(32768);
    const m31 M31_262144 = m31(262144);

    // sub_p_bit must be 0 or 1
    // constraint: sub_p_bit * (sub_p_bit - 1) = 0
    cuda_evaluator1.add_constraint(
        mul(sub(sub_p_bit_col252, M31_1), sub_p_bit_col252)
    );

    // ===================== carry_0 constraint =====================
    // Process first 3 limbs of a0/b0/c0/p0 (limb_0, limb_1, limb_2)
    // Formula: carry_0 = ((a0[0] + b0[0] - c0[0] - p0[0]*sub_p_bit)
    //                  + 512 * (a0[1] + b0[1] - c0[1] - p0[1]*sub_p_bit)
    //                  + 262144 * (a0[2] + b0[2] - c0[2] - p0[2]*sub_p_bit)) / 16
    // carry_0 range check: carry_0 * (carry_0^2 - 1) = 0, guarantees carry_0 ∈ {-1, 0, 1}
    {
        m31 s = sub(
            sub(
                add(a0_limb_0_col109, b0_limb_0_col157),
                c0_limb_0_col205
            ),
            mul(p0_limb_0_col2, sub_p_bit_col252)
        );
        m31 t1 = sub(
            sub(
                add(a0_limb_1_col110, b0_limb_1_col158),
                c0_limb_1_col206
            ),
            mul(p0_limb_1_col3, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a0_limb_2_col111, b0_limb_2_col159),
                c0_limb_2_col207
            ),
            mul(p0_limb_2_col4, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_0_col253, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_0_col253,
                sub(mul(carry_0_col253, carry_0_col253), M31_1))
        );
    }

    // ===================== carry_1 constraint =====================
    // Process limb_3, limb_4, limb_5 of a0/b0/c0/p0, includes carry from carry_0
    {
        m31 s = add(
            carry_0_col253,
            sub(
                sub(
                    add(a0_limb_3_col112, b0_limb_3_col160),
                    c0_limb_3_col208
                ),
                mul(p0_limb_3_col5, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a0_limb_4_col113, b0_limb_4_col161),
                c0_limb_4_col209
            ),
            mul(p0_limb_4_col6, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a0_limb_5_col114, b0_limb_5_col162),
                c0_limb_5_col210
            ),
            mul(p0_limb_5_col7, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_1_col254, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_1_col254,
                sub(mul(carry_1_col254, carry_1_col254), M31_1))
        );
    }

    // ===================== carry_2 constraint =====================
    // Process limb_6, limb_7, limb_8 of a0/b0/c0/p0, includes carry from carry_1
    {
        m31 s = add(
            carry_1_col254,
            sub(
                sub(
                    add(a0_limb_6_col115, b0_limb_6_col163),
                    c0_limb_6_col211
                ),
                mul(p0_limb_6_col8, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a0_limb_7_col116, b0_limb_7_col164),
                c0_limb_7_col212
            ),
            mul(p0_limb_7_col9, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a0_limb_8_col117, b0_limb_8_col165),
                c0_limb_8_col213
            ),
            mul(p0_limb_8_col10, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_2_col255, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_2_col255,
                sub(mul(carry_2_col255, carry_2_col255), M31_1))
        );
    }

    // ===================== carry_3 constraint (across word boundary: a0→a1) =====================
    // Process last 2 limbs of a0/b0/c0/p0 (limb_9, limb_10) and first limb of a1/b1/c1/p1 (limb_0)
    // Uses M31_128 scaling factor because crossing word boundary
    {
        m31 s = add(
            carry_2_col255,
            sub(
                sub(
                    add(a0_limb_9_col118, b0_limb_9_col166),
                    c0_limb_9_col214
                ),
                mul(p0_limb_9_col11, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a0_limb_10_col119, b0_limb_10_col167),
                c0_limb_10_col215
            ),
            mul(p0_limb_10_col12, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a1_limb_0_col121, b1_limb_0_col169),
                c1_limb_0_col217
            ),
            mul(p1_limb_0_col14, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_32768, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_3_col256, mul(s, M31_128))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_3_col256,
                sub(mul(carry_3_col256, carry_3_col256), M31_1))
        );
    }

    // ===================== carry_4-6 constraint (a1/b1/c1/p1 word internal) =====================
    // carry_4: process limb_1, limb_2, limb_3 of a1
    {
        m31 s = add(
            carry_3_col256,
            sub(
                sub(
                    add(a1_limb_1_col122, b1_limb_1_col170),
                    c1_limb_1_col218
                ),
                mul(p1_limb_1_col15, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a1_limb_2_col123, b1_limb_2_col171),
                c1_limb_2_col219
            ),
            mul(p1_limb_2_col16, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a1_limb_3_col124, b1_limb_3_col172),
                c1_limb_3_col220
            ),
            mul(p1_limb_3_col17, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_4_col257, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_4_col257,
                sub(mul(carry_4_col257, carry_4_col257), M31_1))
        );
    }

    // carry_5: process limb_4, limb_5, limb_6 of a1
    {
        m31 s = add(
            carry_4_col257,
            sub(
                sub(
                    add(a1_limb_4_col125, b1_limb_4_col173),
                    c1_limb_4_col221
                ),
                mul(p1_limb_4_col18, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a1_limb_5_col126, b1_limb_5_col174),
                c1_limb_5_col222
            ),
            mul(p1_limb_5_col19, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a1_limb_6_col127, b1_limb_6_col175),
                c1_limb_6_col223
            ),
            mul(p1_limb_6_col20, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_5_col258, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_5_col258,
                sub(mul(carry_5_col258, carry_5_col258), M31_1))
        );
    }

    // carry_6: process limb_7, limb_8, limb_9 of a1
    {
        m31 s = add(
            carry_5_col258,
            sub(
                sub(
                    add(a1_limb_7_col128, b1_limb_7_col176),
                    c1_limb_7_col224
                ),
                mul(p1_limb_7_col21, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a1_limb_8_col129, b1_limb_8_col177),
                c1_limb_8_col225
            ),
            mul(p1_limb_8_col22, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a1_limb_9_col130, b1_limb_9_col178),
                c1_limb_9_col226
            ),
            mul(p1_limb_9_col23, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_6_col259, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_6_col259,
                sub(mul(carry_6_col259, carry_6_col259), M31_1))
        );
    }

    // ===================== carry_7 constraint (across word boundary: a1→a2) =====================
    // Process last limb of a1/b1/c1/p1 (limb_10) and first 2 limbs of a2/b2/c2/p2 (limb_0, limb_1)
    // Uses M31_128 scaling factor and special M31_64 offset
    {
        m31 s = add(
            carry_6_col259,
            sub(
                sub(
                    add(a1_limb_10_col131, b1_limb_10_col179),
                    c1_limb_10_col227
                ),
                mul(p1_limb_10_col24, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a2_limb_0_col133, b2_limb_0_col181),
                c2_limb_0_col229
            ),
            mul(p2_limb_0_col26, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a2_limb_1_col134, b2_limb_1_col182),
                c2_limb_1_col230
            ),
            mul(p2_limb_1_col27, sub_p_bit_col252)
        );
        s = add(s, mul(M31_64, t1));
        s = add(s, mul(M31_32768, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_7_col260, mul(s, M31_128))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_7_col260,
                sub(mul(carry_7_col260, carry_7_col260), M31_1))
        );
    }

    // ===================== carry_8-9 constraint (a2/b2/c2/p2 word internal) =====================
    // carry_8: process limb_2, limb_3, limb_4 of a2
    {
        m31 s = add(
            carry_7_col260,
            sub(
                sub(
                    add(a2_limb_2_col135, b2_limb_2_col183),
                    c2_limb_2_col231
                ),
                mul(p2_limb_2_col28, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a2_limb_3_col136, b2_limb_3_col184),
                c2_limb_3_col232
            ),
            mul(p2_limb_3_col29, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a2_limb_4_col137, b2_limb_4_col185),
                c2_limb_4_col233
            ),
            mul(p2_limb_4_col30, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_8_col261, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_8_col261,
                sub(mul(carry_8_col261, carry_8_col261), M31_1))
        );
    }

    // carry_9: process limb_5, limb_6, limb_7 of a2
    {
        m31 s = add(
            carry_8_col261,
            sub(
                sub(
                    add(a2_limb_5_col138, b2_limb_5_col186),
                    c2_limb_5_col234
                ),
                mul(p2_limb_5_col31, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a2_limb_6_col139, b2_limb_6_col187),
                c2_limb_6_col235
            ),
            mul(p2_limb_6_col32, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a2_limb_7_col140, b2_limb_7_col188),
                c2_limb_7_col236
            ),
            mul(p2_limb_7_col33, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_9_col262, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_9_col262,
                sub(mul(carry_9_col262, carry_9_col262), M31_1))
        );
    }

    // ===================== carry_10 constraint (across word boundary: a2→a3) =====================
    // Process last 3 limbs of a2/b2/c2/p2 (limb_8, limb_9, limb_10)
    // Uses M31_128 scaling factor, preparing for final a3/b3/c3/p3 word
    {
        m31 s = add(
            carry_9_col262,
            sub(
                sub(
                    add(a2_limb_8_col141, b2_limb_8_col189),
                    c2_limb_8_col237
                ),
                mul(p2_limb_8_col34, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a2_limb_9_col142, b2_limb_9_col190),
                c2_limb_9_col238
            ),
            mul(p2_limb_9_col35, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a2_limb_10_col143, b2_limb_10_col191),
                c2_limb_10_col239
            ),
            mul(p2_limb_10_col36, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_10_col263, mul(s, M31_128))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_10_col263,
                sub(mul(carry_10_col263, carry_10_col263), M31_1))
        );
    }

    // ===================== carry_11-13 constraint (a3/b3/c3/p3 word internal) =====================
    // carry_11: process limb_0, limb_1, limb_2 of a3
    {
        m31 s = add(
            carry_10_col263,
            sub(
                sub(
                    add(a3_limb_0_col145, b3_limb_0_col193),
                    c3_limb_0_col241
                ),
                mul(p3_limb_0_col38, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a3_limb_1_col146, b3_limb_1_col194),
                c3_limb_1_col242
            ),
            mul(p3_limb_1_col39, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a3_limb_2_col147, b3_limb_2_col195),
                c3_limb_2_col243
            ),
            mul(p3_limb_2_col40, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_11_col264, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_11_col264,
                sub(mul(carry_11_col264, carry_11_col264), M31_1))
        );
    }

    // carry_12: process limb_3, limb_4, limb_5 of a3
    {
        m31 s = add(
            carry_11_col264,
            sub(
                sub(
                    add(a3_limb_3_col148, b3_limb_3_col196),
                    c3_limb_3_col244
                ),
                mul(p3_limb_3_col41, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a3_limb_4_col149, b3_limb_4_col197),
                c3_limb_4_col245
            ),
            mul(p3_limb_4_col42, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a3_limb_5_col150, b3_limb_5_col198),
                c3_limb_5_col246
            ),
            mul(p3_limb_5_col43, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_12_col265, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_12_col265,
                sub(mul(carry_12_col265, carry_12_col265), M31_1))
        );
    }

    // carry_13: process limb_6, limb_7, limb_8 of a3
    {
        m31 s = add(
            carry_12_col265,
            sub(
                sub(
                    add(a3_limb_6_col151, b3_limb_6_col199),
                    c3_limb_6_col247
                ),
                mul(p3_limb_6_col44, sub_p_bit_col252)
            )
        );
        m31 t1 = sub(
            sub(
                add(a3_limb_7_col152, b3_limb_7_col200),
                c3_limb_7_col248
            ),
            mul(p3_limb_7_col45, sub_p_bit_col252)
        );
        m31 t2 = sub(
            sub(
                add(a3_limb_8_col153, b3_limb_8_col201),
                c3_limb_8_col249
            ),
            mul(p3_limb_8_col46, sub_p_bit_col252)
        );
        s = add(s, mul(M31_512, t1));
        s = add(s, mul(M31_262144, t2));
        cuda_evaluator1.add_constraint(
            sub(carry_13_col266, mul(s, M31_16))
        );
        cuda_evaluator1.add_constraint(
            mul(carry_13_col266,
                sub(mul(carry_13_col266, carry_13_col266), M31_1))
        );
    }

    // ===================== final constraint: verify last carry must be 0 =====================
    // Process last 2 limbs of a3/b3/c3/p3 (limb_9, limb_10)
    // This constraint ensures the entire addition chain is correctly balanced, with no final carry/borrow
    // Formula: carry_13 + (a3[9] + b3[9] - c3[9] - p3[9]*sub_p_bit)
    //                 + 512 * (a3[10] + b3[10] - c3[10] - p3[10]*sub_p_bit) = 0
    {
        m31 term0 = add(
            carry_13_col266,
            sub(
                sub(
                    add(a3_limb_9_col154, b3_limb_9_col202),
                    c3_limb_9_col250
                ),
                mul(p3_limb_9_col47, sub_p_bit_col252)
            )
        );
        m31 term1 = sub(
            sub(
                add(a3_limb_10_col155, b3_limb_10_col203),
                c3_limb_10_col251
            ),
            mul(p3_limb_10_col48, sub_p_bit_col252)
        );
        cuda_evaluator1.add_constraint(
            add(term0, mul(M31_512, term1))
        );
    }

    // ===================== complete constraint evaluation =====================
    // Save constraint index for this row and accumulated result
    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// ============================================================================
// AddMod main function: orchestrates the entire evaluation process
// ============================================================================

extern "C"
void evaluate_add_mod_builtin(
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
    (void)number_of_columns;

    AddModBuiltin_Eval *add_mod_eval =
        (AddModBuiltin_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    // Clone preprocessed trace (trace0) and base trace (trace1) separately
    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    AddModBuiltin_Eval *device_add_mod_eval =
        cuda_malloc<AddModBuiltin_Eval>(1);
    cuda_mem_copy_host_to_device<AddModBuiltin_Eval>(
        add_mod_eval, device_add_mod_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_add_mod_builtin");

    int block_dim = eval_domain_size < ADD_MOD_BUILTIN_THREAD_COUNT_MAX
        ? eval_domain_size
        : ADD_MOD_BUILTIN_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_add_mod_builtin_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_add_mod_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_add_mod_builtin_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_add_mod_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (unsigned i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = logup_counts ? batching[logup_counts - 1] : 0;

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constraint_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    } else {
        generic_constraint_post_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constraint_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    }

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    generic_constraint_quotients_finalize_kernel<<<num_blocks, block_dim, 0, stream>>>(
        quotients_0,
        quotients_1,
        quotients_2,
        quotients_3,
        numerators,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size,
        should_accumulate
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_add_mod_builtin");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_add_mod_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
