#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_mul_mod_builtin.cuh"
#include "evaluate_range_check_12.cuh"
#include "evaluate_range_check_3_6_6_3.cuh"
#include "evaluate_common.cuh"
#include "mod_utils_common.cuh"
#include "mod_words_to_12_bit_array.cuh"
#include "double_karatsuba_n_8.cuh"

// NOTE:
// Current implementation for mul_mod_builtin CUDA evaluator skeleton:
// - Connects trace / interaction layout, consistent with CPU side
// - Reserves space for RangeCheck / ReadPositive / MemVerify child program includes
// - Does not duplicate all modular multiplication constraints from CPU side (DoubleKaratsuba + ModUtils)
//
// Future work can reference `mul_mod_builtin.rs` and `mod_utils.rs` to gradually port constraints to CUDA side.

#define MUL_MOD_BUILTIN_THREAD_COUNT_MAX 256

template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_mul_mod_builtin_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    MulModBuiltin_Eval *mul_mod_eval,
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

    // Read base trace columns (following CPU mul_mod_builtin::Eval::evaluate order)
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

    // ab_minus_c_div_p_limb column (252-283)
    m31 ab_minus_c_div_p_limb_0_col252 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_1_col253 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_2_col254 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_3_col255 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_4_col256 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_5_col257 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_6_col258 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_7_col259 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_8_col260 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_9_col261 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_10_col262 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_11_col263 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_12_col264 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_13_col265 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_14_col266 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_15_col267 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_16_col268 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_17_col269 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_18_col270 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_19_col271 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_20_col272 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_21_col273 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_22_col274 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_23_col275 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_24_col276 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_25_col277 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_26_col278 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_27_col279 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_28_col280 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_29_col281 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_30_col282 = cuda_evaluator1.next_trace_mask();
    m31 ab_minus_c_div_p_limb_31_col283 = cuda_evaluator1.next_trace_mask();

    // limb_b column for ModWordsTo12BitArray (284-363)
    m31 limb1b_0_col284 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_0_col285 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_0_col286 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_0_col287 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_0_col288 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_1_col289 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_1_col290 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_1_col291 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_1_col292 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_1_col293 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_0_col294 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_0_col295 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_0_col296 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_0_col297 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_0_col298 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_1_col299 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_1_col300 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_1_col301 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_1_col302 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_1_col303 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_0_col304 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_0_col305 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_0_col306 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_0_col307 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_0_col308 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_1_col309 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_1_col310 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_1_col311 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_1_col312 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_1_col313 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_0_col314 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_0_col315 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_0_col316 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_0_col317 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_0_col318 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_1_col319 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_1_col320 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_1_col321 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_1_col322 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_1_col323 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_0_col324 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_0_col325 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_0_col326 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_0_col327 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_0_col328 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_1_col329 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_1_col330 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_1_col331 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_1_col332 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_1_col333 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_0_col334 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_0_col335 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_0_col336 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_0_col337 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_0_col338 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_1_col339 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_1_col340 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_1_col341 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_1_col342 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_1_col343 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_0_col344 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_0_col345 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_0_col346 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_0_col347 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_0_col348 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_1_col349 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_1_col350 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_1_col351 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_1_col352 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_1_col353 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_0_col354 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_0_col355 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_0_col356 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_0_col357 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_0_col358 = cuda_evaluator1.next_trace_mask();
    m31 limb1b_1_col359 = cuda_evaluator1.next_trace_mask();
    m31 limb2b_1_col360 = cuda_evaluator1.next_trace_mask();
    m31 limb5b_1_col361 = cuda_evaluator1.next_trace_mask();
    m31 limb6b_1_col362 = cuda_evaluator1.next_trace_mask();
    m31 limb9b_1_col363 = cuda_evaluator1.next_trace_mask();

    // carry column (364-425)
    m31 carry_0_col364 = cuda_evaluator1.next_trace_mask();
    m31 carry_1_col365 = cuda_evaluator1.next_trace_mask();
    m31 carry_2_col366 = cuda_evaluator1.next_trace_mask();
    m31 carry_3_col367 = cuda_evaluator1.next_trace_mask();
    m31 carry_4_col368 = cuda_evaluator1.next_trace_mask();
    m31 carry_5_col369 = cuda_evaluator1.next_trace_mask();
    m31 carry_6_col370 = cuda_evaluator1.next_trace_mask();
    m31 carry_7_col371 = cuda_evaluator1.next_trace_mask();
    m31 carry_8_col372 = cuda_evaluator1.next_trace_mask();
    m31 carry_9_col373 = cuda_evaluator1.next_trace_mask();
    m31 carry_10_col374 = cuda_evaluator1.next_trace_mask();
    m31 carry_11_col375 = cuda_evaluator1.next_trace_mask();
    m31 carry_12_col376 = cuda_evaluator1.next_trace_mask();
    m31 carry_13_col377 = cuda_evaluator1.next_trace_mask();
    m31 carry_14_col378 = cuda_evaluator1.next_trace_mask();
    m31 carry_15_col379 = cuda_evaluator1.next_trace_mask();
    m31 carry_16_col380 = cuda_evaluator1.next_trace_mask();
    m31 carry_17_col381 = cuda_evaluator1.next_trace_mask();
    m31 carry_18_col382 = cuda_evaluator1.next_trace_mask();
    m31 carry_19_col383 = cuda_evaluator1.next_trace_mask();
    m31 carry_20_col384 = cuda_evaluator1.next_trace_mask();
    m31 carry_21_col385 = cuda_evaluator1.next_trace_mask();
    m31 carry_22_col386 = cuda_evaluator1.next_trace_mask();
    m31 carry_23_col387 = cuda_evaluator1.next_trace_mask();
    m31 carry_24_col388 = cuda_evaluator1.next_trace_mask();
    m31 carry_25_col389 = cuda_evaluator1.next_trace_mask();
    m31 carry_26_col390 = cuda_evaluator1.next_trace_mask();
    m31 carry_27_col391 = cuda_evaluator1.next_trace_mask();
    m31 carry_28_col392 = cuda_evaluator1.next_trace_mask();
    m31 carry_29_col393 = cuda_evaluator1.next_trace_mask();
    m31 carry_30_col394 = cuda_evaluator1.next_trace_mask();
    m31 carry_31_col395 = cuda_evaluator1.next_trace_mask();
    m31 carry_32_col396 = cuda_evaluator1.next_trace_mask();
    m31 carry_33_col397 = cuda_evaluator1.next_trace_mask();
    m31 carry_34_col398 = cuda_evaluator1.next_trace_mask();
    m31 carry_35_col399 = cuda_evaluator1.next_trace_mask();
    m31 carry_36_col400 = cuda_evaluator1.next_trace_mask();
    m31 carry_37_col401 = cuda_evaluator1.next_trace_mask();
    m31 carry_38_col402 = cuda_evaluator1.next_trace_mask();
    m31 carry_39_col403 = cuda_evaluator1.next_trace_mask();
    m31 carry_40_col404 = cuda_evaluator1.next_trace_mask();
    m31 carry_41_col405 = cuda_evaluator1.next_trace_mask();
    m31 carry_42_col406 = cuda_evaluator1.next_trace_mask();
    m31 carry_43_col407 = cuda_evaluator1.next_trace_mask();
    m31 carry_44_col408 = cuda_evaluator1.next_trace_mask();
    m31 carry_45_col409 = cuda_evaluator1.next_trace_mask();
    m31 carry_46_col410 = cuda_evaluator1.next_trace_mask();
    m31 carry_47_col411 = cuda_evaluator1.next_trace_mask();
    m31 carry_48_col412 = cuda_evaluator1.next_trace_mask();
    m31 carry_49_col413 = cuda_evaluator1.next_trace_mask();
    m31 carry_50_col414 = cuda_evaluator1.next_trace_mask();
    m31 carry_51_col415 = cuda_evaluator1.next_trace_mask();
    m31 carry_52_col416 = cuda_evaluator1.next_trace_mask();
    m31 carry_53_col417 = cuda_evaluator1.next_trace_mask();
    m31 carry_54_col418 = cuda_evaluator1.next_trace_mask();
    m31 carry_55_col419 = cuda_evaluator1.next_trace_mask();
    m31 carry_56_col420 = cuda_evaluator1.next_trace_mask();
    m31 carry_57_col421 = cuda_evaluator1.next_trace_mask();
    m31 carry_58_col422 = cuda_evaluator1.next_trace_mask();
    m31 carry_59_col423 = cuda_evaluator1.next_trace_mask();
    m31 carry_60_col424 = cuda_evaluator1.next_trace_mask();
    m31 carry_61_col425 = cuda_evaluator1.next_trace_mask();

    // Call CUDA version ModUtils::evaluate (same parameter order as add_mod_builtin).
    mod_utils_evaluate<EvaluatorT>(
        m31(mul_mod_eval->Claim.mul_mod_builtin_segment_start),
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
        mul_mod_eval->common_lookup_elements,
        &cuda_evaluator1
    );

    // RangeCheck_12 lookups for ab_minus_c_div_p_limb columns (32 lookups)
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_0_col252;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_1_col253;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_2_col254;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_3_col255;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_4_col256;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_5_col257;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_6_col258;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_7_col259;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_8_col260;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_9_col261;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_10_col262;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_11_col263;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_12_col264;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_13_col265;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_14_col266;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_15_col267;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_16_col268;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_17_col269;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_18_col270;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_19_col271;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_20_col272;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_21_col273;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_22_col274;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_23_col275;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_24_col276;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_25_col277;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_26_col278;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_27_col279;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_28_col280;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_29_col281;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_30_col282;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }
    {
        m31 values[1];
        values[0] = ab_minus_c_div_p_limb_31_col283;
        m31 rc12_vals[2] = {RANGE_CHECK_12_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc12_vals);
    }

    // ModWordsTo12BitArray call 1: p0+p1 -> p_12bit_0
    m31 p_12bit_0[16];
    mod_words_to_12_bit_array_evaluate<EvaluatorT>(
        p0_limb_0_col2, p0_limb_1_col3, p0_limb_2_col4, p0_limb_3_col5,
        p0_limb_4_col6, p0_limb_5_col7, p0_limb_6_col8, p0_limb_7_col9,
        p0_limb_8_col10, p0_limb_9_col11, p0_limb_10_col12,
        p1_limb_0_col14, p1_limb_1_col15, p1_limb_2_col16, p1_limb_3_col17,
        p1_limb_4_col18, p1_limb_5_col19, p1_limb_6_col20, p1_limb_7_col21,
        p1_limb_8_col22, p1_limb_9_col23, p1_limb_10_col24,
        limb1b_0_col284, limb2b_0_col285, limb5b_0_col286, limb6b_0_col287, limb9b_0_col288,
        limb1b_1_col289, limb2b_1_col290, limb5b_1_col291, limb6b_1_col292, limb9b_1_col293,
        mul_mod_eval->common_lookup_elements,
        p_12bit_0, &cuda_evaluator1
    );

    // ModWordsTo12BitArray call 2: p2+p3 -> p_12bit_1
    m31 p_12bit_1[16];
    mod_words_to_12_bit_array_evaluate<EvaluatorT>(
        p2_limb_0_col26, p2_limb_1_col27, p2_limb_2_col28, p2_limb_3_col29,
        p2_limb_4_col30, p2_limb_5_col31, p2_limb_6_col32, p2_limb_7_col33,
        p2_limb_8_col34, p2_limb_9_col35, p2_limb_10_col36,
        p3_limb_0_col38, p3_limb_1_col39, p3_limb_2_col40, p3_limb_3_col41,
        p3_limb_4_col42, p3_limb_5_col43, p3_limb_6_col44, p3_limb_7_col45,
        p3_limb_8_col46, p3_limb_9_col47, p3_limb_10_col48,
        limb1b_0_col294, limb2b_0_col295, limb5b_0_col296, limb6b_0_col297, limb9b_0_col298,
        limb1b_1_col299, limb2b_1_col300, limb5b_1_col301, limb6b_1_col302, limb9b_1_col303,
        mul_mod_eval->common_lookup_elements,
        p_12bit_1, &cuda_evaluator1
    );

    // ModWordsTo12BitArray call 3: a0+a1 -> a_12bit_0
    m31 a_12bit_0[16];
    mod_words_to_12_bit_array_evaluate<EvaluatorT>(
        a0_limb_0_col109, a0_limb_1_col110, a0_limb_2_col111, a0_limb_3_col112,
        a0_limb_4_col113, a0_limb_5_col114, a0_limb_6_col115, a0_limb_7_col116,
        a0_limb_8_col117, a0_limb_9_col118, a0_limb_10_col119,
        a1_limb_0_col121, a1_limb_1_col122, a1_limb_2_col123, a1_limb_3_col124,
        a1_limb_4_col125, a1_limb_5_col126, a1_limb_6_col127, a1_limb_7_col128,
        a1_limb_8_col129, a1_limb_9_col130, a1_limb_10_col131,
        limb1b_0_col304, limb2b_0_col305, limb5b_0_col306, limb6b_0_col307, limb9b_0_col308,
        limb1b_1_col309, limb2b_1_col310, limb5b_1_col311, limb6b_1_col312, limb9b_1_col313,
        mul_mod_eval->common_lookup_elements,
        a_12bit_0, &cuda_evaluator1
    );

    // ModWordsTo12BitArray call 4: a2+a3 -> a_12bit_1
    m31 a_12bit_1[16];
    mod_words_to_12_bit_array_evaluate<EvaluatorT>(
        a2_limb_0_col133, a2_limb_1_col134, a2_limb_2_col135, a2_limb_3_col136,
        a2_limb_4_col137, a2_limb_5_col138, a2_limb_6_col139, a2_limb_7_col140,
        a2_limb_8_col141, a2_limb_9_col142, a2_limb_10_col143,
        a3_limb_0_col145, a3_limb_1_col146, a3_limb_2_col147, a3_limb_3_col148,
        a3_limb_4_col149, a3_limb_5_col150, a3_limb_6_col151, a3_limb_7_col152,
        a3_limb_8_col153, a3_limb_9_col154, a3_limb_10_col155,
        limb1b_0_col314, limb2b_0_col315, limb5b_0_col316, limb6b_0_col317, limb9b_0_col318,
        limb1b_1_col319, limb2b_1_col320, limb5b_1_col321, limb6b_1_col322, limb9b_1_col323,
        mul_mod_eval->common_lookup_elements,
        a_12bit_1, &cuda_evaluator1
    );

    // ModWordsTo12BitArray call 5: b0+b1 -> b_12bit_0
    m31 b_12bit_0[16];
    mod_words_to_12_bit_array_evaluate<EvaluatorT>(
        b0_limb_0_col157, b0_limb_1_col158, b0_limb_2_col159, b0_limb_3_col160,
        b0_limb_4_col161, b0_limb_5_col162, b0_limb_6_col163, b0_limb_7_col164,
        b0_limb_8_col165, b0_limb_9_col166, b0_limb_10_col167,
        b1_limb_0_col169, b1_limb_1_col170, b1_limb_2_col171, b1_limb_3_col172,
        b1_limb_4_col173, b1_limb_5_col174, b1_limb_6_col175, b1_limb_7_col176,
        b1_limb_8_col177, b1_limb_9_col178, b1_limb_10_col179,
        limb1b_0_col324, limb2b_0_col325, limb5b_0_col326, limb6b_0_col327, limb9b_0_col328,
        limb1b_1_col329, limb2b_1_col330, limb5b_1_col331, limb6b_1_col332, limb9b_1_col333,
        mul_mod_eval->common_lookup_elements,
        b_12bit_0, &cuda_evaluator1
    );

    // ModWordsTo12BitArray call 6: b2+b3 -> b_12bit_1
    m31 b_12bit_1[16];
    mod_words_to_12_bit_array_evaluate<EvaluatorT>(
        b2_limb_0_col181, b2_limb_1_col182, b2_limb_2_col183, b2_limb_3_col184,
        b2_limb_4_col185, b2_limb_5_col186, b2_limb_6_col187, b2_limb_7_col188,
        b2_limb_8_col189, b2_limb_9_col190, b2_limb_10_col191,
        b3_limb_0_col193, b3_limb_1_col194, b3_limb_2_col195, b3_limb_3_col196,
        b3_limb_4_col197, b3_limb_5_col198, b3_limb_6_col199, b3_limb_7_col200,
        b3_limb_8_col201, b3_limb_9_col202, b3_limb_10_col203,
        limb1b_0_col334, limb2b_0_col335, limb5b_0_col336, limb6b_0_col337, limb9b_0_col338,
        limb1b_1_col339, limb2b_1_col340, limb5b_1_col341, limb6b_1_col342, limb9b_1_col343,
        mul_mod_eval->common_lookup_elements,
        b_12bit_1, &cuda_evaluator1
    );

    // ModWordsTo12BitArray call 7: c0+c1 -> c_12bit_0
    m31 c_12bit_0[16];
    mod_words_to_12_bit_array_evaluate<EvaluatorT>(
        c0_limb_0_col205, c0_limb_1_col206, c0_limb_2_col207, c0_limb_3_col208,
        c0_limb_4_col209, c0_limb_5_col210, c0_limb_6_col211, c0_limb_7_col212,
        c0_limb_8_col213, c0_limb_9_col214, c0_limb_10_col215,
        c1_limb_0_col217, c1_limb_1_col218, c1_limb_2_col219, c1_limb_3_col220,
        c1_limb_4_col221, c1_limb_5_col222, c1_limb_6_col223, c1_limb_7_col224,
        c1_limb_8_col225, c1_limb_9_col226, c1_limb_10_col227,
        limb1b_0_col344, limb2b_0_col345, limb5b_0_col346, limb6b_0_col347, limb9b_0_col348,
        limb1b_1_col349, limb2b_1_col350, limb5b_1_col351, limb6b_1_col352, limb9b_1_col353,
        mul_mod_eval->common_lookup_elements,
        c_12bit_0, &cuda_evaluator1
    );

    // ModWordsTo12BitArray call 8: c2+c3 -> c_12bit_1
    m31 c_12bit_1[16];
    mod_words_to_12_bit_array_evaluate<EvaluatorT>(
        c2_limb_0_col229, c2_limb_1_col230, c2_limb_2_col231, c2_limb_3_col232,
        c2_limb_4_col233, c2_limb_5_col234, c2_limb_6_col235, c2_limb_7_col236,
        c2_limb_8_col237, c2_limb_9_col238, c2_limb_10_col239,
        c3_limb_0_col241, c3_limb_1_col242, c3_limb_2_col243, c3_limb_3_col244,
        c3_limb_4_col245, c3_limb_5_col246, c3_limb_6_col247, c3_limb_7_col248,
        c3_limb_8_col249, c3_limb_9_col250, c3_limb_10_col251,
        limb1b_0_col354, limb2b_0_col355, limb5b_0_col356, limb6b_0_col357, limb9b_0_col358,
        limb1b_1_col359, limb2b_1_col360, limb5b_1_col361, limb6b_1_col362, limb9b_1_col363,
        mul_mod_eval->common_lookup_elements,
        c_12bit_1, &cuda_evaluator1
    );

    // DoubleKaratsubaN8 call 1: a * b -> ab_product
    m31 ab_input[64];
    for (int i = 0; i < 16; i++) {
        ab_input[i] = a_12bit_0[i];
        ab_input[16 + i] = a_12bit_1[i];
        ab_input[32 + i] = b_12bit_0[i];
        ab_input[48 + i] = b_12bit_1[i];
    }
    m31 ab_product[63];
    double_karatsuba_n_8_evaluate(ab_input, ab_product);

    // DoubleKaratsubaN8 call 2: q * p -> qp_product
    m31 qp_input[64];
    qp_input[0] = ab_minus_c_div_p_limb_0_col252;
    qp_input[1] = ab_minus_c_div_p_limb_1_col253;
    qp_input[2] = ab_minus_c_div_p_limb_2_col254;
    qp_input[3] = ab_minus_c_div_p_limb_3_col255;
    qp_input[4] = ab_minus_c_div_p_limb_4_col256;
    qp_input[5] = ab_minus_c_div_p_limb_5_col257;
    qp_input[6] = ab_minus_c_div_p_limb_6_col258;
    qp_input[7] = ab_minus_c_div_p_limb_7_col259;
    qp_input[8] = ab_minus_c_div_p_limb_8_col260;
    qp_input[9] = ab_minus_c_div_p_limb_9_col261;
    qp_input[10] = ab_minus_c_div_p_limb_10_col262;
    qp_input[11] = ab_minus_c_div_p_limb_11_col263;
    qp_input[12] = ab_minus_c_div_p_limb_12_col264;
    qp_input[13] = ab_minus_c_div_p_limb_13_col265;
    qp_input[14] = ab_minus_c_div_p_limb_14_col266;
    qp_input[15] = ab_minus_c_div_p_limb_15_col267;
    qp_input[16] = ab_minus_c_div_p_limb_16_col268;
    qp_input[17] = ab_minus_c_div_p_limb_17_col269;
    qp_input[18] = ab_minus_c_div_p_limb_18_col270;
    qp_input[19] = ab_minus_c_div_p_limb_19_col271;
    qp_input[20] = ab_minus_c_div_p_limb_20_col272;
    qp_input[21] = ab_minus_c_div_p_limb_21_col273;
    qp_input[22] = ab_minus_c_div_p_limb_22_col274;
    qp_input[23] = ab_minus_c_div_p_limb_23_col275;
    qp_input[24] = ab_minus_c_div_p_limb_24_col276;
    qp_input[25] = ab_minus_c_div_p_limb_25_col277;
    qp_input[26] = ab_minus_c_div_p_limb_26_col278;
    qp_input[27] = ab_minus_c_div_p_limb_27_col279;
    qp_input[28] = ab_minus_c_div_p_limb_28_col280;
    qp_input[29] = ab_minus_c_div_p_limb_29_col281;
    qp_input[30] = ab_minus_c_div_p_limb_30_col282;
    qp_input[31] = ab_minus_c_div_p_limb_31_col283;
    for (int i = 0; i < 16; i++) {
        qp_input[32 + i] = p_12bit_0[i];
        qp_input[48 + i] = p_12bit_1[i];
    }
    m31 qp_product[63];
    double_karatsuba_n_8_evaluate(qp_input, qp_product);

    // c_12bit combination
    m31 c_12bit[32];
    for (int i = 0; i < 16; i++) {
        c_12bit[i] = c_12bit_0[i];
        c_12bit[16 + i] = c_12bit_1[i];
    }

    // carry array
    m31 carry[62];
    carry[0] = carry_0_col364;
    carry[1] = carry_1_col365;
    carry[2] = carry_2_col366;
    carry[3] = carry_3_col367;
    carry[4] = carry_4_col368;
    carry[5] = carry_5_col369;
    carry[6] = carry_6_col370;
    carry[7] = carry_7_col371;
    carry[8] = carry_8_col372;
    carry[9] = carry_9_col373;
    carry[10] = carry_10_col374;
    carry[11] = carry_11_col375;
    carry[12] = carry_12_col376;
    carry[13] = carry_13_col377;
    carry[14] = carry_14_col378;
    carry[15] = carry_15_col379;
    carry[16] = carry_16_col380;
    carry[17] = carry_17_col381;
    carry[18] = carry_18_col382;
    carry[19] = carry_19_col383;
    carry[20] = carry_20_col384;
    carry[21] = carry_21_col385;
    carry[22] = carry_22_col386;
    carry[23] = carry_23_col387;
    carry[24] = carry_24_col388;
    carry[25] = carry_25_col389;
    carry[26] = carry_26_col390;
    carry[27] = carry_27_col391;
    carry[28] = carry_28_col392;
    carry[29] = carry_29_col393;
    carry[30] = carry_30_col394;
    carry[31] = carry_31_col395;
    carry[32] = carry_32_col396;
    carry[33] = carry_33_col397;
    carry[34] = carry_34_col398;
    carry[35] = carry_35_col399;
    carry[36] = carry_36_col400;
    carry[37] = carry_37_col401;
    carry[38] = carry_38_col402;
    carry[39] = carry_39_col403;
    carry[40] = carry_40_col404;
    carry[41] = carry_41_col405;
    carry[42] = carry_42_col406;
    carry[43] = carry_43_col407;
    carry[44] = carry_44_col408;
    carry[45] = carry_45_col409;
    carry[46] = carry_46_col410;
    carry[47] = carry_47_col411;
    carry[48] = carry_48_col412;
    carry[49] = carry_49_col413;
    carry[50] = carry_50_col414;
    carry[51] = carry_51_col415;
    carry[52] = carry_52_col416;
    carry[53] = carry_53_col417;
    carry[54] = carry_54_col418;
    carry[55] = carry_55_col419;
    carry[56] = carry_56_col420;
    carry[57] = carry_57_col421;
    carry[58] = carry_58_col422;
    carry[59] = carry_59_col423;
    carry[60] = carry_60_col424;
    carry[61] = carry_61_col425;

    // carry constraint and RangeCheck_18 lookups
    // Formula: carry[i] = (prev_carry - c_limb + (ab_product[i] - qp_product[i])) * 524288
    m31 M31_524288 = m31(524288);
    m31 M31_131072 = m31(131072);

    // carry_0 constraint (prev_carry = 0)
    {
        m31 diff = add(sub(m31(0), c_12bit[0]), sub(ab_product[0], qp_product[0]));
        cuda_evaluator1.add_constraint(sub(carry[0], mul(diff, M31_524288)));
        m31 values[1] = {add(carry[0], M31_131072)};
        m31 rc18_vals[2] = {RANGE_CHECK_18_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc18_vals);
    }

    // carry_1 to carry_31 constraint (uses c_12bit[1-31])
    for (int i = 1; i < 32; i++) {
        m31 diff = add(sub(carry[i-1], c_12bit[i]), sub(ab_product[i], qp_product[i]));
        cuda_evaluator1.add_constraint(sub(carry[i], mul(diff, M31_524288)));
        m31 values[1] = {add(carry[i], M31_131072)};
        m31 rc18_vals[2] = {RANGE_CHECK_18_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc18_vals);
    }

    // carry_32 to carry_61 constraint (no c_12bit entry, because c only has 32 limbs)
    for (int i = 32; i < 62; i++) {
        m31 diff = add(carry[i-1], sub(ab_product[i], qp_product[i]));
        cuda_evaluator1.add_constraint(sub(carry[i], mul(diff, M31_524288)));
        m31 values[1] = {add(carry[i], M31_131072)};
        m31 rc18_vals[2] = {RANGE_CHECK_18_RELATION_ID, values[0]};
        cuda_evaluator1.add_to_relation<2>(mul_mod_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc18_vals);
    }

    // final limb constraint: (ab_product[62] + carry[61]) - qp_product[62] = 0
    cuda_evaluator1.add_constraint(sub(add(ab_product[62], carry[61]), qp_product[62]));

    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

extern "C"
void evaluate_mul_mod_builtin(
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

    MulModBuiltin_Eval *mul_mod_eval =
        (MulModBuiltin_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    // Clone trace0 and trace1 separately
    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    MulModBuiltin_Eval *device_mul_mod_eval =
        cuda_malloc<MulModBuiltin_Eval>(1);
    cuda_mem_copy_host_to_device<MulModBuiltin_Eval>(
        mul_mod_eval, device_mul_mod_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_mul_mod_builtin");

    int block_dim = eval_domain_size < MUL_MOD_BUILTIN_THREAD_COUNT_MAX
        ? eval_domain_size
        : MUL_MOD_BUILTIN_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_mul_mod_builtin_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_mul_mod_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_mul_mod_builtin_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_mul_mod_eval,
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
    global_timer.end("evaluate_mul_mod_builtin");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_mul_mod_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
