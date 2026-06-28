#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_generic_opcode.cuh"
#include "evaluate_common.cuh"

#define GENERIC_OPCODE_THREAD_COUNT_MAX 256

// Auto-generated from Rust source: generic_opcode.rs + all subroutines
// N_TRACE_COLUMNS = 243, constraints = 164, relations = 67

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_generic_opcode_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    GenericOpcode_Eval *generic_opcode_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    EvaluatorT cuda_evaluator(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        {{0,0},{0,0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // ===== Read 243 trace columns =====
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 offset0_col3 = cuda_evaluator.next_trace_mask();
    m31 offset1_col4 = cuda_evaluator.next_trace_mask();
    m31 offset2_col5 = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp_col6 = cuda_evaluator.next_trace_mask();
    m31 op0_base_fp_col7 = cuda_evaluator.next_trace_mask();
    m31 op1_imm_col8 = cuda_evaluator.next_trace_mask();
    m31 op1_base_fp_col9 = cuda_evaluator.next_trace_mask();
    m31 op1_base_ap_col10 = cuda_evaluator.next_trace_mask();
    m31 res_add_col11 = cuda_evaluator.next_trace_mask();
    m31 res_mul_col12 = cuda_evaluator.next_trace_mask();
    m31 pc_update_jump_col13 = cuda_evaluator.next_trace_mask();
    m31 pc_update_jump_rel_col14 = cuda_evaluator.next_trace_mask();
    m31 pc_update_jnz_col15 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_col16 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col17 = cuda_evaluator.next_trace_mask();
    m31 opcode_call_col18 = cuda_evaluator.next_trace_mask();
    m31 opcode_ret_col19 = cuda_evaluator.next_trace_mask();
    m31 opcode_assert_eq_col20 = cuda_evaluator.next_trace_mask();
    m31 dst_src_col21 = cuda_evaluator.next_trace_mask();
    m31 dst_id_col22 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_0_col23 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_1_col24 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_2_col25 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_3_col26 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_4_col27 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_5_col28 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_6_col29 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_7_col30 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_8_col31 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_9_col32 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_10_col33 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_11_col34 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_12_col35 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_13_col36 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_14_col37 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_15_col38 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_16_col39 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_17_col40 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_18_col41 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_19_col42 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_20_col43 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_21_col44 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_22_col45 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_23_col46 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_24_col47 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_25_col48 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_26_col49 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_27_col50 = cuda_evaluator.next_trace_mask();
    m31 op0_src_col51 = cuda_evaluator.next_trace_mask();
    m31 op0_id_col52 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_0_col53 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_1_col54 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_2_col55 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_3_col56 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_4_col57 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_5_col58 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_6_col59 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_7_col60 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_8_col61 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_9_col62 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_10_col63 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_11_col64 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_12_col65 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_13_col66 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_14_col67 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_15_col68 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_16_col69 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_17_col70 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_18_col71 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_19_col72 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_20_col73 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_21_col74 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_22_col75 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_23_col76 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_24_col77 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_25_col78 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_26_col79 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_27_col80 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col81 = cuda_evaluator.next_trace_mask();
    m31 op1_src_col82 = cuda_evaluator.next_trace_mask();
    m31 op1_id_col83 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_0_col84 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_1_col85 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_2_col86 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_3_col87 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_4_col88 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_5_col89 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_6_col90 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_7_col91 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_8_col92 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_9_col93 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_10_col94 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_11_col95 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_12_col96 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_13_col97 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_14_col98 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_15_col99 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_16_col100 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_17_col101 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_18_col102 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_19_col103 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_20_col104 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_21_col105 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_22_col106 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_23_col107 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_24_col108 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_25_col109 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_26_col110 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_27_col111 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_0_col112 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_1_col113 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_2_col114 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_3_col115 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_4_col116 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_5_col117 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_6_col118 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_7_col119 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_8_col120 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_9_col121 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_10_col122 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_11_col123 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_12_col124 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_13_col125 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_14_col126 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_15_col127 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_16_col128 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_17_col129 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_18_col130 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_19_col131 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_20_col132 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_21_col133 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_22_col134 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_23_col135 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_24_col136 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_25_col137 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_26_col138 = cuda_evaluator.next_trace_mask();
    m31 add_res_limb_27_col139 = cuda_evaluator.next_trace_mask();
    m31 sub_p_bit_col140 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_0_col141 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_1_col142 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_2_col143 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_3_col144 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_4_col145 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_5_col146 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_6_col147 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_7_col148 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_8_col149 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_9_col150 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_10_col151 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_11_col152 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_12_col153 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_13_col154 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_14_col155 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_15_col156 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_16_col157 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_17_col158 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_18_col159 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_19_col160 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_20_col161 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_21_col162 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_22_col163 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_23_col164 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_24_col165 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_25_col166 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_26_col167 = cuda_evaluator.next_trace_mask();
    m31 mul_res_limb_27_col168 = cuda_evaluator.next_trace_mask();
    m31 k_col169 = cuda_evaluator.next_trace_mask();
    m31 carry_0_col170 = cuda_evaluator.next_trace_mask();
    m31 carry_1_col171 = cuda_evaluator.next_trace_mask();
    m31 carry_2_col172 = cuda_evaluator.next_trace_mask();
    m31 carry_3_col173 = cuda_evaluator.next_trace_mask();
    m31 carry_4_col174 = cuda_evaluator.next_trace_mask();
    m31 carry_5_col175 = cuda_evaluator.next_trace_mask();
    m31 carry_6_col176 = cuda_evaluator.next_trace_mask();
    m31 carry_7_col177 = cuda_evaluator.next_trace_mask();
    m31 carry_8_col178 = cuda_evaluator.next_trace_mask();
    m31 carry_9_col179 = cuda_evaluator.next_trace_mask();
    m31 carry_10_col180 = cuda_evaluator.next_trace_mask();
    m31 carry_11_col181 = cuda_evaluator.next_trace_mask();
    m31 carry_12_col182 = cuda_evaluator.next_trace_mask();
    m31 carry_13_col183 = cuda_evaluator.next_trace_mask();
    m31 carry_14_col184 = cuda_evaluator.next_trace_mask();
    m31 carry_15_col185 = cuda_evaluator.next_trace_mask();
    m31 carry_16_col186 = cuda_evaluator.next_trace_mask();
    m31 carry_17_col187 = cuda_evaluator.next_trace_mask();
    m31 carry_18_col188 = cuda_evaluator.next_trace_mask();
    m31 carry_19_col189 = cuda_evaluator.next_trace_mask();
    m31 carry_20_col190 = cuda_evaluator.next_trace_mask();
    m31 carry_21_col191 = cuda_evaluator.next_trace_mask();
    m31 carry_22_col192 = cuda_evaluator.next_trace_mask();
    m31 carry_23_col193 = cuda_evaluator.next_trace_mask();
    m31 carry_24_col194 = cuda_evaluator.next_trace_mask();
    m31 carry_25_col195 = cuda_evaluator.next_trace_mask();
    m31 carry_26_col196 = cuda_evaluator.next_trace_mask();
    m31 res_limb_0_col197 = cuda_evaluator.next_trace_mask();
    m31 res_limb_1_col198 = cuda_evaluator.next_trace_mask();
    m31 res_limb_2_col199 = cuda_evaluator.next_trace_mask();
    m31 res_limb_3_col200 = cuda_evaluator.next_trace_mask();
    m31 res_limb_4_col201 = cuda_evaluator.next_trace_mask();
    m31 res_limb_5_col202 = cuda_evaluator.next_trace_mask();
    m31 res_limb_6_col203 = cuda_evaluator.next_trace_mask();
    m31 res_limb_7_col204 = cuda_evaluator.next_trace_mask();
    m31 res_limb_8_col205 = cuda_evaluator.next_trace_mask();
    m31 res_limb_9_col206 = cuda_evaluator.next_trace_mask();
    m31 res_limb_10_col207 = cuda_evaluator.next_trace_mask();
    m31 res_limb_11_col208 = cuda_evaluator.next_trace_mask();
    m31 res_limb_12_col209 = cuda_evaluator.next_trace_mask();
    m31 res_limb_13_col210 = cuda_evaluator.next_trace_mask();
    m31 res_limb_14_col211 = cuda_evaluator.next_trace_mask();
    m31 res_limb_15_col212 = cuda_evaluator.next_trace_mask();
    m31 res_limb_16_col213 = cuda_evaluator.next_trace_mask();
    m31 res_limb_17_col214 = cuda_evaluator.next_trace_mask();
    m31 res_limb_18_col215 = cuda_evaluator.next_trace_mask();
    m31 res_limb_19_col216 = cuda_evaluator.next_trace_mask();
    m31 res_limb_20_col217 = cuda_evaluator.next_trace_mask();
    m31 res_limb_21_col218 = cuda_evaluator.next_trace_mask();
    m31 res_limb_22_col219 = cuda_evaluator.next_trace_mask();
    m31 res_limb_23_col220 = cuda_evaluator.next_trace_mask();
    m31 res_limb_24_col221 = cuda_evaluator.next_trace_mask();
    m31 res_limb_25_col222 = cuda_evaluator.next_trace_mask();
    m31 res_limb_26_col223 = cuda_evaluator.next_trace_mask();
    m31 res_limb_27_col224 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col225 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col226 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col227 = cuda_evaluator.next_trace_mask();
    m31 msb_col228 = cuda_evaluator.next_trace_mask();
    m31 mid_limbs_set_col229 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col230 = cuda_evaluator.next_trace_mask();
    m31 dst_sum_squares_inv_col231 = cuda_evaluator.next_trace_mask();
    m31 dst_sum_inv_col232 = cuda_evaluator.next_trace_mask();
    m31 op1_as_rel_imm_cond_col233 = cuda_evaluator.next_trace_mask();
    m31 msb_col234 = cuda_evaluator.next_trace_mask();
    m31 mid_limbs_set_col235 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col236 = cuda_evaluator.next_trace_mask();
    m31 next_pc_jnz_col237 = cuda_evaluator.next_trace_mask();
    m31 next_pc_col238 = cuda_evaluator.next_trace_mask();
    m31 next_ap_col239 = cuda_evaluator.next_trace_mask();
    m31 range_check_29_bot11bits_col240 = cuda_evaluator.next_trace_mask();
    m31 next_fp_col241 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // ===== DecodeGenericInstruction =====
    // 5 intermediates computed from flags
    m31 op1_base_op0 = cuda_evaluator.add_intermediate(sub(sub(sub(m31(1), op1_imm_col8), op1_base_fp_col9), op1_base_ap_col10));
    m31 res_op1 = cuda_evaluator.add_intermediate(sub(sub(sub(m31(1), res_add_col11), res_mul_col12), pc_update_jnz_col15));
    m31 pc_update_regular = cuda_evaluator.add_intermediate(sub(sub(sub(m31(1), pc_update_jump_col13), pc_update_jump_rel_col14), pc_update_jnz_col15));
    m31 ap_update_regular = cuda_evaluator.add_intermediate(sub(sub(sub(m31(1), ap_update_add_col16), ap_update_add_1_col17), opcode_call_col18));
    m31 fp_update_regular = cuda_evaluator.add_intermediate(sub(sub(m31(1), opcode_call_col18), opcode_ret_col19));

    // DecodeInstructionDf7a6: 15 flag constraints (15)
    cuda_evaluator.add_constraint(mul(dst_base_fp_col6, sub(dst_base_fp_col6, m31(1))));
    cuda_evaluator.add_constraint(mul(op0_base_fp_col7, sub(op0_base_fp_col7, m31(1))));
    cuda_evaluator.add_constraint(mul(op1_imm_col8, sub(op1_imm_col8, m31(1))));
    cuda_evaluator.add_constraint(mul(op1_base_fp_col9, sub(op1_base_fp_col9, m31(1))));
    cuda_evaluator.add_constraint(mul(op1_base_ap_col10, sub(op1_base_ap_col10, m31(1))));
    cuda_evaluator.add_constraint(mul(res_add_col11, sub(res_add_col11, m31(1))));
    cuda_evaluator.add_constraint(mul(res_mul_col12, sub(res_mul_col12, m31(1))));
    cuda_evaluator.add_constraint(mul(pc_update_jump_col13, sub(pc_update_jump_col13, m31(1))));
    cuda_evaluator.add_constraint(mul(pc_update_jump_rel_col14, sub(pc_update_jump_rel_col14, m31(1))));
    cuda_evaluator.add_constraint(mul(pc_update_jnz_col15, sub(pc_update_jnz_col15, m31(1))));
    cuda_evaluator.add_constraint(mul(ap_update_add_col16, sub(ap_update_add_col16, m31(1))));
    cuda_evaluator.add_constraint(mul(ap_update_add_1_col17, sub(ap_update_add_1_col17, m31(1))));
    cuda_evaluator.add_constraint(mul(opcode_call_col18, sub(opcode_call_col18, m31(1))));
    cuda_evaluator.add_constraint(mul(opcode_ret_col19, sub(opcode_ret_col19, m31(1))));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(opcode_assert_eq_col20, m31(1))));

    // DecodeGenericInstruction: 5 additional constraints (op1_src, res_logic, pc_update, ap_update, opcode)
    cuda_evaluator.add_constraint(mul(op1_base_op0, sub(m31(1), op1_base_op0)));
    cuda_evaluator.add_constraint(mul(res_op1, sub(m31(1), res_op1)));
    cuda_evaluator.add_constraint(mul(pc_update_regular, sub(m31(1), pc_update_regular)));
    cuda_evaluator.add_constraint(mul(ap_update_regular, sub(m31(1), ap_update_regular)));
    cuda_evaluator.add_constraint(mul(fp_update_regular, sub(m31(1), fp_update_regular)));

    // verify_instruction relation: packed flags split into 2 values (1 relation)
    { m31 v[7] = { VERIFY_INSTRUCTION_RELATION_ID, input_pc_col0, offset0_col3, offset1_col4, offset2_col5, add(add(add(add(add(mul(dst_base_fp_col6, m31(8)), mul(op0_base_fp_col7, m31(16))), mul(op1_imm_col8, m31(32))), mul(op1_base_fp_col9, m31(64))), mul(op1_base_ap_col10, m31(128))), mul(res_add_col11, m31(256))), add(add(add(add(add(add(add(add(mul(res_mul_col12, m31(1)), mul(pc_update_jump_col13, m31(2))), mul(pc_update_jump_rel_col14, m31(4))), mul(pc_update_jnz_col15, m31(8))), mul(ap_update_add_col16, m31(16))), mul(ap_update_add_1_col17, m31(32))), mul(opcode_call_col18, m31(64))), mul(opcode_ret_col19, m31(128))), mul(opcode_assert_eq_col20, m31(256))) }; cuda_evaluator.add_to_relation<7>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }

    m31 instruction_size = add(m31(1), op1_imm_col8);
    m31 offset0 = sub(offset0_col3, m31(32768));
    m31 offset1 = sub(offset1_col4, m31(32768));
    m31 offset2 = sub(offset2_col5, m31(32768));

    // ===== EvalOperands =====

    // dst_src constraint (1)
    cuda_evaluator.add_constraint(sub(dst_src_col21, add(mul(dst_base_fp_col6, input_fp_col2), mul(sub(m31(1), dst_base_fp_col6), input_ap_col1))));

    // ReadPositiveNumBits252 for dst (2 relations)
    { m31 v[3] = { MEMORY_ADDRESS_TO_ID_RELATION_ID, add(dst_src_col21, offset0), dst_id_col22 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[30] = { MEMORY_ID_TO_BIG_RELATION_ID, dst_id_col22, dst_limb_0_col23, dst_limb_1_col24, dst_limb_2_col25, dst_limb_3_col26, dst_limb_4_col27, dst_limb_5_col28, dst_limb_6_col29, dst_limb_7_col30, dst_limb_8_col31, dst_limb_9_col32, dst_limb_10_col33, dst_limb_11_col34, dst_limb_12_col35, dst_limb_13_col36, dst_limb_14_col37, dst_limb_15_col38, dst_limb_16_col39, dst_limb_17_col40, dst_limb_18_col41, dst_limb_19_col42, dst_limb_20_col43, dst_limb_21_col44, dst_limb_22_col45, dst_limb_23_col46, dst_limb_24_col47, dst_limb_25_col48, dst_limb_26_col49, dst_limb_27_col50 }; cuda_evaluator.add_to_relation<30>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }

    // op0_src constraint (1)
    cuda_evaluator.add_constraint(sub(op0_src_col51, add(mul(op0_base_fp_col7, input_fp_col2), mul(sub(m31(1), op0_base_fp_col7), input_ap_col1))));

    // ReadPositiveNumBits252 for op0 (2 relations)
    { m31 v[3] = { MEMORY_ADDRESS_TO_ID_RELATION_ID, add(op0_src_col51, offset1), op0_id_col52 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[30] = { MEMORY_ID_TO_BIG_RELATION_ID, op0_id_col52, op0_limb_0_col53, op0_limb_1_col54, op0_limb_2_col55, op0_limb_3_col56, op0_limb_4_col57, op0_limb_5_col58, op0_limb_6_col59, op0_limb_7_col60, op0_limb_8_col61, op0_limb_9_col62, op0_limb_10_col63, op0_limb_11_col64, op0_limb_12_col65, op0_limb_13_col66, op0_limb_14_col67, op0_limb_15_col68, op0_limb_16_col69, op0_limb_17_col70, op0_limb_18_col71, op0_limb_19_col72, op0_limb_20_col73, op0_limb_21_col74, op0_limb_22_col75, op0_limb_23_col76, op0_limb_24_col77, op0_limb_25_col78, op0_limb_26_col79, op0_limb_27_col80 }; cuda_evaluator.add_to_relation<30>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }

    // CondFelt252AsAddr for op0 (cond=op1_base_op0) (3 constraints)
    cuda_evaluator.add_constraint(mul(op1_base_op0, add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(op0_limb_4_col57, op0_limb_5_col58), op0_limb_6_col59), op0_limb_7_col60), op0_limb_8_col61), op0_limb_9_col62), op0_limb_10_col63), op0_limb_11_col64), op0_limb_12_col65), op0_limb_13_col66), op0_limb_14_col67), op0_limb_15_col68), op0_limb_16_col69), op0_limb_17_col70), op0_limb_18_col71), op0_limb_19_col72), op0_limb_20_col73), op0_limb_21_col74), op0_limb_22_col75), op0_limb_23_col76), op0_limb_24_col77), op0_limb_25_col78), op0_limb_26_col79), op0_limb_27_col80)));
    cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col81, sub(partial_limb_msb_col81, m31(1))), op1_base_op0));
    m31 crc2_op0_remainder = cuda_evaluator.add_intermediate(sub(op0_limb_3_col56, mul(partial_limb_msb_col81, m31(2))));
    cuda_evaluator.add_constraint(mul(mul(crc2_op0_remainder, sub(crc2_op0_remainder, m31(1))), op1_base_op0));
    m31 op0_as_addr = add(add(add(op0_limb_0_col53, mul(op0_limb_1_col54, m31(512))), mul(op0_limb_2_col55, m31(262144))), mul(op0_limb_3_col56, m31(134217728)));

    // op1_src constraint (1)
    cuda_evaluator.add_constraint(sub(op1_src_col82, add(add(add(mul(op1_base_fp_col9, input_fp_col2), mul(op1_base_ap_col10, input_ap_col1)), mul(op1_imm_col8, input_pc_col0)), mul(op1_base_op0, op0_as_addr))));

    // ReadPositiveNumBits252 for op1 (2 relations)
    { m31 v[3] = { MEMORY_ADDRESS_TO_ID_RELATION_ID, add(op1_src_col82, offset2), op1_id_col83 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[30] = { MEMORY_ID_TO_BIG_RELATION_ID, op1_id_col83, op1_limb_0_col84, op1_limb_1_col85, op1_limb_2_col86, op1_limb_3_col87, op1_limb_4_col88, op1_limb_5_col89, op1_limb_6_col90, op1_limb_7_col91, op1_limb_8_col92, op1_limb_9_col93, op1_limb_10_col94, op1_limb_11_col95, op1_limb_12_col96, op1_limb_13_col97, op1_limb_14_col98, op1_limb_15_col99, op1_limb_16_col100, op1_limb_17_col101, op1_limb_18_col102, op1_limb_19_col103, op1_limb_20_col104, op1_limb_21_col105, op1_limb_22_col106, op1_limb_23_col107, op1_limb_24_col108, op1_limb_25_col109, op1_limb_26_col110, op1_limb_27_col111 }; cuda_evaluator.add_to_relation<30>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }

    // ===== Add252: RangeCheckMemValueN28 for add_res (14 relations) =====
    { m31 v[3] = { RANGE_CHECK_9_9_RELATION_ID, add_res_limb_0_col112, add_res_limb_1_col113 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_B_RELATION_ID, add_res_limb_2_col114, add_res_limb_3_col115 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_C_RELATION_ID, add_res_limb_4_col116, add_res_limb_5_col117 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_D_RELATION_ID, add_res_limb_6_col118, add_res_limb_7_col119 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_E_RELATION_ID, add_res_limb_8_col120, add_res_limb_9_col121 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_F_RELATION_ID, add_res_limb_10_col122, add_res_limb_11_col123 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_G_RELATION_ID, add_res_limb_12_col124, add_res_limb_13_col125 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_H_RELATION_ID, add_res_limb_14_col126, add_res_limb_15_col127 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_RELATION_ID, add_res_limb_16_col128, add_res_limb_17_col129 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_B_RELATION_ID, add_res_limb_18_col130, add_res_limb_19_col131 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_C_RELATION_ID, add_res_limb_20_col132, add_res_limb_21_col133 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_D_RELATION_ID, add_res_limb_22_col134, add_res_limb_23_col135 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_E_RELATION_ID, add_res_limb_24_col136, add_res_limb_25_col137 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_F_RELATION_ID, add_res_limb_26_col138, add_res_limb_27_col139 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }

    // ===== VerifyAdd252 (11 constraints, 9 intermediates) =====
    cuda_evaluator.add_constraint(mul(sub_p_bit_col140, sub(sub_p_bit_col140, m31(1))));
    m31 carry_tmp_1 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_2_col55, op1_limb_2_col86), mul(sub(add(add(op0_limb_1_col54, op1_limb_1_col85), mul(sub(sub(add(op0_limb_0_col53, op1_limb_0_col84), add_res_limb_0_col112), sub_p_bit_col140), m31(4194304))), add_res_limb_1_col113), m31(4194304))), add_res_limb_2_col114), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_1, sub(mul(carry_tmp_1, carry_tmp_1), m31(1))));
    m31 carry_tmp_2 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_5_col58, op1_limb_5_col89), mul(sub(add(add(op0_limb_4_col57, op1_limb_4_col88), mul(sub(add(add(op0_limb_3_col56, op1_limb_3_col87), carry_tmp_1), add_res_limb_3_col115), m31(4194304))), add_res_limb_4_col116), m31(4194304))), add_res_limb_5_col117), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_2, sub(mul(carry_tmp_2, carry_tmp_2), m31(1))));
    m31 carry_tmp_3 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_8_col61, op1_limb_8_col92), mul(sub(add(add(op0_limb_7_col60, op1_limb_7_col91), mul(sub(add(add(op0_limb_6_col59, op1_limb_6_col90), carry_tmp_2), add_res_limb_6_col118), m31(4194304))), add_res_limb_7_col119), m31(4194304))), add_res_limb_8_col120), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_3, sub(mul(carry_tmp_3, carry_tmp_3), m31(1))));
    m31 carry_tmp_4 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_11_col64, op1_limb_11_col95), mul(sub(add(add(op0_limb_10_col63, op1_limb_10_col94), mul(sub(add(add(op0_limb_9_col62, op1_limb_9_col93), carry_tmp_3), add_res_limb_9_col121), m31(4194304))), add_res_limb_10_col122), m31(4194304))), add_res_limb_11_col123), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_4, sub(mul(carry_tmp_4, carry_tmp_4), m31(1))));
    m31 carry_tmp_5 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_14_col67, op1_limb_14_col98), mul(sub(add(add(op0_limb_13_col66, op1_limb_13_col97), mul(sub(add(add(op0_limb_12_col65, op1_limb_12_col96), carry_tmp_4), add_res_limb_12_col124), m31(4194304))), add_res_limb_13_col125), m31(4194304))), add_res_limb_14_col126), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_5, sub(mul(carry_tmp_5, carry_tmp_5), m31(1))));
    m31 carry_tmp_6 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_17_col70, op1_limb_17_col101), mul(sub(add(add(op0_limb_16_col69, op1_limb_16_col100), mul(sub(add(add(op0_limb_15_col68, op1_limb_15_col99), carry_tmp_5), add_res_limb_15_col127), m31(4194304))), add_res_limb_16_col128), m31(4194304))), add_res_limb_17_col129), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_6, sub(mul(carry_tmp_6, carry_tmp_6), m31(1))));
    m31 carry_tmp_7 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_20_col73, op1_limb_20_col104), mul(sub(add(add(op0_limb_19_col72, op1_limb_19_col103), mul(sub(add(add(op0_limb_18_col71, op1_limb_18_col102), carry_tmp_6), add_res_limb_18_col130), m31(4194304))), add_res_limb_19_col131), m31(4194304))), add_res_limb_20_col132), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_7, sub(mul(carry_tmp_7, carry_tmp_7), m31(1))));
    m31 carry_tmp_8 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_23_col76, op1_limb_23_col107), mul(sub(add(add(op0_limb_22_col75, op1_limb_22_col106), mul(sub(sub(add(add(op0_limb_21_col74, op1_limb_21_col105), carry_tmp_7), add_res_limb_21_col133), mul(m31(136), sub_p_bit_col140)), m31(4194304))), add_res_limb_22_col134), m31(4194304))), add_res_limb_23_col135), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_8, sub(mul(carry_tmp_8, carry_tmp_8), m31(1))));
    m31 carry_tmp_9 = cuda_evaluator.add_intermediate(mul(sub(add(add(op0_limb_26_col79, op1_limb_26_col110), mul(sub(add(add(op0_limb_25_col78, op1_limb_25_col109), mul(sub(add(add(op0_limb_24_col77, op1_limb_24_col108), carry_tmp_8), add_res_limb_24_col136), m31(4194304))), add_res_limb_25_col137), m31(4194304))), add_res_limb_26_col138), m31(4194304)));
    cuda_evaluator.add_constraint(mul(carry_tmp_9, sub(mul(carry_tmp_9, carry_tmp_9), m31(1))));
    cuda_evaluator.add_constraint(sub(add(add(op0_limb_27_col80, op1_limb_27_col111), carry_tmp_9), add(add_res_limb_27_col139, mul(m31(256), sub_p_bit_col140))));

    // ===== Mul252: RangeCheckMemValueN28 for mul_res (14 relations) =====
    { m31 v[3] = { RANGE_CHECK_9_9_RELATION_ID, mul_res_limb_0_col141, mul_res_limb_1_col142 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_B_RELATION_ID, mul_res_limb_2_col143, mul_res_limb_3_col144 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_C_RELATION_ID, mul_res_limb_4_col145, mul_res_limb_5_col146 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_D_RELATION_ID, mul_res_limb_6_col147, mul_res_limb_7_col148 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_E_RELATION_ID, mul_res_limb_8_col149, mul_res_limb_9_col150 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_F_RELATION_ID, mul_res_limb_10_col151, mul_res_limb_11_col152 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_G_RELATION_ID, mul_res_limb_12_col153, mul_res_limb_13_col154 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_H_RELATION_ID, mul_res_limb_14_col155, mul_res_limb_15_col156 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_RELATION_ID, mul_res_limb_16_col157, mul_res_limb_17_col158 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_B_RELATION_ID, mul_res_limb_18_col159, mul_res_limb_19_col160 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_C_RELATION_ID, mul_res_limb_20_col161, mul_res_limb_21_col162 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_D_RELATION_ID, mul_res_limb_22_col163, mul_res_limb_23_col164 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_E_RELATION_ID, mul_res_limb_24_col165, mul_res_limb_25_col166 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[3] = { RANGE_CHECK_9_9_F_RELATION_ID, mul_res_limb_26_col167, mul_res_limb_27_col168 }; cuda_evaluator.add_to_relation<3>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }

    // ===== VerifyMul252 (28 constraints, 28 relations) =====
    // DoubleKaratsuba1454B -> SingleKaratsubaN7 (pure intermediates)

    m31 dk_sk1_z0_0 = cuda_evaluator.add_intermediate(mul(op0_limb_0_col53, op1_limb_0_col84));
    m31 dk_sk1_z0_1 = cuda_evaluator.add_intermediate(add(mul(op0_limb_0_col53, op1_limb_1_col85), mul(op0_limb_1_col54, op1_limb_0_col84)));
    m31 dk_sk1_z0_2 = cuda_evaluator.add_intermediate(add(add(mul(op0_limb_0_col53, op1_limb_2_col86), mul(op0_limb_1_col54, op1_limb_1_col85)), mul(op0_limb_2_col55, op1_limb_0_col84)));
    m31 dk_sk1_z0_3 = cuda_evaluator.add_intermediate(add(add(add(mul(op0_limb_0_col53, op1_limb_3_col87), mul(op0_limb_1_col54, op1_limb_2_col86)), mul(op0_limb_2_col55, op1_limb_1_col85)), mul(op0_limb_3_col56, op1_limb_0_col84)));
    m31 dk_sk1_z0_4 = cuda_evaluator.add_intermediate(add(add(add(add(mul(op0_limb_0_col53, op1_limb_4_col88), mul(op0_limb_1_col54, op1_limb_3_col87)), mul(op0_limb_2_col55, op1_limb_2_col86)), mul(op0_limb_3_col56, op1_limb_1_col85)), mul(op0_limb_4_col57, op1_limb_0_col84)));
    m31 dk_sk1_z0_5 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(op0_limb_0_col53, op1_limb_5_col89), mul(op0_limb_1_col54, op1_limb_4_col88)), mul(op0_limb_2_col55, op1_limb_3_col87)), mul(op0_limb_3_col56, op1_limb_2_col86)), mul(op0_limb_4_col57, op1_limb_1_col85)), mul(op0_limb_5_col58, op1_limb_0_col84)));
    m31 dk_sk1_z0_6 = cuda_evaluator.add_intermediate(add(add(add(add(add(add(mul(op0_limb_0_col53, op1_limb_6_col90), mul(op0_limb_1_col54, op1_limb_5_col89)), mul(op0_limb_2_col55, op1_limb_4_col88)), mul(op0_limb_3_col56, op1_limb_3_col87)), mul(op0_limb_4_col57, op1_limb_2_col86)), mul(op0_limb_5_col58, op1_limb_1_col85)), mul(op0_limb_6_col59, op1_limb_0_col84)));
    m31 dk_sk1_z0_7 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(op0_limb_1_col54, op1_limb_6_col90), mul(op0_limb_2_col55, op1_limb_5_col89)), mul(op0_limb_3_col56, op1_limb_4_col88)), mul(op0_limb_4_col57, op1_limb_3_col87)), mul(op0_limb_5_col58, op1_limb_2_col86)), mul(op0_limb_6_col59, op1_limb_1_col85)));
    m31 dk_sk1_z0_8 = cuda_evaluator.add_intermediate(add(add(add(add(mul(op0_limb_2_col55, op1_limb_6_col90), mul(op0_limb_3_col56, op1_limb_5_col89)), mul(op0_limb_4_col57, op1_limb_4_col88)), mul(op0_limb_5_col58, op1_limb_3_col87)), mul(op0_limb_6_col59, op1_limb_2_col86)));
    m31 dk_sk1_z0_9 = cuda_evaluator.add_intermediate(add(add(add(mul(op0_limb_3_col56, op1_limb_6_col90), mul(op0_limb_4_col57, op1_limb_5_col89)), mul(op0_limb_5_col58, op1_limb_4_col88)), mul(op0_limb_6_col59, op1_limb_3_col87)));
    m31 dk_sk1_z0_10 = cuda_evaluator.add_intermediate(add(add(mul(op0_limb_4_col57, op1_limb_6_col90), mul(op0_limb_5_col58, op1_limb_5_col89)), mul(op0_limb_6_col59, op1_limb_4_col88)));
    m31 dk_sk1_z0_11 = cuda_evaluator.add_intermediate(add(mul(op0_limb_5_col58, op1_limb_6_col90), mul(op0_limb_6_col59, op1_limb_5_col89)));
    m31 dk_sk1_z0_12 = cuda_evaluator.add_intermediate(mul(op0_limb_6_col59, op1_limb_6_col90));
    m31 dk_sk1_z2_0 = cuda_evaluator.add_intermediate(mul(op0_limb_7_col60, op1_limb_7_col91));
    m31 dk_sk1_z2_1 = cuda_evaluator.add_intermediate(add(mul(op0_limb_7_col60, op1_limb_8_col92), mul(op0_limb_8_col61, op1_limb_7_col91)));
    m31 dk_sk1_z2_2 = cuda_evaluator.add_intermediate(add(add(mul(op0_limb_7_col60, op1_limb_9_col93), mul(op0_limb_8_col61, op1_limb_8_col92)), mul(op0_limb_9_col62, op1_limb_7_col91)));
    m31 dk_sk1_z2_3 = cuda_evaluator.add_intermediate(add(add(add(mul(op0_limb_7_col60, op1_limb_10_col94), mul(op0_limb_8_col61, op1_limb_9_col93)), mul(op0_limb_9_col62, op1_limb_8_col92)), mul(op0_limb_10_col63, op1_limb_7_col91)));
    m31 dk_sk1_z2_4 = cuda_evaluator.add_intermediate(add(add(add(add(mul(op0_limb_7_col60, op1_limb_11_col95), mul(op0_limb_8_col61, op1_limb_10_col94)), mul(op0_limb_9_col62, op1_limb_9_col93)), mul(op0_limb_10_col63, op1_limb_8_col92)), mul(op0_limb_11_col64, op1_limb_7_col91)));
    m31 dk_sk1_z2_5 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(op0_limb_7_col60, op1_limb_12_col96), mul(op0_limb_8_col61, op1_limb_11_col95)), mul(op0_limb_9_col62, op1_limb_10_col94)), mul(op0_limb_10_col63, op1_limb_9_col93)), mul(op0_limb_11_col64, op1_limb_8_col92)), mul(op0_limb_12_col65, op1_limb_7_col91)));
    m31 dk_sk1_z2_6 = cuda_evaluator.add_intermediate(add(add(add(add(add(add(mul(op0_limb_7_col60, op1_limb_13_col97), mul(op0_limb_8_col61, op1_limb_12_col96)), mul(op0_limb_9_col62, op1_limb_11_col95)), mul(op0_limb_10_col63, op1_limb_10_col94)), mul(op0_limb_11_col64, op1_limb_9_col93)), mul(op0_limb_12_col65, op1_limb_8_col92)), mul(op0_limb_13_col66, op1_limb_7_col91)));
    m31 dk_sk1_z2_7 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(op0_limb_8_col61, op1_limb_13_col97), mul(op0_limb_9_col62, op1_limb_12_col96)), mul(op0_limb_10_col63, op1_limb_11_col95)), mul(op0_limb_11_col64, op1_limb_10_col94)), mul(op0_limb_12_col65, op1_limb_9_col93)), mul(op0_limb_13_col66, op1_limb_8_col92)));
    m31 dk_sk1_z2_8 = cuda_evaluator.add_intermediate(add(add(add(add(mul(op0_limb_9_col62, op1_limb_13_col97), mul(op0_limb_10_col63, op1_limb_12_col96)), mul(op0_limb_11_col64, op1_limb_11_col95)), mul(op0_limb_12_col65, op1_limb_10_col94)), mul(op0_limb_13_col66, op1_limb_9_col93)));
    m31 dk_sk1_z2_9 = cuda_evaluator.add_intermediate(add(add(add(mul(op0_limb_10_col63, op1_limb_13_col97), mul(op0_limb_11_col64, op1_limb_12_col96)), mul(op0_limb_12_col65, op1_limb_11_col95)), mul(op0_limb_13_col66, op1_limb_10_col94)));
    m31 dk_sk1_z2_10 = cuda_evaluator.add_intermediate(add(add(mul(op0_limb_11_col64, op1_limb_13_col97), mul(op0_limb_12_col65, op1_limb_12_col96)), mul(op0_limb_13_col66, op1_limb_11_col95)));
    m31 dk_sk1_z2_11 = cuda_evaluator.add_intermediate(add(mul(op0_limb_12_col65, op1_limb_13_col97), mul(op0_limb_13_col66, op1_limb_12_col96)));
    m31 dk_sk1_z2_12 = cuda_evaluator.add_intermediate(mul(op0_limb_13_col66, op1_limb_13_col97));
    m31 dk_sk1_xs_0 = cuda_evaluator.add_intermediate(add(op0_limb_0_col53, op0_limb_7_col60));
    m31 dk_sk1_xs_1 = cuda_evaluator.add_intermediate(add(op0_limb_1_col54, op0_limb_8_col61));
    m31 dk_sk1_xs_2 = cuda_evaluator.add_intermediate(add(op0_limb_2_col55, op0_limb_9_col62));
    m31 dk_sk1_xs_3 = cuda_evaluator.add_intermediate(add(op0_limb_3_col56, op0_limb_10_col63));
    m31 dk_sk1_xs_4 = cuda_evaluator.add_intermediate(add(op0_limb_4_col57, op0_limb_11_col64));
    m31 dk_sk1_xs_5 = cuda_evaluator.add_intermediate(add(op0_limb_5_col58, op0_limb_12_col65));
    m31 dk_sk1_xs_6 = cuda_evaluator.add_intermediate(add(op0_limb_6_col59, op0_limb_13_col66));
    m31 dk_sk1_ys_0 = cuda_evaluator.add_intermediate(add(op1_limb_0_col84, op1_limb_7_col91));
    m31 dk_sk1_ys_1 = cuda_evaluator.add_intermediate(add(op1_limb_1_col85, op1_limb_8_col92));
    m31 dk_sk1_ys_2 = cuda_evaluator.add_intermediate(add(op1_limb_2_col86, op1_limb_9_col93));
    m31 dk_sk1_ys_3 = cuda_evaluator.add_intermediate(add(op1_limb_3_col87, op1_limb_10_col94));
    m31 dk_sk1_ys_4 = cuda_evaluator.add_intermediate(add(op1_limb_4_col88, op1_limb_11_col95));
    m31 dk_sk1_ys_5 = cuda_evaluator.add_intermediate(add(op1_limb_5_col89, op1_limb_12_col96));
    m31 dk_sk1_ys_6 = cuda_evaluator.add_intermediate(add(op1_limb_6_col90, op1_limb_13_col97));
    m31 dk_sk2_z0_0 = cuda_evaluator.add_intermediate(mul(op0_limb_14_col67, op1_limb_14_col98));
    m31 dk_sk2_z0_1 = cuda_evaluator.add_intermediate(add(mul(op0_limb_14_col67, op1_limb_15_col99), mul(op0_limb_15_col68, op1_limb_14_col98)));
    m31 dk_sk2_z0_2 = cuda_evaluator.add_intermediate(add(add(mul(op0_limb_14_col67, op1_limb_16_col100), mul(op0_limb_15_col68, op1_limb_15_col99)), mul(op0_limb_16_col69, op1_limb_14_col98)));
    m31 dk_sk2_z0_3 = cuda_evaluator.add_intermediate(add(add(add(mul(op0_limb_14_col67, op1_limb_17_col101), mul(op0_limb_15_col68, op1_limb_16_col100)), mul(op0_limb_16_col69, op1_limb_15_col99)), mul(op0_limb_17_col70, op1_limb_14_col98)));
    m31 dk_sk2_z0_4 = cuda_evaluator.add_intermediate(add(add(add(add(mul(op0_limb_14_col67, op1_limb_18_col102), mul(op0_limb_15_col68, op1_limb_17_col101)), mul(op0_limb_16_col69, op1_limb_16_col100)), mul(op0_limb_17_col70, op1_limb_15_col99)), mul(op0_limb_18_col71, op1_limb_14_col98)));
    m31 dk_sk2_z0_5 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(op0_limb_14_col67, op1_limb_19_col103), mul(op0_limb_15_col68, op1_limb_18_col102)), mul(op0_limb_16_col69, op1_limb_17_col101)), mul(op0_limb_17_col70, op1_limb_16_col100)), mul(op0_limb_18_col71, op1_limb_15_col99)), mul(op0_limb_19_col72, op1_limb_14_col98)));
    m31 dk_sk2_z0_6 = cuda_evaluator.add_intermediate(add(add(add(add(add(add(mul(op0_limb_14_col67, op1_limb_20_col104), mul(op0_limb_15_col68, op1_limb_19_col103)), mul(op0_limb_16_col69, op1_limb_18_col102)), mul(op0_limb_17_col70, op1_limb_17_col101)), mul(op0_limb_18_col71, op1_limb_16_col100)), mul(op0_limb_19_col72, op1_limb_15_col99)), mul(op0_limb_20_col73, op1_limb_14_col98)));
    m31 dk_sk2_z0_7 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(op0_limb_15_col68, op1_limb_20_col104), mul(op0_limb_16_col69, op1_limb_19_col103)), mul(op0_limb_17_col70, op1_limb_18_col102)), mul(op0_limb_18_col71, op1_limb_17_col101)), mul(op0_limb_19_col72, op1_limb_16_col100)), mul(op0_limb_20_col73, op1_limb_15_col99)));
    m31 dk_sk2_z0_8 = cuda_evaluator.add_intermediate(add(add(add(add(mul(op0_limb_16_col69, op1_limb_20_col104), mul(op0_limb_17_col70, op1_limb_19_col103)), mul(op0_limb_18_col71, op1_limb_18_col102)), mul(op0_limb_19_col72, op1_limb_17_col101)), mul(op0_limb_20_col73, op1_limb_16_col100)));
    m31 dk_sk2_z0_9 = cuda_evaluator.add_intermediate(add(add(add(mul(op0_limb_17_col70, op1_limb_20_col104), mul(op0_limb_18_col71, op1_limb_19_col103)), mul(op0_limb_19_col72, op1_limb_18_col102)), mul(op0_limb_20_col73, op1_limb_17_col101)));
    m31 dk_sk2_z0_10 = cuda_evaluator.add_intermediate(add(add(mul(op0_limb_18_col71, op1_limb_20_col104), mul(op0_limb_19_col72, op1_limb_19_col103)), mul(op0_limb_20_col73, op1_limb_18_col102)));
    m31 dk_sk2_z0_11 = cuda_evaluator.add_intermediate(add(mul(op0_limb_19_col72, op1_limb_20_col104), mul(op0_limb_20_col73, op1_limb_19_col103)));
    m31 dk_sk2_z0_12 = cuda_evaluator.add_intermediate(mul(op0_limb_20_col73, op1_limb_20_col104));
    m31 dk_sk2_z2_0 = cuda_evaluator.add_intermediate(mul(op0_limb_21_col74, op1_limb_21_col105));
    m31 dk_sk2_z2_1 = cuda_evaluator.add_intermediate(add(mul(op0_limb_21_col74, op1_limb_22_col106), mul(op0_limb_22_col75, op1_limb_21_col105)));
    m31 dk_sk2_z2_2 = cuda_evaluator.add_intermediate(add(add(mul(op0_limb_21_col74, op1_limb_23_col107), mul(op0_limb_22_col75, op1_limb_22_col106)), mul(op0_limb_23_col76, op1_limb_21_col105)));
    m31 dk_sk2_z2_3 = cuda_evaluator.add_intermediate(add(add(add(mul(op0_limb_21_col74, op1_limb_24_col108), mul(op0_limb_22_col75, op1_limb_23_col107)), mul(op0_limb_23_col76, op1_limb_22_col106)), mul(op0_limb_24_col77, op1_limb_21_col105)));
    m31 dk_sk2_z2_4 = cuda_evaluator.add_intermediate(add(add(add(add(mul(op0_limb_21_col74, op1_limb_25_col109), mul(op0_limb_22_col75, op1_limb_24_col108)), mul(op0_limb_23_col76, op1_limb_23_col107)), mul(op0_limb_24_col77, op1_limb_22_col106)), mul(op0_limb_25_col78, op1_limb_21_col105)));
    m31 dk_sk2_z2_5 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(op0_limb_21_col74, op1_limb_26_col110), mul(op0_limb_22_col75, op1_limb_25_col109)), mul(op0_limb_23_col76, op1_limb_24_col108)), mul(op0_limb_24_col77, op1_limb_23_col107)), mul(op0_limb_25_col78, op1_limb_22_col106)), mul(op0_limb_26_col79, op1_limb_21_col105)));
    m31 dk_sk2_z2_6 = cuda_evaluator.add_intermediate(add(add(add(add(add(add(mul(op0_limb_21_col74, op1_limb_27_col111), mul(op0_limb_22_col75, op1_limb_26_col110)), mul(op0_limb_23_col76, op1_limb_25_col109)), mul(op0_limb_24_col77, op1_limb_24_col108)), mul(op0_limb_25_col78, op1_limb_23_col107)), mul(op0_limb_26_col79, op1_limb_22_col106)), mul(op0_limb_27_col80, op1_limb_21_col105)));
    m31 dk_sk2_z2_7 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(op0_limb_22_col75, op1_limb_27_col111), mul(op0_limb_23_col76, op1_limb_26_col110)), mul(op0_limb_24_col77, op1_limb_25_col109)), mul(op0_limb_25_col78, op1_limb_24_col108)), mul(op0_limb_26_col79, op1_limb_23_col107)), mul(op0_limb_27_col80, op1_limb_22_col106)));
    m31 dk_sk2_z2_8 = cuda_evaluator.add_intermediate(add(add(add(add(mul(op0_limb_23_col76, op1_limb_27_col111), mul(op0_limb_24_col77, op1_limb_26_col110)), mul(op0_limb_25_col78, op1_limb_25_col109)), mul(op0_limb_26_col79, op1_limb_24_col108)), mul(op0_limb_27_col80, op1_limb_23_col107)));
    m31 dk_sk2_z2_9 = cuda_evaluator.add_intermediate(add(add(add(mul(op0_limb_24_col77, op1_limb_27_col111), mul(op0_limb_25_col78, op1_limb_26_col110)), mul(op0_limb_26_col79, op1_limb_25_col109)), mul(op0_limb_27_col80, op1_limb_24_col108)));
    m31 dk_sk2_z2_10 = cuda_evaluator.add_intermediate(add(add(mul(op0_limb_25_col78, op1_limb_27_col111), mul(op0_limb_26_col79, op1_limb_26_col110)), mul(op0_limb_27_col80, op1_limb_25_col109)));
    m31 dk_sk2_z2_11 = cuda_evaluator.add_intermediate(add(mul(op0_limb_26_col79, op1_limb_27_col111), mul(op0_limb_27_col80, op1_limb_26_col110)));
    m31 dk_sk2_z2_12 = cuda_evaluator.add_intermediate(mul(op0_limb_27_col80, op1_limb_27_col111));
    m31 dk_sk2_xs_0 = cuda_evaluator.add_intermediate(add(op0_limb_14_col67, op0_limb_21_col74));
    m31 dk_sk2_xs_1 = cuda_evaluator.add_intermediate(add(op0_limb_15_col68, op0_limb_22_col75));
    m31 dk_sk2_xs_2 = cuda_evaluator.add_intermediate(add(op0_limb_16_col69, op0_limb_23_col76));
    m31 dk_sk2_xs_3 = cuda_evaluator.add_intermediate(add(op0_limb_17_col70, op0_limb_24_col77));
    m31 dk_sk2_xs_4 = cuda_evaluator.add_intermediate(add(op0_limb_18_col71, op0_limb_25_col78));
    m31 dk_sk2_xs_5 = cuda_evaluator.add_intermediate(add(op0_limb_19_col72, op0_limb_26_col79));
    m31 dk_sk2_xs_6 = cuda_evaluator.add_intermediate(add(op0_limb_20_col73, op0_limb_27_col80));
    m31 dk_sk2_ys_0 = cuda_evaluator.add_intermediate(add(op1_limb_14_col98, op1_limb_21_col105));
    m31 dk_sk2_ys_1 = cuda_evaluator.add_intermediate(add(op1_limb_15_col99, op1_limb_22_col106));
    m31 dk_sk2_ys_2 = cuda_evaluator.add_intermediate(add(op1_limb_16_col100, op1_limb_23_col107));
    m31 dk_sk2_ys_3 = cuda_evaluator.add_intermediate(add(op1_limb_17_col101, op1_limb_24_col108));
    m31 dk_sk2_ys_4 = cuda_evaluator.add_intermediate(add(op1_limb_18_col102, op1_limb_25_col109));
    m31 dk_sk2_ys_5 = cuda_evaluator.add_intermediate(add(op1_limb_19_col103, op1_limb_26_col110));
    m31 dk_sk2_ys_6 = cuda_evaluator.add_intermediate(add(op1_limb_20_col104, op1_limb_27_col111));
    m31 dk_xsum_0 = cuda_evaluator.add_intermediate(add(op0_limb_0_col53, op0_limb_14_col67));
    m31 dk_xsum_1 = cuda_evaluator.add_intermediate(add(op0_limb_1_col54, op0_limb_15_col68));
    m31 dk_xsum_2 = cuda_evaluator.add_intermediate(add(op0_limb_2_col55, op0_limb_16_col69));
    m31 dk_xsum_3 = cuda_evaluator.add_intermediate(add(op0_limb_3_col56, op0_limb_17_col70));
    m31 dk_xsum_4 = cuda_evaluator.add_intermediate(add(op0_limb_4_col57, op0_limb_18_col71));
    m31 dk_xsum_5 = cuda_evaluator.add_intermediate(add(op0_limb_5_col58, op0_limb_19_col72));
    m31 dk_xsum_6 = cuda_evaluator.add_intermediate(add(op0_limb_6_col59, op0_limb_20_col73));
    m31 dk_xsum_7 = cuda_evaluator.add_intermediate(add(op0_limb_7_col60, op0_limb_21_col74));
    m31 dk_xsum_8 = cuda_evaluator.add_intermediate(add(op0_limb_8_col61, op0_limb_22_col75));
    m31 dk_xsum_9 = cuda_evaluator.add_intermediate(add(op0_limb_9_col62, op0_limb_23_col76));
    m31 dk_xsum_10 = cuda_evaluator.add_intermediate(add(op0_limb_10_col63, op0_limb_24_col77));
    m31 dk_xsum_11 = cuda_evaluator.add_intermediate(add(op0_limb_11_col64, op0_limb_25_col78));
    m31 dk_xsum_12 = cuda_evaluator.add_intermediate(add(op0_limb_12_col65, op0_limb_26_col79));
    m31 dk_xsum_13 = cuda_evaluator.add_intermediate(add(op0_limb_13_col66, op0_limb_27_col80));
    m31 dk_ysum_0 = cuda_evaluator.add_intermediate(add(op1_limb_0_col84, op1_limb_14_col98));
    m31 dk_ysum_1 = cuda_evaluator.add_intermediate(add(op1_limb_1_col85, op1_limb_15_col99));
    m31 dk_ysum_2 = cuda_evaluator.add_intermediate(add(op1_limb_2_col86, op1_limb_16_col100));
    m31 dk_ysum_3 = cuda_evaluator.add_intermediate(add(op1_limb_3_col87, op1_limb_17_col101));
    m31 dk_ysum_4 = cuda_evaluator.add_intermediate(add(op1_limb_4_col88, op1_limb_18_col102));
    m31 dk_ysum_5 = cuda_evaluator.add_intermediate(add(op1_limb_5_col89, op1_limb_19_col103));
    m31 dk_ysum_6 = cuda_evaluator.add_intermediate(add(op1_limb_6_col90, op1_limb_20_col104));
    m31 dk_ysum_7 = cuda_evaluator.add_intermediate(add(op1_limb_7_col91, op1_limb_21_col105));
    m31 dk_ysum_8 = cuda_evaluator.add_intermediate(add(op1_limb_8_col92, op1_limb_22_col106));
    m31 dk_ysum_9 = cuda_evaluator.add_intermediate(add(op1_limb_9_col93, op1_limb_23_col107));
    m31 dk_ysum_10 = cuda_evaluator.add_intermediate(add(op1_limb_10_col94, op1_limb_24_col108));
    m31 dk_ysum_11 = cuda_evaluator.add_intermediate(add(op1_limb_11_col95, op1_limb_25_col109));
    m31 dk_ysum_12 = cuda_evaluator.add_intermediate(add(op1_limb_12_col96, op1_limb_26_col110));
    m31 dk_ysum_13 = cuda_evaluator.add_intermediate(add(op1_limb_13_col97, op1_limb_27_col111));
    m31 dk_sk3_z0_0 = cuda_evaluator.add_intermediate(mul(dk_xsum_0, dk_ysum_0));
    m31 dk_sk3_z0_1 = cuda_evaluator.add_intermediate(add(mul(dk_xsum_0, dk_ysum_1), mul(dk_xsum_1, dk_ysum_0)));
    m31 dk_sk3_z0_2 = cuda_evaluator.add_intermediate(add(add(mul(dk_xsum_0, dk_ysum_2), mul(dk_xsum_1, dk_ysum_1)), mul(dk_xsum_2, dk_ysum_0)));
    m31 dk_sk3_z0_3 = cuda_evaluator.add_intermediate(add(add(add(mul(dk_xsum_0, dk_ysum_3), mul(dk_xsum_1, dk_ysum_2)), mul(dk_xsum_2, dk_ysum_1)), mul(dk_xsum_3, dk_ysum_0)));
    m31 dk_sk3_z0_4 = cuda_evaluator.add_intermediate(add(add(add(add(mul(dk_xsum_0, dk_ysum_4), mul(dk_xsum_1, dk_ysum_3)), mul(dk_xsum_2, dk_ysum_2)), mul(dk_xsum_3, dk_ysum_1)), mul(dk_xsum_4, dk_ysum_0)));
    m31 dk_sk3_z0_5 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(dk_xsum_0, dk_ysum_5), mul(dk_xsum_1, dk_ysum_4)), mul(dk_xsum_2, dk_ysum_3)), mul(dk_xsum_3, dk_ysum_2)), mul(dk_xsum_4, dk_ysum_1)), mul(dk_xsum_5, dk_ysum_0)));
    m31 dk_sk3_z0_6 = cuda_evaluator.add_intermediate(add(add(add(add(add(add(mul(dk_xsum_0, dk_ysum_6), mul(dk_xsum_1, dk_ysum_5)), mul(dk_xsum_2, dk_ysum_4)), mul(dk_xsum_3, dk_ysum_3)), mul(dk_xsum_4, dk_ysum_2)), mul(dk_xsum_5, dk_ysum_1)), mul(dk_xsum_6, dk_ysum_0)));
    m31 dk_sk3_z0_7 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(dk_xsum_1, dk_ysum_6), mul(dk_xsum_2, dk_ysum_5)), mul(dk_xsum_3, dk_ysum_4)), mul(dk_xsum_4, dk_ysum_3)), mul(dk_xsum_5, dk_ysum_2)), mul(dk_xsum_6, dk_ysum_1)));
    m31 dk_sk3_z0_8 = cuda_evaluator.add_intermediate(add(add(add(add(mul(dk_xsum_2, dk_ysum_6), mul(dk_xsum_3, dk_ysum_5)), mul(dk_xsum_4, dk_ysum_4)), mul(dk_xsum_5, dk_ysum_3)), mul(dk_xsum_6, dk_ysum_2)));
    m31 dk_sk3_z0_9 = cuda_evaluator.add_intermediate(add(add(add(mul(dk_xsum_3, dk_ysum_6), mul(dk_xsum_4, dk_ysum_5)), mul(dk_xsum_5, dk_ysum_4)), mul(dk_xsum_6, dk_ysum_3)));
    m31 dk_sk3_z0_10 = cuda_evaluator.add_intermediate(add(add(mul(dk_xsum_4, dk_ysum_6), mul(dk_xsum_5, dk_ysum_5)), mul(dk_xsum_6, dk_ysum_4)));
    m31 dk_sk3_z0_11 = cuda_evaluator.add_intermediate(add(mul(dk_xsum_5, dk_ysum_6), mul(dk_xsum_6, dk_ysum_5)));
    m31 dk_sk3_z0_12 = cuda_evaluator.add_intermediate(mul(dk_xsum_6, dk_ysum_6));
    m31 dk_sk3_z2_0 = cuda_evaluator.add_intermediate(mul(dk_xsum_7, dk_ysum_7));
    m31 dk_sk3_z2_1 = cuda_evaluator.add_intermediate(add(mul(dk_xsum_7, dk_ysum_8), mul(dk_xsum_8, dk_ysum_7)));
    m31 dk_sk3_z2_2 = cuda_evaluator.add_intermediate(add(add(mul(dk_xsum_7, dk_ysum_9), mul(dk_xsum_8, dk_ysum_8)), mul(dk_xsum_9, dk_ysum_7)));
    m31 dk_sk3_z2_3 = cuda_evaluator.add_intermediate(add(add(add(mul(dk_xsum_7, dk_ysum_10), mul(dk_xsum_8, dk_ysum_9)), mul(dk_xsum_9, dk_ysum_8)), mul(dk_xsum_10, dk_ysum_7)));
    m31 dk_sk3_z2_4 = cuda_evaluator.add_intermediate(add(add(add(add(mul(dk_xsum_7, dk_ysum_11), mul(dk_xsum_8, dk_ysum_10)), mul(dk_xsum_9, dk_ysum_9)), mul(dk_xsum_10, dk_ysum_8)), mul(dk_xsum_11, dk_ysum_7)));
    m31 dk_sk3_z2_5 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(dk_xsum_7, dk_ysum_12), mul(dk_xsum_8, dk_ysum_11)), mul(dk_xsum_9, dk_ysum_10)), mul(dk_xsum_10, dk_ysum_9)), mul(dk_xsum_11, dk_ysum_8)), mul(dk_xsum_12, dk_ysum_7)));
    m31 dk_sk3_z2_6 = cuda_evaluator.add_intermediate(add(add(add(add(add(add(mul(dk_xsum_7, dk_ysum_13), mul(dk_xsum_8, dk_ysum_12)), mul(dk_xsum_9, dk_ysum_11)), mul(dk_xsum_10, dk_ysum_10)), mul(dk_xsum_11, dk_ysum_9)), mul(dk_xsum_12, dk_ysum_8)), mul(dk_xsum_13, dk_ysum_7)));
    m31 dk_sk3_z2_7 = cuda_evaluator.add_intermediate(add(add(add(add(add(mul(dk_xsum_8, dk_ysum_13), mul(dk_xsum_9, dk_ysum_12)), mul(dk_xsum_10, dk_ysum_11)), mul(dk_xsum_11, dk_ysum_10)), mul(dk_xsum_12, dk_ysum_9)), mul(dk_xsum_13, dk_ysum_8)));
    m31 dk_sk3_z2_8 = cuda_evaluator.add_intermediate(add(add(add(add(mul(dk_xsum_9, dk_ysum_13), mul(dk_xsum_10, dk_ysum_12)), mul(dk_xsum_11, dk_ysum_11)), mul(dk_xsum_12, dk_ysum_10)), mul(dk_xsum_13, dk_ysum_9)));
    m31 dk_sk3_z2_9 = cuda_evaluator.add_intermediate(add(add(add(mul(dk_xsum_10, dk_ysum_13), mul(dk_xsum_11, dk_ysum_12)), mul(dk_xsum_12, dk_ysum_11)), mul(dk_xsum_13, dk_ysum_10)));
    m31 dk_sk3_z2_10 = cuda_evaluator.add_intermediate(add(add(mul(dk_xsum_11, dk_ysum_13), mul(dk_xsum_12, dk_ysum_12)), mul(dk_xsum_13, dk_ysum_11)));
    m31 dk_sk3_z2_11 = cuda_evaluator.add_intermediate(add(mul(dk_xsum_12, dk_ysum_13), mul(dk_xsum_13, dk_ysum_12)));
    m31 dk_sk3_z2_12 = cuda_evaluator.add_intermediate(mul(dk_xsum_13, dk_ysum_13));
    m31 dk_sk3_xs_0 = cuda_evaluator.add_intermediate(add(dk_xsum_0, dk_xsum_7));
    m31 dk_sk3_xs_1 = cuda_evaluator.add_intermediate(add(dk_xsum_1, dk_xsum_8));
    m31 dk_sk3_xs_2 = cuda_evaluator.add_intermediate(add(dk_xsum_2, dk_xsum_9));
    m31 dk_sk3_xs_3 = cuda_evaluator.add_intermediate(add(dk_xsum_3, dk_xsum_10));
    m31 dk_sk3_xs_4 = cuda_evaluator.add_intermediate(add(dk_xsum_4, dk_xsum_11));
    m31 dk_sk3_xs_5 = cuda_evaluator.add_intermediate(add(dk_xsum_5, dk_xsum_12));
    m31 dk_sk3_xs_6 = cuda_evaluator.add_intermediate(add(dk_xsum_6, dk_xsum_13));
    m31 dk_sk3_ys_0 = cuda_evaluator.add_intermediate(add(dk_ysum_0, dk_ysum_7));
    m31 dk_sk3_ys_1 = cuda_evaluator.add_intermediate(add(dk_ysum_1, dk_ysum_8));
    m31 dk_sk3_ys_2 = cuda_evaluator.add_intermediate(add(dk_ysum_2, dk_ysum_9));
    m31 dk_sk3_ys_3 = cuda_evaluator.add_intermediate(add(dk_ysum_3, dk_ysum_10));
    m31 dk_sk3_ys_4 = cuda_evaluator.add_intermediate(add(dk_ysum_4, dk_ysum_11));
    m31 dk_sk3_ys_5 = cuda_evaluator.add_intermediate(add(dk_ysum_5, dk_ysum_12));
    m31 dk_sk3_ys_6 = cuda_evaluator.add_intermediate(add(dk_ysum_6, dk_ysum_13));

    // VerifyMul252: conv = dk_out - c for 0..27, dk_out for 28..54
    m31 conv_0 = cuda_evaluator.add_intermediate(sub(dk_sk1_z0_0, mul_res_limb_0_col141));
    m31 conv_1 = cuda_evaluator.add_intermediate(sub(dk_sk1_z0_1, mul_res_limb_1_col142));
    m31 conv_2 = cuda_evaluator.add_intermediate(sub(dk_sk1_z0_2, mul_res_limb_2_col143));
    m31 conv_3 = cuda_evaluator.add_intermediate(sub(dk_sk1_z0_3, mul_res_limb_3_col144));
    m31 conv_4 = cuda_evaluator.add_intermediate(sub(dk_sk1_z0_4, mul_res_limb_4_col145));
    m31 conv_5 = cuda_evaluator.add_intermediate(sub(dk_sk1_z0_5, mul_res_limb_5_col146));
    m31 conv_6 = cuda_evaluator.add_intermediate(sub(dk_sk1_z0_6, mul_res_limb_6_col147));
    m31 conv_7 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z0_7, sub(sub(mul(dk_sk1_xs_0, dk_sk1_ys_0), dk_sk1_z0_0), dk_sk1_z2_0)), mul_res_limb_7_col148));
    m31 conv_8 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z0_8, sub(sub(add(mul(dk_sk1_xs_0, dk_sk1_ys_1), mul(dk_sk1_xs_1, dk_sk1_ys_0)), dk_sk1_z0_1), dk_sk1_z2_1)), mul_res_limb_8_col149));
    m31 conv_9 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z0_9, sub(sub(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_2), mul(dk_sk1_xs_1, dk_sk1_ys_1)), mul(dk_sk1_xs_2, dk_sk1_ys_0)), dk_sk1_z0_2), dk_sk1_z2_2)), mul_res_limb_9_col150));
    m31 conv_10 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z0_10, sub(sub(add(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_3), mul(dk_sk1_xs_1, dk_sk1_ys_2)), mul(dk_sk1_xs_2, dk_sk1_ys_1)), mul(dk_sk1_xs_3, dk_sk1_ys_0)), dk_sk1_z0_3), dk_sk1_z2_3)), mul_res_limb_10_col151));
    m31 conv_11 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z0_11, sub(sub(add(add(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_4), mul(dk_sk1_xs_1, dk_sk1_ys_3)), mul(dk_sk1_xs_2, dk_sk1_ys_2)), mul(dk_sk1_xs_3, dk_sk1_ys_1)), mul(dk_sk1_xs_4, dk_sk1_ys_0)), dk_sk1_z0_4), dk_sk1_z2_4)), mul_res_limb_11_col152));
    m31 conv_12 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z0_12, sub(sub(add(add(add(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_5), mul(dk_sk1_xs_1, dk_sk1_ys_4)), mul(dk_sk1_xs_2, dk_sk1_ys_3)), mul(dk_sk1_xs_3, dk_sk1_ys_2)), mul(dk_sk1_xs_4, dk_sk1_ys_1)), mul(dk_sk1_xs_5, dk_sk1_ys_0)), dk_sk1_z0_5), dk_sk1_z2_5)), mul_res_limb_12_col153));
    m31 conv_13 = cuda_evaluator.add_intermediate(sub(sub(sub(add(add(add(add(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_6), mul(dk_sk1_xs_1, dk_sk1_ys_5)), mul(dk_sk1_xs_2, dk_sk1_ys_4)), mul(dk_sk1_xs_3, dk_sk1_ys_3)), mul(dk_sk1_xs_4, dk_sk1_ys_2)), mul(dk_sk1_xs_5, dk_sk1_ys_1)), mul(dk_sk1_xs_6, dk_sk1_ys_0)), dk_sk1_z0_6), dk_sk1_z2_6), mul_res_limb_13_col154));
    m31 conv_14 = cuda_evaluator.add_intermediate(sub(add(add(dk_sk1_z2_0, sub(sub(add(add(add(add(add(mul(dk_sk1_xs_1, dk_sk1_ys_6), mul(dk_sk1_xs_2, dk_sk1_ys_5)), mul(dk_sk1_xs_3, dk_sk1_ys_4)), mul(dk_sk1_xs_4, dk_sk1_ys_3)), mul(dk_sk1_xs_5, dk_sk1_ys_2)), mul(dk_sk1_xs_6, dk_sk1_ys_1)), dk_sk1_z0_7), dk_sk1_z2_7)), sub(sub(dk_sk3_z0_0, dk_sk1_z0_0), dk_sk2_z0_0)), mul_res_limb_14_col155));
    m31 conv_15 = cuda_evaluator.add_intermediate(sub(add(add(dk_sk1_z2_1, sub(sub(add(add(add(add(mul(dk_sk1_xs_2, dk_sk1_ys_6), mul(dk_sk1_xs_3, dk_sk1_ys_5)), mul(dk_sk1_xs_4, dk_sk1_ys_4)), mul(dk_sk1_xs_5, dk_sk1_ys_3)), mul(dk_sk1_xs_6, dk_sk1_ys_2)), dk_sk1_z0_8), dk_sk1_z2_8)), sub(sub(dk_sk3_z0_1, dk_sk1_z0_1), dk_sk2_z0_1)), mul_res_limb_15_col156));
    m31 conv_16 = cuda_evaluator.add_intermediate(sub(add(add(dk_sk1_z2_2, sub(sub(add(add(add(mul(dk_sk1_xs_3, dk_sk1_ys_6), mul(dk_sk1_xs_4, dk_sk1_ys_5)), mul(dk_sk1_xs_5, dk_sk1_ys_4)), mul(dk_sk1_xs_6, dk_sk1_ys_3)), dk_sk1_z0_9), dk_sk1_z2_9)), sub(sub(dk_sk3_z0_2, dk_sk1_z0_2), dk_sk2_z0_2)), mul_res_limb_16_col157));
    m31 conv_17 = cuda_evaluator.add_intermediate(sub(add(add(dk_sk1_z2_3, sub(sub(add(add(mul(dk_sk1_xs_4, dk_sk1_ys_6), mul(dk_sk1_xs_5, dk_sk1_ys_5)), mul(dk_sk1_xs_6, dk_sk1_ys_4)), dk_sk1_z0_10), dk_sk1_z2_10)), sub(sub(dk_sk3_z0_3, dk_sk1_z0_3), dk_sk2_z0_3)), mul_res_limb_17_col158));
    m31 conv_18 = cuda_evaluator.add_intermediate(sub(add(add(dk_sk1_z2_4, sub(sub(add(mul(dk_sk1_xs_5, dk_sk1_ys_6), mul(dk_sk1_xs_6, dk_sk1_ys_5)), dk_sk1_z0_11), dk_sk1_z2_11)), sub(sub(dk_sk3_z0_4, dk_sk1_z0_4), dk_sk2_z0_4)), mul_res_limb_18_col159));
    m31 conv_19 = cuda_evaluator.add_intermediate(sub(add(add(dk_sk1_z2_5, sub(sub(mul(dk_sk1_xs_6, dk_sk1_ys_6), dk_sk1_z0_12), dk_sk1_z2_12)), sub(sub(dk_sk3_z0_5, dk_sk1_z0_5), dk_sk2_z0_5)), mul_res_limb_19_col160));
    m31 conv_20 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z2_6, sub(sub(dk_sk3_z0_6, dk_sk1_z0_6), dk_sk2_z0_6)), mul_res_limb_20_col161));
    m31 conv_21 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z2_7, sub(sub(add(dk_sk3_z0_7, sub(sub(mul(dk_sk3_xs_0, dk_sk3_ys_0), dk_sk3_z0_0), dk_sk3_z2_0)), add(dk_sk1_z0_7, sub(sub(mul(dk_sk1_xs_0, dk_sk1_ys_0), dk_sk1_z0_0), dk_sk1_z2_0))), add(dk_sk2_z0_7, sub(sub(mul(dk_sk2_xs_0, dk_sk2_ys_0), dk_sk2_z0_0), dk_sk2_z2_0)))), mul_res_limb_21_col162));
    m31 conv_22 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z2_8, sub(sub(add(dk_sk3_z0_8, sub(sub(add(mul(dk_sk3_xs_0, dk_sk3_ys_1), mul(dk_sk3_xs_1, dk_sk3_ys_0)), dk_sk3_z0_1), dk_sk3_z2_1)), add(dk_sk1_z0_8, sub(sub(add(mul(dk_sk1_xs_0, dk_sk1_ys_1), mul(dk_sk1_xs_1, dk_sk1_ys_0)), dk_sk1_z0_1), dk_sk1_z2_1))), add(dk_sk2_z0_8, sub(sub(add(mul(dk_sk2_xs_0, dk_sk2_ys_1), mul(dk_sk2_xs_1, dk_sk2_ys_0)), dk_sk2_z0_1), dk_sk2_z2_1)))), mul_res_limb_22_col163));
    m31 conv_23 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z2_9, sub(sub(add(dk_sk3_z0_9, sub(sub(add(add(mul(dk_sk3_xs_0, dk_sk3_ys_2), mul(dk_sk3_xs_1, dk_sk3_ys_1)), mul(dk_sk3_xs_2, dk_sk3_ys_0)), dk_sk3_z0_2), dk_sk3_z2_2)), add(dk_sk1_z0_9, sub(sub(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_2), mul(dk_sk1_xs_1, dk_sk1_ys_1)), mul(dk_sk1_xs_2, dk_sk1_ys_0)), dk_sk1_z0_2), dk_sk1_z2_2))), add(dk_sk2_z0_9, sub(sub(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_2), mul(dk_sk2_xs_1, dk_sk2_ys_1)), mul(dk_sk2_xs_2, dk_sk2_ys_0)), dk_sk2_z0_2), dk_sk2_z2_2)))), mul_res_limb_23_col164));
    m31 conv_24 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z2_10, sub(sub(add(dk_sk3_z0_10, sub(sub(add(add(add(mul(dk_sk3_xs_0, dk_sk3_ys_3), mul(dk_sk3_xs_1, dk_sk3_ys_2)), mul(dk_sk3_xs_2, dk_sk3_ys_1)), mul(dk_sk3_xs_3, dk_sk3_ys_0)), dk_sk3_z0_3), dk_sk3_z2_3)), add(dk_sk1_z0_10, sub(sub(add(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_3), mul(dk_sk1_xs_1, dk_sk1_ys_2)), mul(dk_sk1_xs_2, dk_sk1_ys_1)), mul(dk_sk1_xs_3, dk_sk1_ys_0)), dk_sk1_z0_3), dk_sk1_z2_3))), add(dk_sk2_z0_10, sub(sub(add(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_3), mul(dk_sk2_xs_1, dk_sk2_ys_2)), mul(dk_sk2_xs_2, dk_sk2_ys_1)), mul(dk_sk2_xs_3, dk_sk2_ys_0)), dk_sk2_z0_3), dk_sk2_z2_3)))), mul_res_limb_24_col165));
    m31 conv_25 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z2_11, sub(sub(add(dk_sk3_z0_11, sub(sub(add(add(add(add(mul(dk_sk3_xs_0, dk_sk3_ys_4), mul(dk_sk3_xs_1, dk_sk3_ys_3)), mul(dk_sk3_xs_2, dk_sk3_ys_2)), mul(dk_sk3_xs_3, dk_sk3_ys_1)), mul(dk_sk3_xs_4, dk_sk3_ys_0)), dk_sk3_z0_4), dk_sk3_z2_4)), add(dk_sk1_z0_11, sub(sub(add(add(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_4), mul(dk_sk1_xs_1, dk_sk1_ys_3)), mul(dk_sk1_xs_2, dk_sk1_ys_2)), mul(dk_sk1_xs_3, dk_sk1_ys_1)), mul(dk_sk1_xs_4, dk_sk1_ys_0)), dk_sk1_z0_4), dk_sk1_z2_4))), add(dk_sk2_z0_11, sub(sub(add(add(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_4), mul(dk_sk2_xs_1, dk_sk2_ys_3)), mul(dk_sk2_xs_2, dk_sk2_ys_2)), mul(dk_sk2_xs_3, dk_sk2_ys_1)), mul(dk_sk2_xs_4, dk_sk2_ys_0)), dk_sk2_z0_4), dk_sk2_z2_4)))), mul_res_limb_25_col166));
    m31 conv_26 = cuda_evaluator.add_intermediate(sub(add(dk_sk1_z2_12, sub(sub(add(dk_sk3_z0_12, sub(sub(add(add(add(add(add(mul(dk_sk3_xs_0, dk_sk3_ys_5), mul(dk_sk3_xs_1, dk_sk3_ys_4)), mul(dk_sk3_xs_2, dk_sk3_ys_3)), mul(dk_sk3_xs_3, dk_sk3_ys_2)), mul(dk_sk3_xs_4, dk_sk3_ys_1)), mul(dk_sk3_xs_5, dk_sk3_ys_0)), dk_sk3_z0_5), dk_sk3_z2_5)), add(dk_sk1_z0_12, sub(sub(add(add(add(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_5), mul(dk_sk1_xs_1, dk_sk1_ys_4)), mul(dk_sk1_xs_2, dk_sk1_ys_3)), mul(dk_sk1_xs_3, dk_sk1_ys_2)), mul(dk_sk1_xs_4, dk_sk1_ys_1)), mul(dk_sk1_xs_5, dk_sk1_ys_0)), dk_sk1_z0_5), dk_sk1_z2_5))), add(dk_sk2_z0_12, sub(sub(add(add(add(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_5), mul(dk_sk2_xs_1, dk_sk2_ys_4)), mul(dk_sk2_xs_2, dk_sk2_ys_3)), mul(dk_sk2_xs_3, dk_sk2_ys_2)), mul(dk_sk2_xs_4, dk_sk2_ys_1)), mul(dk_sk2_xs_5, dk_sk2_ys_0)), dk_sk2_z0_5), dk_sk2_z2_5)))), mul_res_limb_26_col167));
    m31 conv_27 = cuda_evaluator.add_intermediate(sub(sub(sub(sub(sub(add(add(add(add(add(add(mul(dk_sk3_xs_0, dk_sk3_ys_6), mul(dk_sk3_xs_1, dk_sk3_ys_5)), mul(dk_sk3_xs_2, dk_sk3_ys_4)), mul(dk_sk3_xs_3, dk_sk3_ys_3)), mul(dk_sk3_xs_4, dk_sk3_ys_2)), mul(dk_sk3_xs_5, dk_sk3_ys_1)), mul(dk_sk3_xs_6, dk_sk3_ys_0)), dk_sk3_z0_6), dk_sk3_z2_6), sub(sub(add(add(add(add(add(add(mul(dk_sk1_xs_0, dk_sk1_ys_6), mul(dk_sk1_xs_1, dk_sk1_ys_5)), mul(dk_sk1_xs_2, dk_sk1_ys_4)), mul(dk_sk1_xs_3, dk_sk1_ys_3)), mul(dk_sk1_xs_4, dk_sk1_ys_2)), mul(dk_sk1_xs_5, dk_sk1_ys_1)), mul(dk_sk1_xs_6, dk_sk1_ys_0)), dk_sk1_z0_6), dk_sk1_z2_6)), sub(sub(add(add(add(add(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_6), mul(dk_sk2_xs_1, dk_sk2_ys_5)), mul(dk_sk2_xs_2, dk_sk2_ys_4)), mul(dk_sk2_xs_3, dk_sk2_ys_3)), mul(dk_sk2_xs_4, dk_sk2_ys_2)), mul(dk_sk2_xs_5, dk_sk2_ys_1)), mul(dk_sk2_xs_6, dk_sk2_ys_0)), dk_sk2_z0_6), dk_sk2_z2_6)), mul_res_limb_27_col168));
    m31 conv_28 = cuda_evaluator.add_intermediate(add(dk_sk2_z0_0, sub(sub(add(dk_sk3_z2_0, sub(sub(add(add(add(add(add(mul(dk_sk3_xs_1, dk_sk3_ys_6), mul(dk_sk3_xs_2, dk_sk3_ys_5)), mul(dk_sk3_xs_3, dk_sk3_ys_4)), mul(dk_sk3_xs_4, dk_sk3_ys_3)), mul(dk_sk3_xs_5, dk_sk3_ys_2)), mul(dk_sk3_xs_6, dk_sk3_ys_1)), dk_sk3_z0_7), dk_sk3_z2_7)), add(dk_sk1_z2_0, sub(sub(add(add(add(add(add(mul(dk_sk1_xs_1, dk_sk1_ys_6), mul(dk_sk1_xs_2, dk_sk1_ys_5)), mul(dk_sk1_xs_3, dk_sk1_ys_4)), mul(dk_sk1_xs_4, dk_sk1_ys_3)), mul(dk_sk1_xs_5, dk_sk1_ys_2)), mul(dk_sk1_xs_6, dk_sk1_ys_1)), dk_sk1_z0_7), dk_sk1_z2_7))), add(dk_sk2_z2_0, sub(sub(add(add(add(add(add(mul(dk_sk2_xs_1, dk_sk2_ys_6), mul(dk_sk2_xs_2, dk_sk2_ys_5)), mul(dk_sk2_xs_3, dk_sk2_ys_4)), mul(dk_sk2_xs_4, dk_sk2_ys_3)), mul(dk_sk2_xs_5, dk_sk2_ys_2)), mul(dk_sk2_xs_6, dk_sk2_ys_1)), dk_sk2_z0_7), dk_sk2_z2_7)))));
    m31 conv_29 = cuda_evaluator.add_intermediate(add(dk_sk2_z0_1, sub(sub(add(dk_sk3_z2_1, sub(sub(add(add(add(add(mul(dk_sk3_xs_2, dk_sk3_ys_6), mul(dk_sk3_xs_3, dk_sk3_ys_5)), mul(dk_sk3_xs_4, dk_sk3_ys_4)), mul(dk_sk3_xs_5, dk_sk3_ys_3)), mul(dk_sk3_xs_6, dk_sk3_ys_2)), dk_sk3_z0_8), dk_sk3_z2_8)), add(dk_sk1_z2_1, sub(sub(add(add(add(add(mul(dk_sk1_xs_2, dk_sk1_ys_6), mul(dk_sk1_xs_3, dk_sk1_ys_5)), mul(dk_sk1_xs_4, dk_sk1_ys_4)), mul(dk_sk1_xs_5, dk_sk1_ys_3)), mul(dk_sk1_xs_6, dk_sk1_ys_2)), dk_sk1_z0_8), dk_sk1_z2_8))), add(dk_sk2_z2_1, sub(sub(add(add(add(add(mul(dk_sk2_xs_2, dk_sk2_ys_6), mul(dk_sk2_xs_3, dk_sk2_ys_5)), mul(dk_sk2_xs_4, dk_sk2_ys_4)), mul(dk_sk2_xs_5, dk_sk2_ys_3)), mul(dk_sk2_xs_6, dk_sk2_ys_2)), dk_sk2_z0_8), dk_sk2_z2_8)))));
    m31 conv_30 = cuda_evaluator.add_intermediate(add(dk_sk2_z0_2, sub(sub(add(dk_sk3_z2_2, sub(sub(add(add(add(mul(dk_sk3_xs_3, dk_sk3_ys_6), mul(dk_sk3_xs_4, dk_sk3_ys_5)), mul(dk_sk3_xs_5, dk_sk3_ys_4)), mul(dk_sk3_xs_6, dk_sk3_ys_3)), dk_sk3_z0_9), dk_sk3_z2_9)), add(dk_sk1_z2_2, sub(sub(add(add(add(mul(dk_sk1_xs_3, dk_sk1_ys_6), mul(dk_sk1_xs_4, dk_sk1_ys_5)), mul(dk_sk1_xs_5, dk_sk1_ys_4)), mul(dk_sk1_xs_6, dk_sk1_ys_3)), dk_sk1_z0_9), dk_sk1_z2_9))), add(dk_sk2_z2_2, sub(sub(add(add(add(mul(dk_sk2_xs_3, dk_sk2_ys_6), mul(dk_sk2_xs_4, dk_sk2_ys_5)), mul(dk_sk2_xs_5, dk_sk2_ys_4)), mul(dk_sk2_xs_6, dk_sk2_ys_3)), dk_sk2_z0_9), dk_sk2_z2_9)))));
    m31 conv_31 = cuda_evaluator.add_intermediate(add(dk_sk2_z0_3, sub(sub(add(dk_sk3_z2_3, sub(sub(add(add(mul(dk_sk3_xs_4, dk_sk3_ys_6), mul(dk_sk3_xs_5, dk_sk3_ys_5)), mul(dk_sk3_xs_6, dk_sk3_ys_4)), dk_sk3_z0_10), dk_sk3_z2_10)), add(dk_sk1_z2_3, sub(sub(add(add(mul(dk_sk1_xs_4, dk_sk1_ys_6), mul(dk_sk1_xs_5, dk_sk1_ys_5)), mul(dk_sk1_xs_6, dk_sk1_ys_4)), dk_sk1_z0_10), dk_sk1_z2_10))), add(dk_sk2_z2_3, sub(sub(add(add(mul(dk_sk2_xs_4, dk_sk2_ys_6), mul(dk_sk2_xs_5, dk_sk2_ys_5)), mul(dk_sk2_xs_6, dk_sk2_ys_4)), dk_sk2_z0_10), dk_sk2_z2_10)))));
    m31 conv_32 = cuda_evaluator.add_intermediate(add(dk_sk2_z0_4, sub(sub(add(dk_sk3_z2_4, sub(sub(add(mul(dk_sk3_xs_5, dk_sk3_ys_6), mul(dk_sk3_xs_6, dk_sk3_ys_5)), dk_sk3_z0_11), dk_sk3_z2_11)), add(dk_sk1_z2_4, sub(sub(add(mul(dk_sk1_xs_5, dk_sk1_ys_6), mul(dk_sk1_xs_6, dk_sk1_ys_5)), dk_sk1_z0_11), dk_sk1_z2_11))), add(dk_sk2_z2_4, sub(sub(add(mul(dk_sk2_xs_5, dk_sk2_ys_6), mul(dk_sk2_xs_6, dk_sk2_ys_5)), dk_sk2_z0_11), dk_sk2_z2_11)))));
    m31 conv_33 = cuda_evaluator.add_intermediate(add(dk_sk2_z0_5, sub(sub(add(dk_sk3_z2_5, sub(sub(mul(dk_sk3_xs_6, dk_sk3_ys_6), dk_sk3_z0_12), dk_sk3_z2_12)), add(dk_sk1_z2_5, sub(sub(mul(dk_sk1_xs_6, dk_sk1_ys_6), dk_sk1_z0_12), dk_sk1_z2_12))), add(dk_sk2_z2_5, sub(sub(mul(dk_sk2_xs_6, dk_sk2_ys_6), dk_sk2_z0_12), dk_sk2_z2_12)))));
    m31 conv_34 = cuda_evaluator.add_intermediate(add(dk_sk2_z0_6, sub(sub(dk_sk3_z2_6, dk_sk1_z2_6), dk_sk2_z2_6)));
    m31 conv_35 = cuda_evaluator.add_intermediate(add(add(dk_sk2_z0_7, sub(sub(mul(dk_sk2_xs_0, dk_sk2_ys_0), dk_sk2_z0_0), dk_sk2_z2_0)), sub(sub(dk_sk3_z2_7, dk_sk1_z2_7), dk_sk2_z2_7)));
    m31 conv_36 = cuda_evaluator.add_intermediate(add(add(dk_sk2_z0_8, sub(sub(add(mul(dk_sk2_xs_0, dk_sk2_ys_1), mul(dk_sk2_xs_1, dk_sk2_ys_0)), dk_sk2_z0_1), dk_sk2_z2_1)), sub(sub(dk_sk3_z2_8, dk_sk1_z2_8), dk_sk2_z2_8)));
    m31 conv_37 = cuda_evaluator.add_intermediate(add(add(dk_sk2_z0_9, sub(sub(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_2), mul(dk_sk2_xs_1, dk_sk2_ys_1)), mul(dk_sk2_xs_2, dk_sk2_ys_0)), dk_sk2_z0_2), dk_sk2_z2_2)), sub(sub(dk_sk3_z2_9, dk_sk1_z2_9), dk_sk2_z2_9)));
    m31 conv_38 = cuda_evaluator.add_intermediate(add(add(dk_sk2_z0_10, sub(sub(add(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_3), mul(dk_sk2_xs_1, dk_sk2_ys_2)), mul(dk_sk2_xs_2, dk_sk2_ys_1)), mul(dk_sk2_xs_3, dk_sk2_ys_0)), dk_sk2_z0_3), dk_sk2_z2_3)), sub(sub(dk_sk3_z2_10, dk_sk1_z2_10), dk_sk2_z2_10)));
    m31 conv_39 = cuda_evaluator.add_intermediate(add(add(dk_sk2_z0_11, sub(sub(add(add(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_4), mul(dk_sk2_xs_1, dk_sk2_ys_3)), mul(dk_sk2_xs_2, dk_sk2_ys_2)), mul(dk_sk2_xs_3, dk_sk2_ys_1)), mul(dk_sk2_xs_4, dk_sk2_ys_0)), dk_sk2_z0_4), dk_sk2_z2_4)), sub(sub(dk_sk3_z2_11, dk_sk1_z2_11), dk_sk2_z2_11)));
    m31 conv_40 = cuda_evaluator.add_intermediate(add(add(dk_sk2_z0_12, sub(sub(add(add(add(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_5), mul(dk_sk2_xs_1, dk_sk2_ys_4)), mul(dk_sk2_xs_2, dk_sk2_ys_3)), mul(dk_sk2_xs_3, dk_sk2_ys_2)), mul(dk_sk2_xs_4, dk_sk2_ys_1)), mul(dk_sk2_xs_5, dk_sk2_ys_0)), dk_sk2_z0_5), dk_sk2_z2_5)), sub(sub(dk_sk3_z2_12, dk_sk1_z2_12), dk_sk2_z2_12)));
    m31 conv_41 = cuda_evaluator.add_intermediate(sub(sub(add(add(add(add(add(add(mul(dk_sk2_xs_0, dk_sk2_ys_6), mul(dk_sk2_xs_1, dk_sk2_ys_5)), mul(dk_sk2_xs_2, dk_sk2_ys_4)), mul(dk_sk2_xs_3, dk_sk2_ys_3)), mul(dk_sk2_xs_4, dk_sk2_ys_2)), mul(dk_sk2_xs_5, dk_sk2_ys_1)), mul(dk_sk2_xs_6, dk_sk2_ys_0)), dk_sk2_z0_6), dk_sk2_z2_6));
    m31 conv_42 = cuda_evaluator.add_intermediate(add(dk_sk2_z2_0, sub(sub(add(add(add(add(add(mul(dk_sk2_xs_1, dk_sk2_ys_6), mul(dk_sk2_xs_2, dk_sk2_ys_5)), mul(dk_sk2_xs_3, dk_sk2_ys_4)), mul(dk_sk2_xs_4, dk_sk2_ys_3)), mul(dk_sk2_xs_5, dk_sk2_ys_2)), mul(dk_sk2_xs_6, dk_sk2_ys_1)), dk_sk2_z0_7), dk_sk2_z2_7)));
    m31 conv_43 = cuda_evaluator.add_intermediate(add(dk_sk2_z2_1, sub(sub(add(add(add(add(mul(dk_sk2_xs_2, dk_sk2_ys_6), mul(dk_sk2_xs_3, dk_sk2_ys_5)), mul(dk_sk2_xs_4, dk_sk2_ys_4)), mul(dk_sk2_xs_5, dk_sk2_ys_3)), mul(dk_sk2_xs_6, dk_sk2_ys_2)), dk_sk2_z0_8), dk_sk2_z2_8)));
    m31 conv_44 = cuda_evaluator.add_intermediate(add(dk_sk2_z2_2, sub(sub(add(add(add(mul(dk_sk2_xs_3, dk_sk2_ys_6), mul(dk_sk2_xs_4, dk_sk2_ys_5)), mul(dk_sk2_xs_5, dk_sk2_ys_4)), mul(dk_sk2_xs_6, dk_sk2_ys_3)), dk_sk2_z0_9), dk_sk2_z2_9)));
    m31 conv_45 = cuda_evaluator.add_intermediate(add(dk_sk2_z2_3, sub(sub(add(add(mul(dk_sk2_xs_4, dk_sk2_ys_6), mul(dk_sk2_xs_5, dk_sk2_ys_5)), mul(dk_sk2_xs_6, dk_sk2_ys_4)), dk_sk2_z0_10), dk_sk2_z2_10)));
    m31 conv_46 = cuda_evaluator.add_intermediate(add(dk_sk2_z2_4, sub(sub(add(mul(dk_sk2_xs_5, dk_sk2_ys_6), mul(dk_sk2_xs_6, dk_sk2_ys_5)), dk_sk2_z0_11), dk_sk2_z2_11)));
    m31 conv_47 = cuda_evaluator.add_intermediate(add(dk_sk2_z2_5, sub(sub(mul(dk_sk2_xs_6, dk_sk2_ys_6), dk_sk2_z0_12), dk_sk2_z2_12)));
    m31 conv_48 = cuda_evaluator.add_intermediate(dk_sk2_z2_6);
    m31 conv_49 = cuda_evaluator.add_intermediate(dk_sk2_z2_7);
    m31 conv_50 = cuda_evaluator.add_intermediate(dk_sk2_z2_8);
    m31 conv_51 = cuda_evaluator.add_intermediate(dk_sk2_z2_9);
    m31 conv_52 = cuda_evaluator.add_intermediate(dk_sk2_z2_10);
    m31 conv_53 = cuda_evaluator.add_intermediate(dk_sk2_z2_11);
    m31 conv_54 = cuda_evaluator.add_intermediate(dk_sk2_z2_12);

    // conv_mod computation (28 intermediates)
    m31 cm_0 = cuda_evaluator.add_intermediate(add(sub(mul(m31(32), conv_0), mul(m31(4), conv_21)), mul(m31(8), conv_49)));
    m31 cm_1 = cuda_evaluator.add_intermediate(add(sub(add(conv_0, mul(m31(32), conv_1)), mul(m31(4), conv_22)), mul(m31(8), conv_50)));
    m31 cm_2 = cuda_evaluator.add_intermediate(add(sub(add(conv_1, mul(m31(32), conv_2)), mul(m31(4), conv_23)), mul(m31(8), conv_51)));
    m31 cm_3 = cuda_evaluator.add_intermediate(add(sub(add(conv_2, mul(m31(32), conv_3)), mul(m31(4), conv_24)), mul(m31(8), conv_52)));
    m31 cm_4 = cuda_evaluator.add_intermediate(add(sub(add(conv_3, mul(m31(32), conv_4)), mul(m31(4), conv_25)), mul(m31(8), conv_53)));
    m31 cm_5 = cuda_evaluator.add_intermediate(add(sub(add(conv_4, mul(m31(32), conv_5)), mul(m31(4), conv_26)), mul(m31(8), conv_54)));
    m31 cm_6 = cuda_evaluator.add_intermediate(sub(add(conv_5, mul(m31(32), conv_6)), mul(m31(4), conv_27)));
    m31 cm_7 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_0), conv_6), mul(m31(32), conv_7)), mul(m31(4), conv_28)));
    m31 cm_8 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_1), conv_7), mul(m31(32), conv_8)), mul(m31(4), conv_29)));
    m31 cm_9 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_2), conv_8), mul(m31(32), conv_9)), mul(m31(4), conv_30)));
    m31 cm_10 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_3), conv_9), mul(m31(32), conv_10)), mul(m31(4), conv_31)));
    m31 cm_11 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_4), conv_10), mul(m31(32), conv_11)), mul(m31(4), conv_32)));
    m31 cm_12 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_5), conv_11), mul(m31(32), conv_12)), mul(m31(4), conv_33)));
    m31 cm_13 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_6), conv_12), mul(m31(32), conv_13)), mul(m31(4), conv_34)));
    m31 cm_14 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_7), conv_13), mul(m31(32), conv_14)), mul(m31(4), conv_35)));
    m31 cm_15 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_8), conv_14), mul(m31(32), conv_15)), mul(m31(4), conv_36)));
    m31 cm_16 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_9), conv_15), mul(m31(32), conv_16)), mul(m31(4), conv_37)));
    m31 cm_17 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_10), conv_16), mul(m31(32), conv_17)), mul(m31(4), conv_38)));
    m31 cm_18 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_11), conv_17), mul(m31(32), conv_18)), mul(m31(4), conv_39)));
    m31 cm_19 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_12), conv_18), mul(m31(32), conv_19)), mul(m31(4), conv_40)));
    m31 cm_20 = cuda_evaluator.add_intermediate(sub(add(add(mul(m31(2), conv_13), conv_19), mul(m31(32), conv_20)), mul(m31(4), conv_41)));
    m31 cm_21 = cuda_evaluator.add_intermediate(add(sub(add(mul(m31(2), conv_14), conv_20), mul(m31(4), conv_42)), mul(m31(64), conv_49)));
    m31 cm_22 = cuda_evaluator.add_intermediate(add(add(sub(mul(m31(2), conv_15), mul(m31(4), conv_43)), mul(m31(2), conv_49)), mul(m31(64), conv_50)));
    m31 cm_23 = cuda_evaluator.add_intermediate(add(add(sub(mul(m31(2), conv_16), mul(m31(4), conv_44)), mul(m31(2), conv_50)), mul(m31(64), conv_51)));
    m31 cm_24 = cuda_evaluator.add_intermediate(add(add(sub(mul(m31(2), conv_17), mul(m31(4), conv_45)), mul(m31(2), conv_51)), mul(m31(64), conv_52)));
    m31 cm_25 = cuda_evaluator.add_intermediate(add(add(sub(mul(m31(2), conv_18), mul(m31(4), conv_46)), mul(m31(2), conv_52)), mul(m31(64), conv_53)));
    m31 cm_26 = cuda_evaluator.add_intermediate(add(add(sub(mul(m31(2), conv_19), mul(m31(4), conv_47)), mul(m31(2), conv_53)), mul(m31(64), conv_54)));
    m31 cm_27 = cuda_evaluator.add_intermediate(add(sub(mul(m31(2), conv_20), mul(m31(4), conv_48)), mul(m31(2), conv_54)));

    // VerifyMul252 carry constraints (28 constraints) and range check relations (28 relations)
    { m31 v[2] = { RANGE_CHECK_20_RELATION_ID, add(k_col169, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_0_col170, m31(512)), sub(cm_0, k_col169)));
    { m31 v[2] = { RANGE_CHECK_20_B_RELATION_ID, add(carry_0_col170, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_1_col171, m31(512)), add(cm_1, carry_0_col170)));
    { m31 v[2] = { RANGE_CHECK_20_C_RELATION_ID, add(carry_1_col171, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_2_col172, m31(512)), add(cm_2, carry_1_col171)));
    { m31 v[2] = { RANGE_CHECK_20_D_RELATION_ID, add(carry_2_col172, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_3_col173, m31(512)), add(cm_3, carry_2_col172)));
    { m31 v[2] = { RANGE_CHECK_20_E_RELATION_ID, add(carry_3_col173, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_4_col174, m31(512)), add(cm_4, carry_3_col173)));
    { m31 v[2] = { RANGE_CHECK_20_F_RELATION_ID, add(carry_4_col174, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_5_col175, m31(512)), add(cm_5, carry_4_col174)));
    { m31 v[2] = { RANGE_CHECK_20_G_RELATION_ID, add(carry_5_col175, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_6_col176, m31(512)), add(cm_6, carry_5_col175)));
    { m31 v[2] = { RANGE_CHECK_20_H_RELATION_ID, add(carry_6_col176, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_7_col177, m31(512)), add(cm_7, carry_6_col176)));
    { m31 v[2] = { RANGE_CHECK_20_RELATION_ID, add(carry_7_col177, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_8_col178, m31(512)), add(cm_8, carry_7_col177)));
    { m31 v[2] = { RANGE_CHECK_20_B_RELATION_ID, add(carry_8_col178, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_9_col179, m31(512)), add(cm_9, carry_8_col178)));
    { m31 v[2] = { RANGE_CHECK_20_C_RELATION_ID, add(carry_9_col179, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_10_col180, m31(512)), add(cm_10, carry_9_col179)));
    { m31 v[2] = { RANGE_CHECK_20_D_RELATION_ID, add(carry_10_col180, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_11_col181, m31(512)), add(cm_11, carry_10_col180)));
    { m31 v[2] = { RANGE_CHECK_20_E_RELATION_ID, add(carry_11_col181, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_12_col182, m31(512)), add(cm_12, carry_11_col181)));
    { m31 v[2] = { RANGE_CHECK_20_F_RELATION_ID, add(carry_12_col182, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_13_col183, m31(512)), add(cm_13, carry_12_col182)));
    { m31 v[2] = { RANGE_CHECK_20_G_RELATION_ID, add(carry_13_col183, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_14_col184, m31(512)), add(cm_14, carry_13_col183)));
    { m31 v[2] = { RANGE_CHECK_20_H_RELATION_ID, add(carry_14_col184, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_15_col185, m31(512)), add(cm_15, carry_14_col184)));
    { m31 v[2] = { RANGE_CHECK_20_RELATION_ID, add(carry_15_col185, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_16_col186, m31(512)), add(cm_16, carry_15_col185)));
    { m31 v[2] = { RANGE_CHECK_20_B_RELATION_ID, add(carry_16_col186, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_17_col187, m31(512)), add(cm_17, carry_16_col186)));
    { m31 v[2] = { RANGE_CHECK_20_C_RELATION_ID, add(carry_17_col187, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_18_col188, m31(512)), add(cm_18, carry_17_col187)));
    { m31 v[2] = { RANGE_CHECK_20_D_RELATION_ID, add(carry_18_col188, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_19_col189, m31(512)), add(cm_19, carry_18_col188)));
    { m31 v[2] = { RANGE_CHECK_20_E_RELATION_ID, add(carry_19_col189, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_20_col190, m31(512)), add(cm_20, carry_19_col189)));
    { m31 v[2] = { RANGE_CHECK_20_F_RELATION_ID, add(carry_20_col190, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_21_col191, m31(512)), add(sub(cm_21, mul(m31(136), k_col169)), carry_20_col190)));
    { m31 v[2] = { RANGE_CHECK_20_G_RELATION_ID, add(carry_21_col191, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_22_col192, m31(512)), add(cm_22, carry_21_col191)));
    { m31 v[2] = { RANGE_CHECK_20_H_RELATION_ID, add(carry_22_col192, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_23_col193, m31(512)), add(cm_23, carry_22_col192)));
    { m31 v[2] = { RANGE_CHECK_20_RELATION_ID, add(carry_23_col193, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_24_col194, m31(512)), add(cm_24, carry_23_col193)));
    { m31 v[2] = { RANGE_CHECK_20_B_RELATION_ID, add(carry_24_col194, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_25_col195, m31(512)), add(cm_25, carry_24_col194)));
    { m31 v[2] = { RANGE_CHECK_20_C_RELATION_ID, add(carry_25_col195, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(sub(mul(carry_26_col196, m31(512)), add(cm_26, carry_25_col195)));
    { m31 v[2] = { RANGE_CHECK_20_D_RELATION_ID, add(carry_26_col196, m31(524288)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    cuda_evaluator.add_constraint(add(sub(cm_27, mul(m31(256), k_col169)), carry_26_col196));

    // EvalOperands: 28 res constraints
    m31 not_jnz = cuda_evaluator.add_intermediate(sub(m31(1), pc_update_jnz_col15));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_0_col112), mul(res_mul_col12, mul_res_limb_0_col141)), mul(res_op1, op1_limb_0_col84)), mul(not_jnz, res_limb_0_col197)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_1_col113), mul(res_mul_col12, mul_res_limb_1_col142)), mul(res_op1, op1_limb_1_col85)), mul(not_jnz, res_limb_1_col198)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_2_col114), mul(res_mul_col12, mul_res_limb_2_col143)), mul(res_op1, op1_limb_2_col86)), mul(not_jnz, res_limb_2_col199)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_3_col115), mul(res_mul_col12, mul_res_limb_3_col144)), mul(res_op1, op1_limb_3_col87)), mul(not_jnz, res_limb_3_col200)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_4_col116), mul(res_mul_col12, mul_res_limb_4_col145)), mul(res_op1, op1_limb_4_col88)), mul(not_jnz, res_limb_4_col201)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_5_col117), mul(res_mul_col12, mul_res_limb_5_col146)), mul(res_op1, op1_limb_5_col89)), mul(not_jnz, res_limb_5_col202)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_6_col118), mul(res_mul_col12, mul_res_limb_6_col147)), mul(res_op1, op1_limb_6_col90)), mul(not_jnz, res_limb_6_col203)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_7_col119), mul(res_mul_col12, mul_res_limb_7_col148)), mul(res_op1, op1_limb_7_col91)), mul(not_jnz, res_limb_7_col204)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_8_col120), mul(res_mul_col12, mul_res_limb_8_col149)), mul(res_op1, op1_limb_8_col92)), mul(not_jnz, res_limb_8_col205)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_9_col121), mul(res_mul_col12, mul_res_limb_9_col150)), mul(res_op1, op1_limb_9_col93)), mul(not_jnz, res_limb_9_col206)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_10_col122), mul(res_mul_col12, mul_res_limb_10_col151)), mul(res_op1, op1_limb_10_col94)), mul(not_jnz, res_limb_10_col207)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_11_col123), mul(res_mul_col12, mul_res_limb_11_col152)), mul(res_op1, op1_limb_11_col95)), mul(not_jnz, res_limb_11_col208)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_12_col124), mul(res_mul_col12, mul_res_limb_12_col153)), mul(res_op1, op1_limb_12_col96)), mul(not_jnz, res_limb_12_col209)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_13_col125), mul(res_mul_col12, mul_res_limb_13_col154)), mul(res_op1, op1_limb_13_col97)), mul(not_jnz, res_limb_13_col210)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_14_col126), mul(res_mul_col12, mul_res_limb_14_col155)), mul(res_op1, op1_limb_14_col98)), mul(not_jnz, res_limb_14_col211)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_15_col127), mul(res_mul_col12, mul_res_limb_15_col156)), mul(res_op1, op1_limb_15_col99)), mul(not_jnz, res_limb_15_col212)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_16_col128), mul(res_mul_col12, mul_res_limb_16_col157)), mul(res_op1, op1_limb_16_col100)), mul(not_jnz, res_limb_16_col213)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_17_col129), mul(res_mul_col12, mul_res_limb_17_col158)), mul(res_op1, op1_limb_17_col101)), mul(not_jnz, res_limb_17_col214)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_18_col130), mul(res_mul_col12, mul_res_limb_18_col159)), mul(res_op1, op1_limb_18_col102)), mul(not_jnz, res_limb_18_col215)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_19_col131), mul(res_mul_col12, mul_res_limb_19_col160)), mul(res_op1, op1_limb_19_col103)), mul(not_jnz, res_limb_19_col216)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_20_col132), mul(res_mul_col12, mul_res_limb_20_col161)), mul(res_op1, op1_limb_20_col104)), mul(not_jnz, res_limb_20_col217)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_21_col133), mul(res_mul_col12, mul_res_limb_21_col162)), mul(res_op1, op1_limb_21_col105)), mul(not_jnz, res_limb_21_col218)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_22_col134), mul(res_mul_col12, mul_res_limb_22_col163)), mul(res_op1, op1_limb_22_col106)), mul(not_jnz, res_limb_22_col219)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_23_col135), mul(res_mul_col12, mul_res_limb_23_col164)), mul(res_op1, op1_limb_23_col107)), mul(not_jnz, res_limb_23_col220)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_24_col136), mul(res_mul_col12, mul_res_limb_24_col165)), mul(res_op1, op1_limb_24_col108)), mul(not_jnz, res_limb_24_col221)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_25_col137), mul(res_mul_col12, mul_res_limb_25_col166)), mul(res_op1, op1_limb_25_col109)), mul(not_jnz, res_limb_25_col222)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_26_col138), mul(res_mul_col12, mul_res_limb_26_col167)), mul(res_op1, op1_limb_26_col110)), mul(not_jnz, res_limb_26_col223)));
    cuda_evaluator.add_constraint(sub(add(add(mul(res_add_col11, add_res_limb_27_col139), mul(res_mul_col12, mul_res_limb_27_col168)), mul(res_op1, op1_limb_27_col111)), mul(not_jnz, res_limb_27_col224)));

    // ===== HandleOpcodes (42 constraints) =====

    // 28 assert_eq constraints
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_0_col197, dst_limb_0_col23)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_1_col198, dst_limb_1_col24)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_2_col199, dst_limb_2_col25)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_3_col200, dst_limb_3_col26)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_4_col201, dst_limb_4_col27)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_5_col202, dst_limb_5_col28)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_6_col203, dst_limb_6_col29)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_7_col204, dst_limb_7_col30)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_8_col205, dst_limb_8_col31)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_9_col206, dst_limb_9_col32)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_10_col207, dst_limb_10_col33)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_11_col208, dst_limb_11_col34)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_12_col209, dst_limb_12_col35)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_13_col210, dst_limb_13_col36)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_14_col211, dst_limb_14_col37)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_15_col212, dst_limb_15_col38)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_16_col213, dst_limb_16_col39)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_17_col214, dst_limb_17_col40)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_18_col215, dst_limb_18_col41)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_19_col216, dst_limb_19_col42)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_20_col217, dst_limb_20_col43)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_21_col218, dst_limb_21_col44)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_22_col219, dst_limb_22_col45)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_23_col220, dst_limb_23_col46)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_24_col221, dst_limb_24_col47)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_25_col222, dst_limb_25_col48)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_26_col223, dst_limb_26_col49)));
    cuda_evaluator.add_constraint(mul(opcode_assert_eq_col20, sub(res_limb_27_col224, dst_limb_27_col50)));

    // ret opcode constraints (3)
    cuda_evaluator.add_constraint(mul(opcode_ret_col19, add(offset0, m31(2))));
    cuda_evaluator.add_constraint(mul(opcode_ret_col19, add(offset2, m31(1))));
    cuda_evaluator.add_constraint(mul(opcode_ret_col19, sub(sub(sub(sub(m31(4), pc_update_jump_col13), dst_base_fp_col6), op1_base_fp_col9), res_op1)));

    // call opcode constraints (3)
    cuda_evaluator.add_constraint(mul(opcode_call_col18, offset0));
    cuda_evaluator.add_constraint(mul(opcode_call_col18, sub(m31(1), offset1)));
    cuda_evaluator.add_constraint(mul(opcode_call_col18, add(op0_base_fp_col7, dst_base_fp_col6)));

    // CondFelt252AsAddr for dst (cond=opcode_call) (3 constraints)
    cuda_evaluator.add_constraint(mul(opcode_call_col18, add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(dst_limb_4_col27, dst_limb_5_col28), dst_limb_6_col29), dst_limb_7_col30), dst_limb_8_col31), dst_limb_9_col32), dst_limb_10_col33), dst_limb_11_col34), dst_limb_12_col35), dst_limb_13_col36), dst_limb_14_col37), dst_limb_15_col38), dst_limb_16_col39), dst_limb_17_col40), dst_limb_18_col41), dst_limb_19_col42), dst_limb_20_col43), dst_limb_21_col44), dst_limb_22_col45), dst_limb_23_col46), dst_limb_24_col47), dst_limb_25_col48), dst_limb_26_col49), dst_limb_27_col50)));
    cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col225, sub(partial_limb_msb_col225, m31(1))), opcode_call_col18));
    m31 crc2_dst_ho = cuda_evaluator.add_intermediate(sub(dst_limb_3_col26, mul(partial_limb_msb_col225, m31(2))));
    cuda_evaluator.add_constraint(mul(mul(crc2_dst_ho, sub(crc2_dst_ho, m31(1))), opcode_call_col18));
    m31 dst_as_addr = add(add(add(dst_limb_0_col23, mul(dst_limb_1_col24, m31(512))), mul(dst_limb_2_col25, m31(262144))), mul(dst_limb_3_col26, m31(134217728)));
    cuda_evaluator.add_constraint(mul(opcode_call_col18, sub(dst_as_addr, input_fp_col2)));

    // CondFelt252AsAddr for op0 (cond=opcode_call, HandleOpcodes) (3 constraints)
    cuda_evaluator.add_constraint(mul(opcode_call_col18, add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(op0_limb_4_col57, op0_limb_5_col58), op0_limb_6_col59), op0_limb_7_col60), op0_limb_8_col61), op0_limb_9_col62), op0_limb_10_col63), op0_limb_11_col64), op0_limb_12_col65), op0_limb_13_col66), op0_limb_14_col67), op0_limb_15_col68), op0_limb_16_col69), op0_limb_17_col70), op0_limb_18_col71), op0_limb_19_col72), op0_limb_20_col73), op0_limb_21_col74), op0_limb_22_col75), op0_limb_23_col76), op0_limb_24_col77), op0_limb_25_col78), op0_limb_26_col79), op0_limb_27_col80)));
    cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col226, sub(partial_limb_msb_col226, m31(1))), opcode_call_col18));
    m31 crc2_op0_ho = cuda_evaluator.add_intermediate(sub(op0_limb_3_col56, mul(partial_limb_msb_col226, m31(2))));
    cuda_evaluator.add_constraint(mul(mul(crc2_op0_ho, sub(crc2_op0_ho, m31(1))), opcode_call_col18));
    m31 op0_as_addr_ho = add(add(add(op0_limb_0_col53, mul(op0_limb_1_col54, m31(512))), mul(op0_limb_2_col55, m31(262144))), mul(op0_limb_3_col56, m31(134217728)));
    cuda_evaluator.add_constraint(mul(opcode_call_col18, sub(op0_as_addr_ho, add(input_pc_col0, instruction_size))));

    // ===== UpdateRegisters (28 constraints, 2 relations) =====

    // CondFelt252AsAddr for dst (cond=opcode_ret) (3 constraints)
    cuda_evaluator.add_constraint(mul(opcode_ret_col19, add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(dst_limb_4_col27, dst_limb_5_col28), dst_limb_6_col29), dst_limb_7_col30), dst_limb_8_col31), dst_limb_9_col32), dst_limb_10_col33), dst_limb_11_col34), dst_limb_12_col35), dst_limb_13_col36), dst_limb_14_col37), dst_limb_15_col38), dst_limb_16_col39), dst_limb_17_col40), dst_limb_18_col41), dst_limb_19_col42), dst_limb_20_col43), dst_limb_21_col44), dst_limb_22_col45), dst_limb_23_col46), dst_limb_24_col47), dst_limb_25_col48), dst_limb_26_col49), dst_limb_27_col50)));
    cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col227, sub(partial_limb_msb_col227, m31(1))), opcode_ret_col19));
    m31 crc2_dst_ur = cuda_evaluator.add_intermediate(sub(dst_limb_3_col26, mul(partial_limb_msb_col227, m31(2))));
    cuda_evaluator.add_constraint(mul(mul(crc2_dst_ur, sub(crc2_dst_ur, m31(1))), opcode_ret_col19));
    m31 dst_as_addr_ur = add(add(add(dst_limb_0_col23, mul(dst_limb_1_col24, m31(512))), mul(dst_limb_2_col25, m31(262144))), mul(dst_limb_3_col26, m31(134217728)));

    // CondFelt252AsRelImm for res (cond = pc_update_jump_rel + ap_update_add) (9 constraints)
    m31 res_cfri_cond = add(pc_update_jump_rel_col14, ap_update_add_col16);
    cuda_evaluator.add_constraint(mul(mul(msb_col228, sub(msb_col228, m31(1))), res_cfri_cond));
    cuda_evaluator.add_constraint(mul(mul(mid_limbs_set_col229, sub(mid_limbs_set_col229, m31(1))), res_cfri_cond));
    cuda_evaluator.add_constraint(mul(mul(mid_limbs_set_col229, sub(msb_col228, m31(1))), res_cfri_cond));
    m31 dss_res_limb3_high = mul(mid_limbs_set_col229, m31(508));
    m31 dss_res_limbs4to20 = mul(mid_limbs_set_col229, m31(511));
    m31 dss_res_limb21 = sub(mul(msb_col228, m31(136)), mid_limbs_set_col229);
    m31 dss_res_limb27 = mul(msb_col228, m31(256));
    m31 crc2_res_cfri_rem = cuda_evaluator.add_intermediate(sub(res_limb_3_col200, dss_res_limb3_high));
    cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col230, sub(partial_limb_msb_col230, m31(1))), res_cfri_cond));
    m31 crc2_res_cfri_val = cuda_evaluator.add_intermediate(sub(crc2_res_cfri_rem, mul(partial_limb_msb_col230, m31(2))));
    cuda_evaluator.add_constraint(mul(mul(crc2_res_cfri_val, sub(crc2_res_cfri_val, m31(1))), res_cfri_cond));
    cuda_evaluator.add_constraint(mul(res_cfri_cond, sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(res_limb_4_col201, res_limb_5_col202), res_limb_6_col203), res_limb_7_col204), res_limb_8_col205), res_limb_9_col206), res_limb_10_col207), res_limb_11_col208), res_limb_12_col209), res_limb_13_col210), res_limb_14_col211), res_limb_15_col212), res_limb_16_col213), res_limb_17_col214), res_limb_18_col215), res_limb_19_col216), res_limb_20_col217), mul(dss_res_limbs4to20, m31(17)))));
    cuda_evaluator.add_constraint(mul(res_cfri_cond, sub(res_limb_21_col218, dss_res_limb21)));
    cuda_evaluator.add_constraint(mul(res_cfri_cond, add(add(add(add(res_limb_22_col219, res_limb_23_col220), res_limb_24_col221), res_limb_25_col222), res_limb_26_col223)));
    cuda_evaluator.add_constraint(mul(res_cfri_cond, sub(res_limb_27_col224, dss_res_limb27)));
    m31 res_as_rel_imm = sub(sub(add(add(add(res_limb_0_col197, mul(res_limb_1_col198, m31(512))), mul(res_limb_2_col199, m31(262144))), mul(crc2_res_cfri_rem, m31(134217728))), msb_col228), mul(m31(536870912), mid_limbs_set_col229));

    // dst_not_p constraint (1 constraint, 3 intermediates)
    m31 diff_p_0 = cuda_evaluator.add_intermediate(sub(dst_limb_0_col23, m31(1)));
    m31 diff_p_21 = cuda_evaluator.add_intermediate(sub(dst_limb_21_col44, m31(136)));
    m31 diff_p_27 = cuda_evaluator.add_intermediate(sub(dst_limb_27_col50, m31(256)));
    cuda_evaluator.add_constraint(sub(mul(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(diff_p_0, diff_p_0), dst_limb_1_col24), dst_limb_2_col25), dst_limb_3_col26), dst_limb_4_col27), dst_limb_5_col28), dst_limb_6_col29), dst_limb_7_col30), dst_limb_8_col31), dst_limb_9_col32), dst_limb_10_col33), dst_limb_11_col34), dst_limb_12_col35), dst_limb_13_col36), dst_limb_14_col37), dst_limb_15_col38), dst_limb_16_col39), dst_limb_17_col40), dst_limb_18_col41), dst_limb_19_col42), dst_limb_20_col43), mul(diff_p_21, diff_p_21)), dst_limb_22_col45), dst_limb_23_col46), dst_limb_24_col47), dst_limb_25_col48), dst_limb_26_col49), mul(diff_p_27, diff_p_27)), dst_sum_squares_inv_col231), m31(1)));

    // op1_as_rel_imm_cond constraint (1 constraint, 1 intermediate)
    m31 dst_sum = cuda_evaluator.add_intermediate(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(dst_limb_0_col23, dst_limb_1_col24), dst_limb_2_col25), dst_limb_3_col26), dst_limb_4_col27), dst_limb_5_col28), dst_limb_6_col29), dst_limb_7_col30), dst_limb_8_col31), dst_limb_9_col32), dst_limb_10_col33), dst_limb_11_col34), dst_limb_12_col35), dst_limb_13_col36), dst_limb_14_col37), dst_limb_15_col38), dst_limb_16_col39), dst_limb_17_col40), dst_limb_18_col41), dst_limb_19_col42), dst_limb_20_col43), dst_limb_21_col44), dst_limb_22_col45), dst_limb_23_col46), dst_limb_24_col47), dst_limb_25_col48), dst_limb_26_col49), dst_limb_27_col50));
    cuda_evaluator.add_constraint(sub(op1_as_rel_imm_cond_col233, mul(pc_update_jnz_col15, dst_sum)));

    // CondFelt252AsRelImm for op1 (cond=op1_as_rel_imm_cond) (9 constraints)
    cuda_evaluator.add_constraint(mul(mul(msb_col234, sub(msb_col234, m31(1))), op1_as_rel_imm_cond_col233));
    cuda_evaluator.add_constraint(mul(mul(mid_limbs_set_col235, sub(mid_limbs_set_col235, m31(1))), op1_as_rel_imm_cond_col233));
    cuda_evaluator.add_constraint(mul(mul(mid_limbs_set_col235, sub(msb_col234, m31(1))), op1_as_rel_imm_cond_col233));
    m31 dss_op1_limb3_high = mul(mid_limbs_set_col235, m31(508));
    m31 dss_op1_limbs4to20 = mul(mid_limbs_set_col235, m31(511));
    m31 dss_op1_limb21 = sub(mul(msb_col234, m31(136)), mid_limbs_set_col235);
    m31 dss_op1_limb27 = mul(msb_col234, m31(256));
    m31 crc2_op1_cfri_rem = cuda_evaluator.add_intermediate(sub(op1_limb_3_col87, dss_op1_limb3_high));
    cuda_evaluator.add_constraint(mul(mul(partial_limb_msb_col236, sub(partial_limb_msb_col236, m31(1))), op1_as_rel_imm_cond_col233));
    m31 crc2_op1_cfri_val = cuda_evaluator.add_intermediate(sub(crc2_op1_cfri_rem, mul(partial_limb_msb_col236, m31(2))));
    cuda_evaluator.add_constraint(mul(mul(crc2_op1_cfri_val, sub(crc2_op1_cfri_val, m31(1))), op1_as_rel_imm_cond_col233));
    cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col233, sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(op1_limb_4_col88, op1_limb_5_col89), op1_limb_6_col90), op1_limb_7_col91), op1_limb_8_col92), op1_limb_9_col93), op1_limb_10_col94), op1_limb_11_col95), op1_limb_12_col96), op1_limb_13_col97), op1_limb_14_col98), op1_limb_15_col99), op1_limb_16_col100), op1_limb_17_col101), op1_limb_18_col102), op1_limb_19_col103), op1_limb_20_col104), mul(dss_op1_limbs4to20, m31(17)))));
    cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col233, sub(op1_limb_21_col105, dss_op1_limb21)));
    cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col233, add(add(add(add(op1_limb_22_col106, op1_limb_23_col107), op1_limb_24_col108), op1_limb_25_col109), op1_limb_26_col110)));
    cuda_evaluator.add_constraint(mul(op1_as_rel_imm_cond_col233, sub(op1_limb_27_col111, dss_op1_limb27)));
    m31 op1_as_rel_imm = sub(sub(add(add(add(op1_limb_0_col84, mul(op1_limb_1_col85, m31(512))), mul(op1_limb_2_col86, m31(262144))), mul(crc2_op1_cfri_rem, m31(134217728))), msb_col234), mul(m31(536870912), mid_limbs_set_col235));

    // Constraint1 for conditional jump (1 constraint)
    cuda_evaluator.add_constraint(mul(sub(next_pc_jnz_col237, add(input_pc_col0, op1_as_rel_imm)), dst_sum));

    // Constraint2 for conditional jump (1 constraint)
    cuda_evaluator.add_constraint(mul(sub(next_pc_jnz_col237, add(input_pc_col0, instruction_size)), sub(mul(dst_sum, dst_sum_inv_col232), m31(1))));

    // next_pc constraint (1 constraint)
    cuda_evaluator.add_constraint(sub(next_pc_col238, add(add(add(mul(pc_update_regular, add(input_pc_col0, instruction_size)), mul(pc_update_jump_col13, res_as_rel_imm)), mul(pc_update_jump_rel_col14, add(input_pc_col0, res_as_rel_imm))), mul(pc_update_jnz_col15, next_pc_jnz_col237))));

    // next_ap constraint (1 constraint)
    cuda_evaluator.add_constraint(sub(next_ap_col239, add(add(add(input_ap_col1, mul(ap_update_add_col16, res_as_rel_imm)), ap_update_add_1_col17), mul(opcode_call_col18, m31(2)))));

    // RangeCheck29 for next_ap (2 relations)
    { m31 v[2] = { RANGE_CHECK_18_RELATION_ID, mul(sub(next_ap_col239, range_check_29_bot11bits_col240), m31(1048576)) }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[2] = { RANGE_CHECK_11_RELATION_ID, range_check_29_bot11bits_col240 }; cuda_evaluator.add_to_relation<2>(generic_opcode_eval->common_lookup_elements, qm31{{m31(1), m31(0)}, {m31(0), m31(0)}}, v); }

    // next_fp constraint (1 constraint)
    cuda_evaluator.add_constraint(sub(next_fp_col241, add(add(mul(fp_update_regular, input_fp_col2), mul(opcode_ret_col19, dst_as_addr_ur)), mul(opcode_call_col18, add(input_ap_col1, m31(2))))));

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // opcodes_in and opcodes_out relations (2 relations)
    { m31 v[4] = { OPCODES_RELATION_ID, input_pc_col0, input_ap_col1, input_fp_col2 }; cuda_evaluator.add_to_relation<4>(generic_opcode_eval->common_lookup_elements, qm31{{enabler, m31(0)}, {m31(0), m31(0)}}, v); }
    { m31 v[4] = { OPCODES_RELATION_ID, next_pc_col238, next_ap_col239, next_fp_col241 }; cuda_evaluator.add_to_relation<4>(generic_opcode_eval->common_lookup_elements, qm31{{neg(enabler), m31(0)}, {m31(0), m31(0)}}, v); }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

void evaluate_generic_opcode(
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
    GenericOpcode_Eval *generic_opcode_eval = (GenericOpcode_Eval *) eval;
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    GenericOpcode_Eval *device_generic_opcode_eval = cuda_malloc<GenericOpcode_Eval>(1);
    cuda_mem_copy_host_to_device<GenericOpcode_Eval>(generic_opcode_eval, device_generic_opcode_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constrain_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_generic_opcode");

    int block_dim = eval_domain_size < GENERIC_OPCODE_THREAD_COUNT_MAX ? eval_domain_size : GENERIC_OPCODE_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_generic_opcode_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_generic_opcode_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constrain_index_array
        );
    } else {
        evaluate_generic_opcode_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_generic_opcode_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constrain_index_array
        );
    }
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = batching[logup_counts - 1];

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constrain_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    } else {
        generic_constraint_post_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constrain_index_array,
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
    global_timer.end("evaluate_generic_opcode");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_generic_opcode_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constrain_index_array);
}

