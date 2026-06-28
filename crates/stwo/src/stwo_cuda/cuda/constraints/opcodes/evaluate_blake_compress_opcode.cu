#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_blake_compress_opcode.cuh"
#include "evaluate_decode_blake_opcode.cuh"
#include "evaluate_verify_blake_word.cuh"
#include "evaluate_create_blake_input.cuh"
#include "evaluate_create_blake_output.cuh"
#include "evaluate_common.cuh"

#define BLAKE_ROUND_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_blake_compress_opcode_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    BlakeCompressOpcode_Eval *blake_compress_opcode_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;


    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_10 = m31(10);
    const m31 M31_15470 = m31(15470);
    const m31 M31_2 = m31(2);
    const m31 M31_23520 = m31(23520);
    const m31 M31_26764 = m31(26764);
    const m31 M31_27145 = m31(27145);
    const m31 M31_3 = m31(3);
    const m31 M31_39685 = m31(39685);
    const m31 M31_4 = m31(4);
    const m31 M31_42319 = m31(42319);
    const m31 M31_44677 = m31(44677);
    const m31 M31_47975 = m31(47975);
    const m31 M31_5 = m31(5);
    const m31 M31_52505 = m31(52505);
    const m31 M31_58983 = m31(58983);
    const m31 M31_6 = m31(6);
    const m31 M31_62322 = m31(62322);
    const m31 M31_62778 = m31(62778);
    const m31 M31_7 = m31(7);

    // Evaluator for preprocessed trace (trace0) - Seq column
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

    // Read seq from preprocessed trace (Seq column)
    m31 seq = cuda_evaluator0.next_trace_mask();

    EvaluatorT cuda_evaluator1(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        {{0,0},{0,0}},
        0,
        cumsum_shift,  // Fixed: use actual cumsum_shift, not zero
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );
    // trace columns
    m31 input_pc_col0            = cuda_evaluator1.next_trace_mask();
    m31 input_ap_col1            = cuda_evaluator1.next_trace_mask();
    m31 input_fp_col2            = cuda_evaluator1.next_trace_mask();

    m31 offset0_col3             = cuda_evaluator1.next_trace_mask();
    m31 offset1_col4             = cuda_evaluator1.next_trace_mask();
    m31 offset2_col5             = cuda_evaluator1.next_trace_mask();
    m31 dst_base_fp_col6         = cuda_evaluator1.next_trace_mask();
    m31 op0_base_fp_col7         = cuda_evaluator1.next_trace_mask();
    m31 op1_base_fp_col8         = cuda_evaluator1.next_trace_mask();
    m31 ap_update_add_1_col9     = cuda_evaluator1.next_trace_mask();
    m31 opcode_extension_col10   = cuda_evaluator1.next_trace_mask();
    m31 mem0_base_col11          = cuda_evaluator1.next_trace_mask();
    m31 op0_id_col12             = cuda_evaluator1.next_trace_mask();
    m31 op0_limb_0_col13         = cuda_evaluator1.next_trace_mask();
    m31 op0_limb_1_col14         = cuda_evaluator1.next_trace_mask();
    m31 op0_limb_2_col15         = cuda_evaluator1.next_trace_mask();
    m31 op0_limb_3_col16         = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col17   = cuda_evaluator1.next_trace_mask();
    m31 mem1_base_col18          = cuda_evaluator1.next_trace_mask();
    m31 op1_id_col19             = cuda_evaluator1.next_trace_mask();
    m31 op1_limb_0_col20         = cuda_evaluator1.next_trace_mask();
    m31 op1_limb_1_col21         = cuda_evaluator1.next_trace_mask();
    m31 op1_limb_2_col22         = cuda_evaluator1.next_trace_mask();
    m31 op1_limb_3_col23         = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col24   = cuda_evaluator1.next_trace_mask();
    m31 ap_id_col25              = cuda_evaluator1.next_trace_mask();
    m31 ap_limb_0_col26          = cuda_evaluator1.next_trace_mask();
    m31 ap_limb_1_col27          = cuda_evaluator1.next_trace_mask();
    m31 ap_limb_2_col28          = cuda_evaluator1.next_trace_mask();
    m31 ap_limb_3_col29          = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col30   = cuda_evaluator1.next_trace_mask();
    m31 mem_dst_base_col31       = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col32        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col33       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col34      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col35    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col36     = cuda_evaluator1.next_trace_mask();
    m31 dst_id_col37             = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col38        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col39       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col40      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col41    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col42     = cuda_evaluator1.next_trace_mask();
    m31 state_0_id_col43         = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col44        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col45       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col46      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col47    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col48     = cuda_evaluator1.next_trace_mask();
    m31 state_1_id_col49         = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col50        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col51       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col52      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col53    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col54     = cuda_evaluator1.next_trace_mask();
    m31 state_2_id_col55         = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col56        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col57       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col58      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col59    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col60     = cuda_evaluator1.next_trace_mask();
    m31 state_3_id_col61         = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col62        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col63       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col64      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col65    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col66     = cuda_evaluator1.next_trace_mask();
    m31 state_4_id_col67         = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col68        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col69       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col70      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col71    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col72     = cuda_evaluator1.next_trace_mask();
    m31 state_5_id_col73         = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col74        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col75       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col76      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col77    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col78     = cuda_evaluator1.next_trace_mask();
    m31 state_6_id_col79         = cuda_evaluator1.next_trace_mask();
    m31 low_16_bits_col80        = cuda_evaluator1.next_trace_mask();
    m31 high_16_bits_col81       = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_col82      = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_col83    = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_col84     = cuda_evaluator1.next_trace_mask();
    m31 state_7_id_col85         = cuda_evaluator1.next_trace_mask();
    m31 ms_8_bits_col86          = cuda_evaluator1.next_trace_mask();
    m31 ms_8_bits_col87          = cuda_evaluator1.next_trace_mask();
    m31 xor_col88                = cuda_evaluator1.next_trace_mask();
    m31 xor_col89                = cuda_evaluator1.next_trace_mask();
    m31 xor_col90                = cuda_evaluator1.next_trace_mask();
    m31 xor_col91                = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_0_col92 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_1_col93 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_2_col94 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_3_col95 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_4_col96 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_5_col97 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_6_col98 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_7_col99 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_8_col100 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_9_col101 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_10_col102 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_11_col103 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_12_col104 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_13_col105 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_14_col106 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_15_col107 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_16_col108 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_17_col109 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_18_col110 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_19_col111 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_20_col112 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_21_col113 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_22_col114 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_23_col115 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_24_col116 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_25_col117 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_26_col118 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_27_col119 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_28_col120 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_29_col121 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_30_col122 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_31_col123 = cuda_evaluator1.next_trace_mask();
    m31 blake_round_output_limb_32_col124 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_0_limb_0_col125 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_0_limb_1_col126 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_1_limb_0_col127 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_1_limb_1_col128 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_2_limb_0_col129 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_2_limb_1_col130 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_3_limb_0_col131 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_3_limb_1_col132 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_4_limb_0_col133 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_4_limb_1_col134 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_5_limb_0_col135 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_5_limb_1_col136 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_6_limb_0_col137 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_6_limb_1_col138 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_7_limb_0_col139 = cuda_evaluator1.next_trace_mask();
    m31 triple_xor_32_output_7_limb_1_col140 = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_0_col141 = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_0_col142 = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_0_col143 = cuda_evaluator1.next_trace_mask();
    m31 new_state_0_id_col144 = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_1_col145 = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_1_col146 = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_1_col147 = cuda_evaluator1.next_trace_mask();
    m31 new_state_1_id_col148 = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_2_col149 = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_2_col150 = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_2_col151 = cuda_evaluator1.next_trace_mask();
    m31 new_state_2_id_col152 = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_3_col153 = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_3_col154 = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_3_col155 = cuda_evaluator1.next_trace_mask();
    m31 new_state_3_id_col156 = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_4_col157 = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_4_col158 = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_4_col159 = cuda_evaluator1.next_trace_mask();
    m31 new_state_4_id_col160 = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_5_col161 = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_5_col162 = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_5_col163 = cuda_evaluator1.next_trace_mask();
    m31 new_state_5_id_col164 = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_6_col165 = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_6_col166 = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_6_col167 = cuda_evaluator1.next_trace_mask();
    m31 new_state_6_id_col168 = cuda_evaluator1.next_trace_mask();
    m31 low_7_ms_bits_7_col169 = cuda_evaluator1.next_trace_mask();
    m31 high_14_ms_bits_7_col170 = cuda_evaluator1.next_trace_mask();
    m31 high_5_ms_bits_7_col171 = cuda_evaluator1.next_trace_mask();
    m31 new_state_7_id_col172 = cuda_evaluator1.next_trace_mask();
    m31 enabler = cuda_evaluator1.next_trace_mask();

    m31 decode_blake_opcode_output_tmp_53f39_29_limb_0;
    m31 decode_blake_opcode_output_tmp_53f39_29_limb_1;
    m31 decode_blake_opcode_output_tmp_53f39_29_limb_2;
    m31 decode_blake_opcode_output_tmp_53f39_29_limb_3;
    m31 decode_blake_opcode_output_tmp_53f39_29_limb_4;
    m31 decode_blake_opcode_output_tmp_53f39_29_limb_5;
    m31 decode_blake_opcode_output_tmp_53f39_29_limb_6;
    evaluate_decode_blake_opcode(
        input_pc_col0, input_ap_col1, input_fp_col2,
        offset0_col3, offset1_col4, offset2_col5,
        dst_base_fp_col6, op0_base_fp_col7, op1_base_fp_col8,
        ap_update_add_1_col9, opcode_extension_col10,
        mem0_base_col11, op0_id_col12,
        op0_limb_0_col13, op0_limb_1_col14, op0_limb_2_col15, op0_limb_3_col16, partial_limb_msb_col17,
        mem1_base_col18, op1_id_col19,
        op1_limb_0_col20, op1_limb_1_col21, op1_limb_2_col22, op1_limb_3_col23, partial_limb_msb_col24,
        ap_id_col25, ap_limb_0_col26, ap_limb_1_col27, ap_limb_2_col28, ap_limb_3_col29, partial_limb_msb_col30,
        mem_dst_base_col31, low_16_bits_col32, high_16_bits_col33,
        low_7_ms_bits_col34, high_14_ms_bits_col35, high_5_ms_bits_col36,
        dst_id_col37,
        &decode_blake_opcode_output_tmp_53f39_29_limb_0,
        &decode_blake_opcode_output_tmp_53f39_29_limb_1,
        &decode_blake_opcode_output_tmp_53f39_29_limb_2,
        &decode_blake_opcode_output_tmp_53f39_29_limb_3,
        &decode_blake_opcode_output_tmp_53f39_29_limb_4,
        &decode_blake_opcode_output_tmp_53f39_29_limb_5,
        &decode_blake_opcode_output_tmp_53f39_29_limb_6,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
    // ==== CreateBlakeRoundInput ====
    m31 create_blake_round_input_output_tmp_53f39_114_limb[32];
    evaluate_create_blake_round_input(
        decode_blake_opcode_output_tmp_53f39_29_limb_0,
        low_16_bits_col32,
        high_16_bits_col33,
        decode_blake_opcode_output_tmp_53f39_29_limb_6,
        low_16_bits_col38, high_16_bits_col39, low_7_ms_bits_col40, high_14_ms_bits_col41, high_5_ms_bits_col42,
        state_0_id_col43, low_16_bits_col44, high_16_bits_col45, low_7_ms_bits_col46, high_14_ms_bits_col47, high_5_ms_bits_col48,
        state_1_id_col49, low_16_bits_col50, high_16_bits_col51, low_7_ms_bits_col52, high_14_ms_bits_col53, high_5_ms_bits_col54,
        state_2_id_col55, low_16_bits_col56, high_16_bits_col57, low_7_ms_bits_col58, high_14_ms_bits_col59, high_5_ms_bits_col60,
        state_3_id_col61, low_16_bits_col62, high_16_bits_col63, low_7_ms_bits_col64, high_14_ms_bits_col65, high_5_ms_bits_col66,
        state_4_id_col67, low_16_bits_col68, high_16_bits_col69, low_7_ms_bits_col70, high_14_ms_bits_col71, high_5_ms_bits_col72,
        state_5_id_col73, low_16_bits_col74, high_16_bits_col75, low_7_ms_bits_col76, high_14_ms_bits_col77, high_5_ms_bits_col78,
        state_6_id_col79, low_16_bits_col80, high_16_bits_col81, low_7_ms_bits_col82, high_14_ms_bits_col83, high_5_ms_bits_col84,
        state_7_id_col85, ms_8_bits_col86, ms_8_bits_col87, xor_col88, xor_col89, xor_col90, xor_col91,
        create_blake_round_input_output_tmp_53f39_114_limb,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
    // ==== RelationEntry for blake_round ====
    {
        m31 values[36] = {
            BLAKE_ROUND_RELATION_ID,
            seq, M31_0,
            low_16_bits_col38, high_16_bits_col39,
            low_16_bits_col44, high_16_bits_col45,
            low_16_bits_col50, high_16_bits_col51,
            low_16_bits_col56, high_16_bits_col57,
            low_16_bits_col62, high_16_bits_col63,
            low_16_bits_col68, high_16_bits_col69,
            low_16_bits_col74, high_16_bits_col75,
            low_16_bits_col80, high_16_bits_col81,
            M31_58983, M31_27145, M31_44677, M31_47975, M31_62322, M31_15470, M31_62778, M31_42319,
            create_blake_round_input_output_tmp_53f39_114_limb[24], create_blake_round_input_output_tmp_53f39_114_limb[25],
            M31_26764, M31_39685, create_blake_round_input_output_tmp_53f39_114_limb[28], create_blake_round_input_output_tmp_53f39_114_limb[29],
            M31_52505, M31_23520, decode_blake_opcode_output_tmp_53f39_29_limb_1
        };
        cuda_evaluator1.add_to_relation<36>(
            blake_compress_opcode_eval->common_lookup_elements,
            qm31{P - 1, 0 , 0 , 0},
            values
        );
    }
    {
        m31 values[36] = {
            BLAKE_ROUND_RELATION_ID,
            seq, M31_10,
            blake_round_output_limb_0_col92, blake_round_output_limb_1_col93, blake_round_output_limb_2_col94,
            blake_round_output_limb_3_col95, blake_round_output_limb_4_col96, blake_round_output_limb_5_col97, blake_round_output_limb_6_col98,
            blake_round_output_limb_7_col99, blake_round_output_limb_8_col100, blake_round_output_limb_9_col101,
            blake_round_output_limb_10_col102, blake_round_output_limb_11_col103, blake_round_output_limb_12_col104, blake_round_output_limb_13_col105,
            blake_round_output_limb_14_col106, blake_round_output_limb_15_col107, blake_round_output_limb_16_col108, blake_round_output_limb_17_col109,
            blake_round_output_limb_18_col110, blake_round_output_limb_19_col111, blake_round_output_limb_20_col112, blake_round_output_limb_21_col113,
            blake_round_output_limb_22_col114, blake_round_output_limb_23_col115, blake_round_output_limb_24_col116, blake_round_output_limb_25_col117,
            blake_round_output_limb_26_col118, blake_round_output_limb_27_col119, blake_round_output_limb_28_col120, blake_round_output_limb_29_col121,
            blake_round_output_limb_30_col122, blake_round_output_limb_31_col123, blake_round_output_limb_32_col124
        };
        cuda_evaluator1.add_to_relation<36>(
            blake_compress_opcode_eval->common_lookup_elements,
            qm31{1},
            values
        );
    }
    // ==== CreateBlakeOutput ====
    m31 create_blake_output_output_tmp_53f39_133_limb[16];
    evaluate_create_blake_output(
        low_16_bits_col38, high_16_bits_col39,
        low_16_bits_col44, high_16_bits_col45,
        low_16_bits_col50, high_16_bits_col51,
        low_16_bits_col56, high_16_bits_col57,
        low_16_bits_col62, high_16_bits_col63,
        low_16_bits_col68, high_16_bits_col69,
        low_16_bits_col74, high_16_bits_col75,
        low_16_bits_col80, high_16_bits_col81,
        blake_round_output_limb_0_col92, blake_round_output_limb_1_col93, blake_round_output_limb_2_col94, blake_round_output_limb_3_col95, blake_round_output_limb_4_col96,
        blake_round_output_limb_5_col97, blake_round_output_limb_6_col98, blake_round_output_limb_7_col99,
        blake_round_output_limb_8_col100, blake_round_output_limb_9_col101, blake_round_output_limb_10_col102,
        blake_round_output_limb_11_col103, blake_round_output_limb_12_col104, blake_round_output_limb_13_col105,
        blake_round_output_limb_14_col106, blake_round_output_limb_15_col107, blake_round_output_limb_16_col108,
        blake_round_output_limb_17_col109, blake_round_output_limb_18_col110, blake_round_output_limb_19_col111,
        blake_round_output_limb_20_col112, blake_round_output_limb_21_col113, blake_round_output_limb_22_col114,
        blake_round_output_limb_23_col115, blake_round_output_limb_24_col116, blake_round_output_limb_25_col117,
        blake_round_output_limb_26_col118, blake_round_output_limb_27_col119, blake_round_output_limb_28_col120,
        blake_round_output_limb_29_col121, blake_round_output_limb_30_col122, blake_round_output_limb_31_col123,
        triple_xor_32_output_0_limb_0_col125, triple_xor_32_output_0_limb_1_col126,
        triple_xor_32_output_1_limb_0_col127, triple_xor_32_output_1_limb_1_col128,
        triple_xor_32_output_2_limb_0_col129, triple_xor_32_output_2_limb_1_col130,
        triple_xor_32_output_3_limb_0_col131, triple_xor_32_output_3_limb_1_col132,
        triple_xor_32_output_4_limb_0_col133, triple_xor_32_output_4_limb_1_col134,
        triple_xor_32_output_5_limb_0_col135, triple_xor_32_output_5_limb_1_col136,
        triple_xor_32_output_6_limb_0_col137, triple_xor_32_output_6_limb_1_col138,
        triple_xor_32_output_7_limb_0_col139, triple_xor_32_output_7_limb_1_col140,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1,
        create_blake_output_output_tmp_53f39_133_limb
    );
    // ==== VerifyBlakeWord::evaluate (keeping variable names) ====
    // 0
    verify_blake_word_evaluate(
        decode_blake_opcode_output_tmp_53f39_29_limb_2,
        triple_xor_32_output_0_limb_0_col125,
        triple_xor_32_output_0_limb_1_col126,
        low_7_ms_bits_0_col141,
        high_14_ms_bits_0_col142,
        high_5_ms_bits_0_col143,
        new_state_0_id_col144,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
    // 1
    verify_blake_word_evaluate(
        add(decode_blake_opcode_output_tmp_53f39_29_limb_2, M31_1),
        triple_xor_32_output_1_limb_0_col127,
        triple_xor_32_output_1_limb_1_col128,
        low_7_ms_bits_1_col145,
        high_14_ms_bits_1_col146,
        high_5_ms_bits_1_col147,
        new_state_1_id_col148,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
    // 2
    verify_blake_word_evaluate(
        add(decode_blake_opcode_output_tmp_53f39_29_limb_2, M31_2),
        triple_xor_32_output_2_limb_0_col129,
        triple_xor_32_output_2_limb_1_col130,
        low_7_ms_bits_2_col149,
        high_14_ms_bits_2_col150,
        high_5_ms_bits_2_col151,
        new_state_2_id_col152,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
    // 3
    verify_blake_word_evaluate(
        add(decode_blake_opcode_output_tmp_53f39_29_limb_2, M31_3),
        triple_xor_32_output_3_limb_0_col131,
        triple_xor_32_output_3_limb_1_col132,
        low_7_ms_bits_3_col153,
        high_14_ms_bits_3_col154,
        high_5_ms_bits_3_col155,
        new_state_3_id_col156,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
    // 4
    verify_blake_word_evaluate(
        add(decode_blake_opcode_output_tmp_53f39_29_limb_2, M31_4),
        triple_xor_32_output_4_limb_0_col133,
        triple_xor_32_output_4_limb_1_col134,
        low_7_ms_bits_4_col157,
        high_14_ms_bits_4_col158,
        high_5_ms_bits_4_col159,
        new_state_4_id_col160,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
   // 5
    verify_blake_word_evaluate(
        add(decode_blake_opcode_output_tmp_53f39_29_limb_2, M31_5),
        triple_xor_32_output_5_limb_0_col135,
        triple_xor_32_output_5_limb_1_col136,
        low_7_ms_bits_5_col161,
        high_14_ms_bits_5_col162,
        high_5_ms_bits_5_col163,
        new_state_5_id_col164,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
   // 6
    verify_blake_word_evaluate(
        add(decode_blake_opcode_output_tmp_53f39_29_limb_2, M31_6),
        triple_xor_32_output_6_limb_0_col137,
        triple_xor_32_output_6_limb_1_col138,
        low_7_ms_bits_6_col165,
        high_14_ms_bits_6_col166,
        high_5_ms_bits_6_col167,
        new_state_6_id_col168,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
    // 7
    verify_blake_word_evaluate(
        add(decode_blake_opcode_output_tmp_53f39_29_limb_2, M31_7),
        triple_xor_32_output_7_limb_0_col139,
        triple_xor_32_output_7_limb_1_col140,
        low_7_ms_bits_7_col169,
        high_14_ms_bits_7_col170,
        high_5_ms_bits_7_col171,
        new_state_7_id_col172,
        blake_compress_opcode_eval->common_lookup_elements,
        &cuda_evaluator1
    );
    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator1.add_constraint(sub(mul(enabler, enabler), enabler));

    // ==== opcodes: relation add ====
    {
        m31 values[4] = {OPCODES_RELATION_ID, input_pc_col0, input_ap_col1, input_fp_col2};
        cuda_evaluator1.add_to_relation<4>(
            blake_compress_opcode_eval->common_lookup_elements,
            qm31{enabler, 0, 0, 0},
            values
        );
    }
    {
        m31 values[4] = {OPCODES_RELATION_ID, add(input_pc_col0, M31_1), add(input_ap_col1, ap_update_add_1_col9), input_fp_col2};
        cuda_evaluator1.add_to_relation<4>(
            blake_compress_opcode_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},  // negative multiplicity using neg() for proper M31 field negation
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;

}

void evaluate_blake_compress_opcode(
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
    BlakeCompressOpcode_Eval *blake_compress_opcode_eval = (BlakeCompressOpcode_Eval *) eval;
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    BlakeCompressOpcode_Eval *device_blake_compress_opcode_eval = cuda_malloc<BlakeCompressOpcode_Eval>(1);
    cuda_mem_copy_host_to_device<BlakeCompressOpcode_Eval>(blake_compress_opcode_eval, device_blake_compress_opcode_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constrain_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_blake_compress_opcode");

    int block_dim = eval_domain_size < BLAKE_ROUND_THREAD_COUNT_MAX ? eval_domain_size : BLAKE_ROUND_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_blake_compress_opcode_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_blake_compress_opcode_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constrain_index_array
        );
    } else {
        evaluate_blake_compress_opcode_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_blake_compress_opcode_eval,
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
    global_timer.end("evaluate_blake_compress_opcode");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_blake_compress_opcode_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constrain_index_array);
}