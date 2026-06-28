#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_blake_round.cuh"
#include "evaluate_read_blake_word.cuh"
#include "evaluate_common.cuh"

#define BLAKE_ROUND_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_blake_round_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    BlakeRound_Eval *eval,
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
        {0},
        0,
        {0},
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );
    const m31 M31_1 = { 1 };
    m31 input_limb_0_col0 = cuda_evaluator.next_trace_mask();
    m31 input_limb_1_col1 = cuda_evaluator.next_trace_mask();
    m31 input_limb_2_col2 = cuda_evaluator.next_trace_mask();
    m31 input_limb_3_col3 = cuda_evaluator.next_trace_mask();
    m31 input_limb_4_col4 = cuda_evaluator.next_trace_mask();
    m31 input_limb_5_col5 = cuda_evaluator.next_trace_mask();
    m31 input_limb_6_col6 = cuda_evaluator.next_trace_mask();
    m31 input_limb_7_col7 = cuda_evaluator.next_trace_mask();
    m31 input_limb_8_col8 = cuda_evaluator.next_trace_mask();
    m31 input_limb_9_col9 = cuda_evaluator.next_trace_mask();
    m31 input_limb_10_col10 = cuda_evaluator.next_trace_mask();
    m31 input_limb_11_col11 = cuda_evaluator.next_trace_mask();
    m31 input_limb_12_col12 = cuda_evaluator.next_trace_mask();
    m31 input_limb_13_col13 = cuda_evaluator.next_trace_mask();
    m31 input_limb_14_col14 = cuda_evaluator.next_trace_mask();
    m31 input_limb_15_col15 = cuda_evaluator.next_trace_mask();
    m31 input_limb_16_col16 = cuda_evaluator.next_trace_mask();
    m31 input_limb_17_col17 = cuda_evaluator.next_trace_mask();
    m31 input_limb_18_col18 = cuda_evaluator.next_trace_mask();
    m31 input_limb_19_col19 = cuda_evaluator.next_trace_mask();
    m31 input_limb_20_col20 = cuda_evaluator.next_trace_mask();
    m31 input_limb_21_col21 = cuda_evaluator.next_trace_mask();
    m31 input_limb_22_col22 = cuda_evaluator.next_trace_mask();
    m31 input_limb_23_col23 = cuda_evaluator.next_trace_mask();
    m31 input_limb_24_col24 = cuda_evaluator.next_trace_mask();
    m31 input_limb_25_col25 = cuda_evaluator.next_trace_mask();
    m31 input_limb_26_col26 = cuda_evaluator.next_trace_mask();
    m31 input_limb_27_col27 = cuda_evaluator.next_trace_mask();
    m31 input_limb_28_col28 = cuda_evaluator.next_trace_mask();
    m31 input_limb_29_col29 = cuda_evaluator.next_trace_mask();
    m31 input_limb_30_col30 = cuda_evaluator.next_trace_mask();
    m31 input_limb_31_col31 = cuda_evaluator.next_trace_mask();
    m31 input_limb_32_col32 = cuda_evaluator.next_trace_mask();
    m31 input_limb_33_col33 = cuda_evaluator.next_trace_mask();
    m31 input_limb_34_col34 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_0_col35 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_1_col36 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_2_col37 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_3_col38 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_4_col39 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_5_col40 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_6_col41 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_7_col42 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_8_col43 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_9_col44 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_10_col45 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_11_col46 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_12_col47 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_13_col48 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_14_col49 = cuda_evaluator.next_trace_mask();
    m31 blake_round_sigma_output_limb_15_col50 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col51 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col52 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col53 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col54 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col55 = cuda_evaluator.next_trace_mask();
    m31 message_word_0_id_col56 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col57 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col58 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col59 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col60 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col61 = cuda_evaluator.next_trace_mask();
    m31 message_word_1_id_col62 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col63 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col64 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col65 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col66 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col67 = cuda_evaluator.next_trace_mask();
    m31 message_word_2_id_col68 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col69 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col70 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col71 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col72 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col73 = cuda_evaluator.next_trace_mask();
    m31 message_word_3_id_col74 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col75 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col76 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col77 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col78 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col79 = cuda_evaluator.next_trace_mask();
    m31 message_word_4_id_col80 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col81 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col82 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col83 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col84 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col85 = cuda_evaluator.next_trace_mask();
    m31 message_word_5_id_col86 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col87 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col88 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col89 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col90 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col91 = cuda_evaluator.next_trace_mask();
    m31 message_word_6_id_col92 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col93 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col94 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col95 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col96 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col97 = cuda_evaluator.next_trace_mask();
    m31 message_word_7_id_col98 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col99 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col100 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col101 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col102 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col103 = cuda_evaluator.next_trace_mask();
    m31 message_word_8_id_col104 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col105 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col106 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col107 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col108 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col109 = cuda_evaluator.next_trace_mask();
    m31 message_word_9_id_col110 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col111 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col112 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col113 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col114 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col115 = cuda_evaluator.next_trace_mask();
    m31 message_word_10_id_col116 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col117 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col118 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col119 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col120 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col121 = cuda_evaluator.next_trace_mask();
    m31 message_word_11_id_col122 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col123 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col124 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col125 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col126 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col127 = cuda_evaluator.next_trace_mask();
    m31 message_word_12_id_col128 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col129 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col130 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col131 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col132 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col133 = cuda_evaluator.next_trace_mask();
    m31 message_word_13_id_col134 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col135 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col136 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col137 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col138 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col139 = cuda_evaluator.next_trace_mask();
    m31 message_word_14_id_col140 = cuda_evaluator.next_trace_mask();
    m31 low_16_bits_col141 = cuda_evaluator.next_trace_mask();
    m31 high_16_bits_col142 = cuda_evaluator.next_trace_mask();
    m31 low_7_ms_bits_col143 = cuda_evaluator.next_trace_mask();
    m31 high_14_ms_bits_col144 = cuda_evaluator.next_trace_mask();
    m31 high_5_ms_bits_col145 = cuda_evaluator.next_trace_mask();
    m31 message_word_15_id_col146 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_0_col147 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_1_col148 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_2_col149 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_3_col150 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_4_col151 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_5_col152 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_6_col153 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_7_col154 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_0_col155 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_1_col156 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_2_col157 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_3_col158 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_4_col159 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_5_col160 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_6_col161 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_7_col162 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_0_col163 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_1_col164 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_2_col165 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_3_col166 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_4_col167 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_5_col168 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_6_col169 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_7_col170 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_0_col171 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_1_col172 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_2_col173 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_3_col174 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_4_col175 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_5_col176 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_6_col177 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_7_col178 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_0_col179 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_1_col180 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_2_col181 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_3_col182 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_4_col183 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_5_col184 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_6_col185 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_7_col186 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_0_col187 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_1_col188 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_2_col189 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_3_col190 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_4_col191 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_5_col192 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_6_col193 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_7_col194 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_0_col195 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_1_col196 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_2_col197 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_3_col198 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_4_col199 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_5_col200 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_6_col201 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_7_col202 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_0_col203 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_1_col204 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_2_col205 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_3_col206 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_4_col207 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_5_col208 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_6_col209 = cuda_evaluator.next_trace_mask();
    m31 blake_g_output_limb_7_col210 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    m31 values_blake_round_sigma[18] = {
        BLAKE_ROUND_SIGMA_RELATION_ID,
        input_limb_1_col1,
        blake_round_sigma_output_limb_0_col35,
        blake_round_sigma_output_limb_1_col36,
        blake_round_sigma_output_limb_2_col37,
        blake_round_sigma_output_limb_3_col38,
        blake_round_sigma_output_limb_4_col39,
        blake_round_sigma_output_limb_5_col40,
        blake_round_sigma_output_limb_6_col41,
        blake_round_sigma_output_limb_7_col42,
        blake_round_sigma_output_limb_8_col43,
        blake_round_sigma_output_limb_9_col44,
        blake_round_sigma_output_limb_10_col45,
        blake_round_sigma_output_limb_11_col46,
        blake_round_sigma_output_limb_12_col47,
        blake_round_sigma_output_limb_13_col48,
        blake_round_sigma_output_limb_14_col49,
        blake_round_sigma_output_limb_15_col50,
    };

    cuda_evaluator.add_to_relation<18>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values_blake_round_sigma);

    m31 read_blake_word_output_tmp_92ff8_9_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_9_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_0_col35),
        low_16_bits_col51,
        high_16_bits_col52,
        low_7_ms_bits_col53,
        high_14_ms_bits_col54,
        high_5_ms_bits_col55,
        message_word_0_id_col56,
        &read_blake_word_output_tmp_92ff8_9_limb_0,
        &read_blake_word_output_tmp_92ff8_9_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_18_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_18_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_1_col36),
        low_16_bits_col57,
        high_16_bits_col58,
        low_7_ms_bits_col59,
        high_14_ms_bits_col60,
        high_5_ms_bits_col61,
        message_word_1_id_col62,
        &read_blake_word_output_tmp_92ff8_18_limb_0,
        &read_blake_word_output_tmp_92ff8_18_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_27_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_27_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_2_col37),
        low_16_bits_col63,
        high_16_bits_col64,
        low_7_ms_bits_col65,
        high_14_ms_bits_col66,
        high_5_ms_bits_col67,
        message_word_2_id_col68,
        &read_blake_word_output_tmp_92ff8_27_limb_0,
        &read_blake_word_output_tmp_92ff8_27_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_36_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_36_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_3_col38),
        low_16_bits_col69,
        high_16_bits_col70,
        low_7_ms_bits_col71,
        high_14_ms_bits_col72,
        high_5_ms_bits_col73,
        message_word_3_id_col74,
        &read_blake_word_output_tmp_92ff8_36_limb_0,
        &read_blake_word_output_tmp_92ff8_36_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_45_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_45_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_4_col39),
        low_16_bits_col75,
        high_16_bits_col76,
        low_7_ms_bits_col77,
        high_14_ms_bits_col78,
        high_5_ms_bits_col79,
        message_word_4_id_col80,
        &read_blake_word_output_tmp_92ff8_45_limb_0,
        &read_blake_word_output_tmp_92ff8_45_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_54_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_54_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_5_col40),
        low_16_bits_col81,
        high_16_bits_col82,
        low_7_ms_bits_col83,
        high_14_ms_bits_col84,
        high_5_ms_bits_col85,
        message_word_5_id_col86,
        &read_blake_word_output_tmp_92ff8_54_limb_0,
        &read_blake_word_output_tmp_92ff8_54_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_63_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_63_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_6_col41),
        low_16_bits_col87,
        high_16_bits_col88,
        low_7_ms_bits_col89,
        high_14_ms_bits_col90,
        high_5_ms_bits_col91,
        message_word_6_id_col92,
        &read_blake_word_output_tmp_92ff8_63_limb_0,
        &read_blake_word_output_tmp_92ff8_63_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_72_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_72_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_7_col42),
        low_16_bits_col93,
        high_16_bits_col94,
        low_7_ms_bits_col95,
        high_14_ms_bits_col96,
        high_5_ms_bits_col97,
        message_word_7_id_col98,
        &read_blake_word_output_tmp_92ff8_72_limb_0,
        &read_blake_word_output_tmp_92ff8_72_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_81_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_81_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_8_col43),
        low_16_bits_col99,
        high_16_bits_col100,
        low_7_ms_bits_col101,
        high_14_ms_bits_col102,
        high_5_ms_bits_col103,
        message_word_8_id_col104,
        &read_blake_word_output_tmp_92ff8_81_limb_0,
        &read_blake_word_output_tmp_92ff8_81_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_90_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_90_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_9_col44),
        low_16_bits_col105,
        high_16_bits_col106,
        low_7_ms_bits_col107,
        high_14_ms_bits_col108,
        high_5_ms_bits_col109,
        message_word_9_id_col110,
        &read_blake_word_output_tmp_92ff8_90_limb_0,
        &read_blake_word_output_tmp_92ff8_90_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_99_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_99_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_10_col45),
        low_16_bits_col111,
        high_16_bits_col112,
        low_7_ms_bits_col113,
        high_14_ms_bits_col114,
        high_5_ms_bits_col115,
        message_word_10_id_col116,
        &read_blake_word_output_tmp_92ff8_99_limb_0,
        &read_blake_word_output_tmp_92ff8_99_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_108_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_108_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_11_col46),
        low_16_bits_col117,
        high_16_bits_col118,
        low_7_ms_bits_col119,
        high_14_ms_bits_col120,
        high_5_ms_bits_col121,
        message_word_11_id_col122,
        &read_blake_word_output_tmp_92ff8_108_limb_0,
        &read_blake_word_output_tmp_92ff8_108_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_117_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_117_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_12_col47),
        low_16_bits_col123,
        high_16_bits_col124,
        low_7_ms_bits_col125,
        high_14_ms_bits_col126,
        high_5_ms_bits_col127,
        message_word_12_id_col128,
        &read_blake_word_output_tmp_92ff8_117_limb_0,
        &read_blake_word_output_tmp_92ff8_117_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_126_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_126_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_13_col48),
        low_16_bits_col129,
        high_16_bits_col130,
        low_7_ms_bits_col131,
        high_14_ms_bits_col132,
        high_5_ms_bits_col133,
        message_word_13_id_col134,
        &read_blake_word_output_tmp_92ff8_126_limb_0,
        &read_blake_word_output_tmp_92ff8_126_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_135_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_135_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_14_col49),
        low_16_bits_col135,
        high_16_bits_col136,
        low_7_ms_bits_col137,
        high_14_ms_bits_col138,
        high_5_ms_bits_col139,
        message_word_14_id_col140,
        &read_blake_word_output_tmp_92ff8_135_limb_0,
        &read_blake_word_output_tmp_92ff8_135_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 read_blake_word_output_tmp_92ff8_144_limb_0 = {0};
    m31 read_blake_word_output_tmp_92ff8_144_limb_1 = {0};
    read_blake_word_evaluate(
        add(input_limb_34_col34, blake_round_sigma_output_limb_15_col50),
        low_16_bits_col141,
        high_16_bits_col142,
        low_7_ms_bits_col143,
        high_14_ms_bits_col144,
        high_5_ms_bits_col145,
        message_word_15_id_col146,
        &read_blake_word_output_tmp_92ff8_144_limb_0,
        &read_blake_word_output_tmp_92ff8_144_limb_1,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 value_blake_g_0[21] = {
        BLAKE_G_RELATION_ID,
        input_limb_2_col2,
        input_limb_3_col3,
        input_limb_10_col10,
        input_limb_11_col11,
        input_limb_18_col18,
        input_limb_19_col19,
        input_limb_26_col26,
        input_limb_27_col27,
        low_16_bits_col51,
        high_16_bits_col52,
        low_16_bits_col57,
        high_16_bits_col58,
        blake_g_output_limb_0_col147,
        blake_g_output_limb_1_col148,
        blake_g_output_limb_2_col149,
        blake_g_output_limb_3_col150,
        blake_g_output_limb_4_col151,
        blake_g_output_limb_5_col152,
        blake_g_output_limb_6_col153,
        blake_g_output_limb_7_col154,
    };
    cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, value_blake_g_0);

    m31 value_blake_g_1[21] = {
        BLAKE_G_RELATION_ID,
        input_limb_4_col4,
        input_limb_5_col5,
        input_limb_12_col12,
        input_limb_13_col13,
        input_limb_20_col20,
        input_limb_21_col21,
        input_limb_28_col28,
        input_limb_29_col29,
        low_16_bits_col63,
        high_16_bits_col64,
        low_16_bits_col69,
        high_16_bits_col70,
        blake_g_output_limb_0_col155,
        blake_g_output_limb_1_col156,
        blake_g_output_limb_2_col157,
        blake_g_output_limb_3_col158,
        blake_g_output_limb_4_col159,
        blake_g_output_limb_5_col160,
        blake_g_output_limb_6_col161,
        blake_g_output_limb_7_col162,
    };
    cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, value_blake_g_1);

    m31 value_blake_g_2[21] = {
        BLAKE_G_RELATION_ID,
        input_limb_6_col6,
        input_limb_7_col7,
        input_limb_14_col14,
        input_limb_15_col15,
        input_limb_22_col22,
        input_limb_23_col23,
        input_limb_30_col30,
        input_limb_31_col31,
        low_16_bits_col75,
        high_16_bits_col76,
        low_16_bits_col81,
        high_16_bits_col82,
        blake_g_output_limb_0_col163,
        blake_g_output_limb_1_col164,
        blake_g_output_limb_2_col165,
        blake_g_output_limb_3_col166,
        blake_g_output_limb_4_col167,
        blake_g_output_limb_5_col168,
        blake_g_output_limb_6_col169,
        blake_g_output_limb_7_col170,
    };
    cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, value_blake_g_2);

    m31 value_blake_g_3[21] = {
        BLAKE_G_RELATION_ID,
        input_limb_8_col8,
        input_limb_9_col9,
        input_limb_16_col16,
        input_limb_17_col17,
        input_limb_24_col24,
        input_limb_25_col25,
        input_limb_32_col32,
        input_limb_33_col33,
        low_16_bits_col87,
        high_16_bits_col88,
        low_16_bits_col93,
        high_16_bits_col94,
        blake_g_output_limb_0_col171,
        blake_g_output_limb_1_col172,
        blake_g_output_limb_2_col173,
        blake_g_output_limb_3_col174,
        blake_g_output_limb_4_col175,
        blake_g_output_limb_5_col176,
        blake_g_output_limb_6_col177,
        blake_g_output_limb_7_col178,
    };
    cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, value_blake_g_3);

    m31 value_blake_g_4[21] = {
        BLAKE_G_RELATION_ID,
        blake_g_output_limb_0_col147,
        blake_g_output_limb_1_col148,
        blake_g_output_limb_2_col157,
        blake_g_output_limb_3_col158,
        blake_g_output_limb_4_col167,
        blake_g_output_limb_5_col168,
        blake_g_output_limb_6_col177,
        blake_g_output_limb_7_col178,
        low_16_bits_col99,
        high_16_bits_col100,
        low_16_bits_col105,
        high_16_bits_col106,
        blake_g_output_limb_0_col179,
        blake_g_output_limb_1_col180,
        blake_g_output_limb_2_col181,
        blake_g_output_limb_3_col182,
        blake_g_output_limb_4_col183,
        blake_g_output_limb_5_col184,
        blake_g_output_limb_6_col185,
        blake_g_output_limb_7_col186,
    };
    cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, value_blake_g_4);

    m31 value_blake_g_5[21] = {
        BLAKE_G_RELATION_ID,
        blake_g_output_limb_0_col155,
        blake_g_output_limb_1_col156,
        blake_g_output_limb_2_col165,
        blake_g_output_limb_3_col166,
        blake_g_output_limb_4_col175,
        blake_g_output_limb_5_col176,
        blake_g_output_limb_6_col153,
        blake_g_output_limb_7_col154,
        low_16_bits_col111,
        high_16_bits_col112,
        low_16_bits_col117,
        high_16_bits_col118,
        blake_g_output_limb_0_col187,
        blake_g_output_limb_1_col188,
        blake_g_output_limb_2_col189,
        blake_g_output_limb_3_col190,
        blake_g_output_limb_4_col191,
        blake_g_output_limb_5_col192,
        blake_g_output_limb_6_col193,
        blake_g_output_limb_7_col194,
    };
    cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, value_blake_g_5);

    m31 value_blake_g_6[21] = {
        BLAKE_G_RELATION_ID,
        blake_g_output_limb_0_col163,
        blake_g_output_limb_1_col164,
        blake_g_output_limb_2_col173,
        blake_g_output_limb_3_col174,
        blake_g_output_limb_4_col151,
        blake_g_output_limb_5_col152,
        blake_g_output_limb_6_col161,
        blake_g_output_limb_7_col162,
        low_16_bits_col123,
        high_16_bits_col124,
        low_16_bits_col129,
        high_16_bits_col130,
        blake_g_output_limb_0_col195,
        blake_g_output_limb_1_col196,
        blake_g_output_limb_2_col197,
        blake_g_output_limb_3_col198,
        blake_g_output_limb_4_col199,
        blake_g_output_limb_5_col200,
        blake_g_output_limb_6_col201,
        blake_g_output_limb_7_col202,
    };
    cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, value_blake_g_6);

    m31 value_blake_g_7[21] = {
        BLAKE_G_RELATION_ID,
        blake_g_output_limb_0_col171,
        blake_g_output_limb_1_col172,
        blake_g_output_limb_2_col149,
        blake_g_output_limb_3_col150,
        blake_g_output_limb_4_col159,
        blake_g_output_limb_5_col160,
        blake_g_output_limb_6_col169,
        blake_g_output_limb_7_col170,
        low_16_bits_col135,
        high_16_bits_col136,
        low_16_bits_col141,
        high_16_bits_col142,
        blake_g_output_limb_0_col203,
        blake_g_output_limb_1_col204,
        blake_g_output_limb_2_col205,
        blake_g_output_limb_3_col206,
        blake_g_output_limb_4_col207,
        blake_g_output_limb_5_col208,
        blake_g_output_limb_6_col209,
        blake_g_output_limb_7_col210,
    };
    cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, value_blake_g_7);

    // blake_round_lookup_elements, enabler
    m31 value_blake_round_0[36] = {
        BLAKE_ROUND_RELATION_ID,
        input_limb_0_col0,
        input_limb_1_col1,
        input_limb_2_col2,
        input_limb_3_col3,
        input_limb_4_col4,
        input_limb_5_col5,
        input_limb_6_col6,
        input_limb_7_col7,
        input_limb_8_col8,
        input_limb_9_col9,
        input_limb_10_col10,
        input_limb_11_col11,
        input_limb_12_col12,
        input_limb_13_col13,
        input_limb_14_col14,
        input_limb_15_col15,
        input_limb_16_col16,
        input_limb_17_col17,
        input_limb_18_col18,
        input_limb_19_col19,
        input_limb_20_col20,
        input_limb_21_col21,
        input_limb_22_col22,
        input_limb_23_col23,
        input_limb_24_col24,
        input_limb_25_col25,
        input_limb_26_col26,
        input_limb_27_col27,
        input_limb_28_col28,
        input_limb_29_col29,
        input_limb_30_col30,
        input_limb_31_col31,
        input_limb_32_col32,
        input_limb_33_col33,
        input_limb_34_col34,
    };
    cuda_evaluator.add_to_relation<36>(eval->common_lookup_elements, qm31{{enabler, 0}, {0, 0}}, value_blake_round_0);

    m31 value_blake_round_1[36] = {
        BLAKE_ROUND_RELATION_ID,
        input_limb_0_col0,
        add(input_limb_1_col1, M31_1),
        blake_g_output_limb_0_col179,
        blake_g_output_limb_1_col180,
        blake_g_output_limb_0_col187,
        blake_g_output_limb_1_col188,
        blake_g_output_limb_0_col195,
        blake_g_output_limb_1_col196,
        blake_g_output_limb_0_col203,
        blake_g_output_limb_1_col204,
        blake_g_output_limb_2_col205,
        blake_g_output_limb_3_col206,
        blake_g_output_limb_2_col181,
        blake_g_output_limb_3_col182,
        blake_g_output_limb_2_col189,
        blake_g_output_limb_3_col190,
        blake_g_output_limb_2_col197,
        blake_g_output_limb_3_col198,
        blake_g_output_limb_4_col199,
        blake_g_output_limb_5_col200,
        blake_g_output_limb_4_col207,
        blake_g_output_limb_5_col208,
        blake_g_output_limb_4_col183,
        blake_g_output_limb_5_col184,
        blake_g_output_limb_4_col191,
        blake_g_output_limb_5_col192,
        blake_g_output_limb_6_col193,
        blake_g_output_limb_7_col194,
        blake_g_output_limb_6_col201,
        blake_g_output_limb_7_col202,
        blake_g_output_limb_6_col209,
        blake_g_output_limb_7_col210,
        blake_g_output_limb_6_col185,
        blake_g_output_limb_7_col186,
        input_limb_34_col34,
    };
    cuda_evaluator.add_to_relation<36>(
        eval->common_lookup_elements,
        qm31{{(enabler == 0 ? 0 : P - enabler), 0}, {0, 0}},
        value_blake_round_1);

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;

}

extern "C" void evaluate_blake_round(
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

    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    BlakeRound_Eval *device_blake_round_eval = cuda_malloc<BlakeRound_Eval>(1);
    cuda_mem_copy_host_to_device<BlakeRound_Eval>((BlakeRound_Eval*)eval, device_blake_round_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_blake_round");

    int block_dim = eval_domain_size < BLAKE_ROUND_THREAD_COUNT_MAX ? eval_domain_size : BLAKE_ROUND_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_blake_round_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_blake_round_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_blake_round_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_blake_round_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
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
        generic_constraint_post_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
        g_should_accumulate_host  // Read from global variable set by dispatcher
    );
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_blake_round");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_blake_round_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
