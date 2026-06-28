/*
============================================
PedersenAggregatorWindowBits18 CUDA Evaluator
============================================

Component: PedersenAggregatorWindowBits18 (206 trace columns, 1 preprocessed column)
Translated from: cairo-air/src/components/pedersen_aggregator_window_bits_18.rs

Flow:
1. Read preprocessed column (seq) from trace0
2. Read 206 trace columns from trace1
3. 2x ReadPositiveKnownIdNumBits252 (inlined: just MemoryIdToBig lookup each)
4. 2x VerifyReduced252 (subroutine call for field element bounds checking)
5. 4x PartialEcMulWindowBits18 lookups (2 chains: input+output each)
6. 1x MemoryIdToBig lookup for output
7. 1x PedersenAggregatorWindowBits18 PROVIDE relation (negative multiplicity)

============================================
*/

#include <cstdio>
#include <vector>

#include "evaluate_pedersen_aggregator.cuh"
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_common.cuh"
#include "relations.cuh"
#include "../evaluate_verify_reduced_252.cuh"

#define PEDERSEN_AGGREGATOR_WB18_THREAD_COUNT_MAX 256

// =====================================================================
// Pre-Kernel: Read trace columns, evaluate constraints, build logup fractions
// =====================================================================
template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_pedersen_aggregator_window_bits_18_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PedersenAggregator_WB18_Eval *agg_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    // Constants
    const m31 M31_512 = m31(512);
    const m31 M31_2 = m31(2);
    const m31 M31_1 = m31(1);

    // Evaluator for preprocessed trace (trace0) -- reads seq
    EvaluatorT cuda_evaluator0(
        trace0_evaluations,
        random_coeff_powers,
        0,
        row,
        {0},
        0,
        {0},
        domain_log_size,
        eval_domain_log_size,
        nullptr,
        0
    );

    // Read 1 preprocessed column: seq
    m31 seq = cuda_evaluator0.next_trace_mask();

    // Evaluator for base trace (trace1) -- reads 206 trace columns + builds logup
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

    // ===================== Read all 206 trace columns =====================
    // col0-col2: input limbs (memory IDs)
    m31 input_limb_0_col0 = cuda_evaluator.next_trace_mask();
    m31 input_limb_1_col1 = cuda_evaluator.next_trace_mask();
    m31 input_limb_2_col2 = cuda_evaluator.next_trace_mask();

    // col3-col30: value limbs for first input (28 limbs)
    m31 value_limb_0_col3 = cuda_evaluator.next_trace_mask();
    m31 value_limb_1_col4 = cuda_evaluator.next_trace_mask();
    m31 value_limb_2_col5 = cuda_evaluator.next_trace_mask();
    m31 value_limb_3_col6 = cuda_evaluator.next_trace_mask();
    m31 value_limb_4_col7 = cuda_evaluator.next_trace_mask();
    m31 value_limb_5_col8 = cuda_evaluator.next_trace_mask();
    m31 value_limb_6_col9 = cuda_evaluator.next_trace_mask();
    m31 value_limb_7_col10 = cuda_evaluator.next_trace_mask();
    m31 value_limb_8_col11 = cuda_evaluator.next_trace_mask();
    m31 value_limb_9_col12 = cuda_evaluator.next_trace_mask();
    m31 value_limb_10_col13 = cuda_evaluator.next_trace_mask();
    m31 value_limb_11_col14 = cuda_evaluator.next_trace_mask();
    m31 value_limb_12_col15 = cuda_evaluator.next_trace_mask();
    m31 value_limb_13_col16 = cuda_evaluator.next_trace_mask();
    m31 value_limb_14_col17 = cuda_evaluator.next_trace_mask();
    m31 value_limb_15_col18 = cuda_evaluator.next_trace_mask();
    m31 value_limb_16_col19 = cuda_evaluator.next_trace_mask();
    m31 value_limb_17_col20 = cuda_evaluator.next_trace_mask();
    m31 value_limb_18_col21 = cuda_evaluator.next_trace_mask();
    m31 value_limb_19_col22 = cuda_evaluator.next_trace_mask();
    m31 value_limb_20_col23 = cuda_evaluator.next_trace_mask();
    m31 value_limb_21_col24 = cuda_evaluator.next_trace_mask();
    m31 value_limb_22_col25 = cuda_evaluator.next_trace_mask();
    m31 value_limb_23_col26 = cuda_evaluator.next_trace_mask();
    m31 value_limb_24_col27 = cuda_evaluator.next_trace_mask();
    m31 value_limb_25_col28 = cuda_evaluator.next_trace_mask();
    m31 value_limb_26_col29 = cuda_evaluator.next_trace_mask();
    m31 value_limb_27_col30 = cuda_evaluator.next_trace_mask();

    // col31-col58: value limbs for second input (28 limbs)
    m31 value_limb_0_col31 = cuda_evaluator.next_trace_mask();
    m31 value_limb_1_col32 = cuda_evaluator.next_trace_mask();
    m31 value_limb_2_col33 = cuda_evaluator.next_trace_mask();
    m31 value_limb_3_col34 = cuda_evaluator.next_trace_mask();
    m31 value_limb_4_col35 = cuda_evaluator.next_trace_mask();
    m31 value_limb_5_col36 = cuda_evaluator.next_trace_mask();
    m31 value_limb_6_col37 = cuda_evaluator.next_trace_mask();
    m31 value_limb_7_col38 = cuda_evaluator.next_trace_mask();
    m31 value_limb_8_col39 = cuda_evaluator.next_trace_mask();
    m31 value_limb_9_col40 = cuda_evaluator.next_trace_mask();
    m31 value_limb_10_col41 = cuda_evaluator.next_trace_mask();
    m31 value_limb_11_col42 = cuda_evaluator.next_trace_mask();
    m31 value_limb_12_col43 = cuda_evaluator.next_trace_mask();
    m31 value_limb_13_col44 = cuda_evaluator.next_trace_mask();
    m31 value_limb_14_col45 = cuda_evaluator.next_trace_mask();
    m31 value_limb_15_col46 = cuda_evaluator.next_trace_mask();
    m31 value_limb_16_col47 = cuda_evaluator.next_trace_mask();
    m31 value_limb_17_col48 = cuda_evaluator.next_trace_mask();
    m31 value_limb_18_col49 = cuda_evaluator.next_trace_mask();
    m31 value_limb_19_col50 = cuda_evaluator.next_trace_mask();
    m31 value_limb_20_col51 = cuda_evaluator.next_trace_mask();
    m31 value_limb_21_col52 = cuda_evaluator.next_trace_mask();
    m31 value_limb_22_col53 = cuda_evaluator.next_trace_mask();
    m31 value_limb_23_col54 = cuda_evaluator.next_trace_mask();
    m31 value_limb_24_col55 = cuda_evaluator.next_trace_mask();
    m31 value_limb_25_col56 = cuda_evaluator.next_trace_mask();
    m31 value_limb_26_col57 = cuda_evaluator.next_trace_mask();
    m31 value_limb_27_col58 = cuda_evaluator.next_trace_mask();

    // col59-col64: VerifyReduced252 columns (2 sets of 3)
    m31 ms_limb_is_max_col59 = cuda_evaluator.next_trace_mask();
    m31 ms_and_mid_limbs_are_max_col60 = cuda_evaluator.next_trace_mask();
    m31 rc_input_col61 = cuda_evaluator.next_trace_mask();
    m31 ms_limb_is_max_col62 = cuda_evaluator.next_trace_mask();
    m31 ms_and_mid_limbs_are_max_col63 = cuda_evaluator.next_trace_mask();
    m31 rc_input_col64 = cuda_evaluator.next_trace_mask();

    // col65-col134: PartialEcMulWindowBits18 output limbs for chain 0 (70 limbs)
    m31 output_limb_0_col65 = cuda_evaluator.next_trace_mask();
    m31 output_limb_1_col66 = cuda_evaluator.next_trace_mask();
    m31 output_limb_2_col67 = cuda_evaluator.next_trace_mask();
    m31 output_limb_3_col68 = cuda_evaluator.next_trace_mask();
    m31 output_limb_4_col69 = cuda_evaluator.next_trace_mask();
    m31 output_limb_5_col70 = cuda_evaluator.next_trace_mask();
    m31 output_limb_6_col71 = cuda_evaluator.next_trace_mask();
    m31 output_limb_7_col72 = cuda_evaluator.next_trace_mask();
    m31 output_limb_8_col73 = cuda_evaluator.next_trace_mask();
    m31 output_limb_9_col74 = cuda_evaluator.next_trace_mask();
    m31 output_limb_10_col75 = cuda_evaluator.next_trace_mask();
    m31 output_limb_11_col76 = cuda_evaluator.next_trace_mask();
    m31 output_limb_12_col77 = cuda_evaluator.next_trace_mask();
    m31 output_limb_13_col78 = cuda_evaluator.next_trace_mask();
    m31 output_limb_14_col79 = cuda_evaluator.next_trace_mask();
    m31 output_limb_15_col80 = cuda_evaluator.next_trace_mask();
    m31 output_limb_16_col81 = cuda_evaluator.next_trace_mask();
    m31 output_limb_17_col82 = cuda_evaluator.next_trace_mask();
    m31 output_limb_18_col83 = cuda_evaluator.next_trace_mask();
    m31 output_limb_19_col84 = cuda_evaluator.next_trace_mask();
    m31 output_limb_20_col85 = cuda_evaluator.next_trace_mask();
    m31 output_limb_21_col86 = cuda_evaluator.next_trace_mask();
    m31 output_limb_22_col87 = cuda_evaluator.next_trace_mask();
    m31 output_limb_23_col88 = cuda_evaluator.next_trace_mask();
    m31 output_limb_24_col89 = cuda_evaluator.next_trace_mask();
    m31 output_limb_25_col90 = cuda_evaluator.next_trace_mask();
    m31 output_limb_26_col91 = cuda_evaluator.next_trace_mask();
    m31 output_limb_27_col92 = cuda_evaluator.next_trace_mask();
    m31 output_limb_28_col93 = cuda_evaluator.next_trace_mask();
    m31 output_limb_29_col94 = cuda_evaluator.next_trace_mask();
    m31 output_limb_30_col95 = cuda_evaluator.next_trace_mask();
    m31 output_limb_31_col96 = cuda_evaluator.next_trace_mask();
    m31 output_limb_32_col97 = cuda_evaluator.next_trace_mask();
    m31 output_limb_33_col98 = cuda_evaluator.next_trace_mask();
    m31 output_limb_34_col99 = cuda_evaluator.next_trace_mask();
    m31 output_limb_35_col100 = cuda_evaluator.next_trace_mask();
    m31 output_limb_36_col101 = cuda_evaluator.next_trace_mask();
    m31 output_limb_37_col102 = cuda_evaluator.next_trace_mask();
    m31 output_limb_38_col103 = cuda_evaluator.next_trace_mask();
    m31 output_limb_39_col104 = cuda_evaluator.next_trace_mask();
    m31 output_limb_40_col105 = cuda_evaluator.next_trace_mask();
    m31 output_limb_41_col106 = cuda_evaluator.next_trace_mask();
    m31 output_limb_42_col107 = cuda_evaluator.next_trace_mask();
    m31 output_limb_43_col108 = cuda_evaluator.next_trace_mask();
    m31 output_limb_44_col109 = cuda_evaluator.next_trace_mask();
    m31 output_limb_45_col110 = cuda_evaluator.next_trace_mask();
    m31 output_limb_46_col111 = cuda_evaluator.next_trace_mask();
    m31 output_limb_47_col112 = cuda_evaluator.next_trace_mask();
    m31 output_limb_48_col113 = cuda_evaluator.next_trace_mask();
    m31 output_limb_49_col114 = cuda_evaluator.next_trace_mask();
    m31 output_limb_50_col115 = cuda_evaluator.next_trace_mask();
    m31 output_limb_51_col116 = cuda_evaluator.next_trace_mask();
    m31 output_limb_52_col117 = cuda_evaluator.next_trace_mask();
    m31 output_limb_53_col118 = cuda_evaluator.next_trace_mask();
    m31 output_limb_54_col119 = cuda_evaluator.next_trace_mask();
    m31 output_limb_55_col120 = cuda_evaluator.next_trace_mask();
    m31 output_limb_56_col121 = cuda_evaluator.next_trace_mask();
    m31 output_limb_57_col122 = cuda_evaluator.next_trace_mask();
    m31 output_limb_58_col123 = cuda_evaluator.next_trace_mask();
    m31 output_limb_59_col124 = cuda_evaluator.next_trace_mask();
    m31 output_limb_60_col125 = cuda_evaluator.next_trace_mask();
    m31 output_limb_61_col126 = cuda_evaluator.next_trace_mask();
    m31 output_limb_62_col127 = cuda_evaluator.next_trace_mask();
    m31 output_limb_63_col128 = cuda_evaluator.next_trace_mask();
    m31 output_limb_64_col129 = cuda_evaluator.next_trace_mask();
    m31 output_limb_65_col130 = cuda_evaluator.next_trace_mask();
    m31 output_limb_66_col131 = cuda_evaluator.next_trace_mask();
    m31 output_limb_67_col132 = cuda_evaluator.next_trace_mask();
    m31 output_limb_68_col133 = cuda_evaluator.next_trace_mask();
    m31 output_limb_69_col134 = cuda_evaluator.next_trace_mask();

    // col135-col204: PartialEcMulWindowBits18 output limbs for chain 1 (70 limbs)
    m31 output2_limb_0_col135 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_1_col136 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_2_col137 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_3_col138 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_4_col139 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_5_col140 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_6_col141 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_7_col142 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_8_col143 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_9_col144 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_10_col145 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_11_col146 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_12_col147 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_13_col148 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_14_col149 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_15_col150 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_16_col151 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_17_col152 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_18_col153 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_19_col154 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_20_col155 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_21_col156 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_22_col157 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_23_col158 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_24_col159 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_25_col160 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_26_col161 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_27_col162 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_28_col163 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_29_col164 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_30_col165 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_31_col166 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_32_col167 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_33_col168 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_34_col169 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_35_col170 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_36_col171 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_37_col172 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_38_col173 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_39_col174 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_40_col175 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_41_col176 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_42_col177 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_43_col178 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_44_col179 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_45_col180 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_46_col181 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_47_col182 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_48_col183 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_49_col184 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_50_col185 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_51_col186 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_52_col187 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_53_col188 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_54_col189 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_55_col190 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_56_col191 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_57_col192 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_58_col193 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_59_col194 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_60_col195 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_61_col196 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_62_col197 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_63_col198 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_64_col199 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_65_col200 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_66_col201 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_67_col202 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_68_col203 = cuda_evaluator.next_trace_mask();
    m31 output2_limb_69_col204 = cuda_evaluator.next_trace_mask();

    // col205: multiplicity
    m31 multiplicity_0 = cuda_evaluator.next_trace_mask();

    // ===================== 1. ReadPositiveKnownIdNumBits252 #1 =====================
    // Inlined: MemoryIdToBig lookup for input_limb_0_col0 with value limbs col3-col30
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_0_col0,
            value_limb_0_col3, value_limb_1_col4, value_limb_2_col5, value_limb_3_col6,
            value_limb_4_col7, value_limb_5_col8, value_limb_6_col9, value_limb_7_col10,
            value_limb_8_col11, value_limb_9_col12, value_limb_10_col13, value_limb_11_col14,
            value_limb_12_col15, value_limb_13_col16, value_limb_14_col17, value_limb_15_col18,
            value_limb_16_col19, value_limb_17_col20, value_limb_18_col21, value_limb_19_col22,
            value_limb_20_col23, value_limb_21_col24, value_limb_22_col25, value_limb_23_col26,
            value_limb_24_col27, value_limb_25_col28, value_limb_26_col29, value_limb_27_col30
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // ===================== 2. ReadPositiveKnownIdNumBits252 #2 =====================
    // Inlined: MemoryIdToBig lookup for input_limb_1_col1 with value limbs col31-col58
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_1_col1,
            value_limb_0_col31, value_limb_1_col32, value_limb_2_col33, value_limb_3_col34,
            value_limb_4_col35, value_limb_5_col36, value_limb_6_col37, value_limb_7_col38,
            value_limb_8_col39, value_limb_9_col40, value_limb_10_col41, value_limb_11_col42,
            value_limb_12_col43, value_limb_13_col44, value_limb_14_col45, value_limb_15_col46,
            value_limb_16_col47, value_limb_17_col48, value_limb_18_col49, value_limb_19_col50,
            value_limb_20_col51, value_limb_21_col52, value_limb_22_col53, value_limb_23_col54,
            value_limb_24_col55, value_limb_25_col56, value_limb_26_col57, value_limb_27_col58
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // ===================== 3. VerifyReduced252 #1 =====================
    // Verifies value limbs col3-col30 are properly reduced
    verify_reduced_252_evaluate(
        value_limb_0_col3, value_limb_1_col4, value_limb_2_col5, value_limb_3_col6,
        value_limb_4_col7, value_limb_5_col8, value_limb_6_col9, value_limb_7_col10,
        value_limb_8_col11, value_limb_9_col12, value_limb_10_col13, value_limb_11_col14,
        value_limb_12_col15, value_limb_13_col16, value_limb_14_col17, value_limb_15_col18,
        value_limb_16_col19, value_limb_17_col20, value_limb_18_col21, value_limb_19_col22,
        value_limb_20_col23, value_limb_21_col24, value_limb_22_col25, value_limb_23_col26,
        value_limb_24_col27, value_limb_25_col28, value_limb_26_col29, value_limb_27_col30,
        ms_limb_is_max_col59,
        ms_and_mid_limbs_are_max_col60,
        rc_input_col61,
        agg_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // ===================== 4. VerifyReduced252 #2 =====================
    // Verifies value limbs col31-col58 are properly reduced
    verify_reduced_252_evaluate(
        value_limb_0_col31, value_limb_1_col32, value_limb_2_col33, value_limb_3_col34,
        value_limb_4_col35, value_limb_5_col36, value_limb_6_col37, value_limb_7_col38,
        value_limb_8_col39, value_limb_9_col40, value_limb_10_col41, value_limb_11_col42,
        value_limb_12_col43, value_limb_13_col44, value_limb_14_col45, value_limb_15_col46,
        value_limb_16_col47, value_limb_17_col48, value_limb_18_col49, value_limb_19_col50,
        value_limb_20_col51, value_limb_21_col52, value_limb_22_col53, value_limb_23_col54,
        value_limb_24_col55, value_limb_25_col56, value_limb_26_col57, value_limb_27_col58,
        ms_limb_is_max_col62,
        ms_and_mid_limbs_are_max_col63,
        rc_input_col64,
        agg_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // ===================== Compute intermediates =====================
    // chain_id_0 = seq * 2
    m31 chain_id_0 = mul(seq, M31_2);
    // chain_id_1 = seq * 2 + 1
    m31 chain_id_1 = add(chain_id_0, M31_1);

    // ===================== 5. PartialEcMulWindowBits18 Input (chain 0) =====================
    // Multiplicity = -1, 73 values
    // [RELATION_ID, chain_id_0, 0, packed_pairs(14), constants(56)]
    {
        m31 values[73];
        values[0] = PARTIAL_EC_MUL_RELATION_ID;
        values[1] = chain_id_0;
        values[2] = m31(0);
        // 14 packed pairs from value limbs col3-col30
        values[3] = add(value_limb_0_col3, mul(value_limb_1_col4, M31_512));
        values[4] = add(value_limb_2_col5, mul(value_limb_3_col6, M31_512));
        values[5] = add(value_limb_4_col7, mul(value_limb_5_col8, M31_512));
        values[6] = add(value_limb_6_col9, mul(value_limb_7_col10, M31_512));
        values[7] = add(value_limb_8_col11, mul(value_limb_9_col12, M31_512));
        values[8] = add(value_limb_10_col13, mul(value_limb_11_col14, M31_512));
        values[9] = add(value_limb_12_col15, mul(value_limb_13_col16, M31_512));
        values[10] = add(value_limb_14_col17, mul(value_limb_15_col18, M31_512));
        values[11] = add(value_limb_16_col19, mul(value_limb_17_col20, M31_512));
        values[12] = add(value_limb_18_col21, mul(value_limb_19_col22, M31_512));
        values[13] = add(value_limb_20_col23, mul(value_limb_21_col24, M31_512));
        values[14] = add(value_limb_22_col25, mul(value_limb_23_col26, M31_512));
        values[15] = add(value_limb_24_col27, mul(value_limb_25_col28, M31_512));
        values[16] = add(value_limb_26_col29, mul(value_limb_27_col30, M31_512));
        // 56 constants from Rust source (lines 510-565)
        values[17] = m31(510);
        values[18] = m31(315);
        values[19] = m31(208);
        values[20] = m31(480);
        values[21] = m31(418);
        values[22] = m31(115);
        values[23] = m31(155);
        values[24] = m31(54);
        values[25] = m31(162);
        values[26] = m31(449);
        values[27] = m31(428);
        values[28] = m31(466);
        values[29] = m31(484);
        values[30] = m31(169);
        values[31] = m31(497);
        values[32] = m31(373);
        values[33] = m31(98);
        values[34] = m31(64);
        values[35] = m31(464);
        values[36] = m31(498);
        values[37] = m31(124);
        values[38] = m31(68);
        values[39] = m31(379);
        values[40] = m31(140);
        values[41] = m31(26);
        values[42] = m31(22);
        values[43] = m31(135);
        values[44] = m31(202);
        values[45] = m31(156);
        values[46] = m31(120);
        values[47] = m31(213);
        values[48] = m31(389);
        values[49] = m31(377);
        values[50] = m31(20);
        values[51] = m31(325);
        values[52] = m31(303);
        values[53] = m31(473);
        values[54] = m31(334);
        values[55] = m31(223);
        values[56] = m31(160);
        values[57] = m31(225);
        values[58] = m31(297);
        values[59] = m31(101);
        values[60] = m31(420);
        values[61] = m31(377);
        values[62] = m31(72);
        values[63] = m31(191);
        values[64] = m31(49);
        values[65] = m31(314);
        values[66] = m31(27);
        values[67] = m31(199);
        values[68] = m31(222);
        values[69] = m31(79);
        values[70] = m31(97);
        values[71] = m31(108);
        values[72] = m31(141);
        cuda_evaluator.add_to_relation<73>(agg_eval->common_lookup_elements, qm31{neg(m31(1)), m31(0)}, values);
    }

    // ===================== 6. PartialEcMulWindowBits18 Output (chain 0) =====================
    // Multiplicity = +1, 73 values
    // [RELATION_ID, chain_id_0, 14, output_limbs_0..69 from col65-col134]
    {
        m31 values[73];
        values[0] = PARTIAL_EC_MUL_RELATION_ID;
        values[1] = chain_id_0;
        values[2] = m31(14);
        values[3] = output_limb_0_col65;
        values[4] = output_limb_1_col66;
        values[5] = output_limb_2_col67;
        values[6] = output_limb_3_col68;
        values[7] = output_limb_4_col69;
        values[8] = output_limb_5_col70;
        values[9] = output_limb_6_col71;
        values[10] = output_limb_7_col72;
        values[11] = output_limb_8_col73;
        values[12] = output_limb_9_col74;
        values[13] = output_limb_10_col75;
        values[14] = output_limb_11_col76;
        values[15] = output_limb_12_col77;
        values[16] = output_limb_13_col78;
        values[17] = output_limb_14_col79;
        values[18] = output_limb_15_col80;
        values[19] = output_limb_16_col81;
        values[20] = output_limb_17_col82;
        values[21] = output_limb_18_col83;
        values[22] = output_limb_19_col84;
        values[23] = output_limb_20_col85;
        values[24] = output_limb_21_col86;
        values[25] = output_limb_22_col87;
        values[26] = output_limb_23_col88;
        values[27] = output_limb_24_col89;
        values[28] = output_limb_25_col90;
        values[29] = output_limb_26_col91;
        values[30] = output_limb_27_col92;
        values[31] = output_limb_28_col93;
        values[32] = output_limb_29_col94;
        values[33] = output_limb_30_col95;
        values[34] = output_limb_31_col96;
        values[35] = output_limb_32_col97;
        values[36] = output_limb_33_col98;
        values[37] = output_limb_34_col99;
        values[38] = output_limb_35_col100;
        values[39] = output_limb_36_col101;
        values[40] = output_limb_37_col102;
        values[41] = output_limb_38_col103;
        values[42] = output_limb_39_col104;
        values[43] = output_limb_40_col105;
        values[44] = output_limb_41_col106;
        values[45] = output_limb_42_col107;
        values[46] = output_limb_43_col108;
        values[47] = output_limb_44_col109;
        values[48] = output_limb_45_col110;
        values[49] = output_limb_46_col111;
        values[50] = output_limb_47_col112;
        values[51] = output_limb_48_col113;
        values[52] = output_limb_49_col114;
        values[53] = output_limb_50_col115;
        values[54] = output_limb_51_col116;
        values[55] = output_limb_52_col117;
        values[56] = output_limb_53_col118;
        values[57] = output_limb_54_col119;
        values[58] = output_limb_55_col120;
        values[59] = output_limb_56_col121;
        values[60] = output_limb_57_col122;
        values[61] = output_limb_58_col123;
        values[62] = output_limb_59_col124;
        values[63] = output_limb_60_col125;
        values[64] = output_limb_61_col126;
        values[65] = output_limb_62_col127;
        values[66] = output_limb_63_col128;
        values[67] = output_limb_64_col129;
        values[68] = output_limb_65_col130;
        values[69] = output_limb_66_col131;
        values[70] = output_limb_67_col132;
        values[71] = output_limb_68_col133;
        values[72] = output_limb_69_col134;
        cuda_evaluator.add_to_relation<73>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // ===================== 7. PartialEcMulWindowBits18 Input (chain 1) =====================
    // Multiplicity = -1, 73 values
    // [RELATION_ID, chain_id_1, 14,
    //   packed_pairs(14) from value limbs col31-col58,
    //   output_limbs 14..69 from first chain output (col79-col134)]
    {
        m31 values[73];
        values[0] = PARTIAL_EC_MUL_RELATION_ID;
        values[1] = chain_id_1;
        values[2] = m31(14);
        // 14 packed pairs from value limbs col31-col58
        values[3] = add(value_limb_0_col31, mul(value_limb_1_col32, M31_512));
        values[4] = add(value_limb_2_col33, mul(value_limb_3_col34, M31_512));
        values[5] = add(value_limb_4_col35, mul(value_limb_5_col36, M31_512));
        values[6] = add(value_limb_6_col37, mul(value_limb_7_col38, M31_512));
        values[7] = add(value_limb_8_col39, mul(value_limb_9_col40, M31_512));
        values[8] = add(value_limb_10_col41, mul(value_limb_11_col42, M31_512));
        values[9] = add(value_limb_12_col43, mul(value_limb_13_col44, M31_512));
        values[10] = add(value_limb_14_col45, mul(value_limb_15_col46, M31_512));
        values[11] = add(value_limb_16_col47, mul(value_limb_17_col48, M31_512));
        values[12] = add(value_limb_18_col49, mul(value_limb_19_col50, M31_512));
        values[13] = add(value_limb_20_col51, mul(value_limb_21_col52, M31_512));
        values[14] = add(value_limb_22_col53, mul(value_limb_23_col54, M31_512));
        values[15] = add(value_limb_24_col55, mul(value_limb_25_col56, M31_512));
        values[16] = add(value_limb_26_col57, mul(value_limb_27_col58, M31_512));
        // 56 values: output_limbs 14..69 from first chain output (col79-col134)
        values[17] = output_limb_14_col79;
        values[18] = output_limb_15_col80;
        values[19] = output_limb_16_col81;
        values[20] = output_limb_17_col82;
        values[21] = output_limb_18_col83;
        values[22] = output_limb_19_col84;
        values[23] = output_limb_20_col85;
        values[24] = output_limb_21_col86;
        values[25] = output_limb_22_col87;
        values[26] = output_limb_23_col88;
        values[27] = output_limb_24_col89;
        values[28] = output_limb_25_col90;
        values[29] = output_limb_26_col91;
        values[30] = output_limb_27_col92;
        values[31] = output_limb_28_col93;
        values[32] = output_limb_29_col94;
        values[33] = output_limb_30_col95;
        values[34] = output_limb_31_col96;
        values[35] = output_limb_32_col97;
        values[36] = output_limb_33_col98;
        values[37] = output_limb_34_col99;
        values[38] = output_limb_35_col100;
        values[39] = output_limb_36_col101;
        values[40] = output_limb_37_col102;
        values[41] = output_limb_38_col103;
        values[42] = output_limb_39_col104;
        values[43] = output_limb_40_col105;
        values[44] = output_limb_41_col106;
        values[45] = output_limb_42_col107;
        values[46] = output_limb_43_col108;
        values[47] = output_limb_44_col109;
        values[48] = output_limb_45_col110;
        values[49] = output_limb_46_col111;
        values[50] = output_limb_47_col112;
        values[51] = output_limb_48_col113;
        values[52] = output_limb_49_col114;
        values[53] = output_limb_50_col115;
        values[54] = output_limb_51_col116;
        values[55] = output_limb_52_col117;
        values[56] = output_limb_53_col118;
        values[57] = output_limb_54_col119;
        values[58] = output_limb_55_col120;
        values[59] = output_limb_56_col121;
        values[60] = output_limb_57_col122;
        values[61] = output_limb_58_col123;
        values[62] = output_limb_59_col124;
        values[63] = output_limb_60_col125;
        values[64] = output_limb_61_col126;
        values[65] = output_limb_62_col127;
        values[66] = output_limb_63_col128;
        values[67] = output_limb_64_col129;
        values[68] = output_limb_65_col130;
        values[69] = output_limb_66_col131;
        values[70] = output_limb_67_col132;
        values[71] = output_limb_68_col133;
        values[72] = output_limb_69_col134;
        cuda_evaluator.add_to_relation<73>(agg_eval->common_lookup_elements, qm31{neg(m31(1)), m31(0)}, values);
    }

    // ===================== 8. PartialEcMulWindowBits18 Output (chain 1) =====================
    // Multiplicity = +1, 73 values
    // [RELATION_ID, chain_id_1, 28, output2_limbs_0..69 from col135-col204]
    {
        m31 values[73];
        values[0] = PARTIAL_EC_MUL_RELATION_ID;
        values[1] = chain_id_1;
        values[2] = m31(28);
        values[3] = output2_limb_0_col135;
        values[4] = output2_limb_1_col136;
        values[5] = output2_limb_2_col137;
        values[6] = output2_limb_3_col138;
        values[7] = output2_limb_4_col139;
        values[8] = output2_limb_5_col140;
        values[9] = output2_limb_6_col141;
        values[10] = output2_limb_7_col142;
        values[11] = output2_limb_8_col143;
        values[12] = output2_limb_9_col144;
        values[13] = output2_limb_10_col145;
        values[14] = output2_limb_11_col146;
        values[15] = output2_limb_12_col147;
        values[16] = output2_limb_13_col148;
        values[17] = output2_limb_14_col149;
        values[18] = output2_limb_15_col150;
        values[19] = output2_limb_16_col151;
        values[20] = output2_limb_17_col152;
        values[21] = output2_limb_18_col153;
        values[22] = output2_limb_19_col154;
        values[23] = output2_limb_20_col155;
        values[24] = output2_limb_21_col156;
        values[25] = output2_limb_22_col157;
        values[26] = output2_limb_23_col158;
        values[27] = output2_limb_24_col159;
        values[28] = output2_limb_25_col160;
        values[29] = output2_limb_26_col161;
        values[30] = output2_limb_27_col162;
        values[31] = output2_limb_28_col163;
        values[32] = output2_limb_29_col164;
        values[33] = output2_limb_30_col165;
        values[34] = output2_limb_31_col166;
        values[35] = output2_limb_32_col167;
        values[36] = output2_limb_33_col168;
        values[37] = output2_limb_34_col169;
        values[38] = output2_limb_35_col170;
        values[39] = output2_limb_36_col171;
        values[40] = output2_limb_37_col172;
        values[41] = output2_limb_38_col173;
        values[42] = output2_limb_39_col174;
        values[43] = output2_limb_40_col175;
        values[44] = output2_limb_41_col176;
        values[45] = output2_limb_42_col177;
        values[46] = output2_limb_43_col178;
        values[47] = output2_limb_44_col179;
        values[48] = output2_limb_45_col180;
        values[49] = output2_limb_46_col181;
        values[50] = output2_limb_47_col182;
        values[51] = output2_limb_48_col183;
        values[52] = output2_limb_49_col184;
        values[53] = output2_limb_50_col185;
        values[54] = output2_limb_51_col186;
        values[55] = output2_limb_52_col187;
        values[56] = output2_limb_53_col188;
        values[57] = output2_limb_54_col189;
        values[58] = output2_limb_55_col190;
        values[59] = output2_limb_56_col191;
        values[60] = output2_limb_57_col192;
        values[61] = output2_limb_58_col193;
        values[62] = output2_limb_59_col194;
        values[63] = output2_limb_60_col195;
        values[64] = output2_limb_61_col196;
        values[65] = output2_limb_62_col197;
        values[66] = output2_limb_63_col198;
        values[67] = output2_limb_64_col199;
        values[68] = output2_limb_65_col200;
        values[69] = output2_limb_66_col201;
        values[70] = output2_limb_67_col202;
        values[71] = output2_limb_68_col203;
        values[72] = output2_limb_69_col204;
        cuda_evaluator.add_to_relation<73>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // ===================== 9. MemoryIdToBig output lookup =====================
    // Multiplicity = +1, 30 values
    // [MEMORY_ID_TO_BIG_RELATION_ID, input_limb_2_col2, output2_limb_14..41 from col149-col176]
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_2_col2,
            output2_limb_14_col149, output2_limb_15_col150, output2_limb_16_col151, output2_limb_17_col152,
            output2_limb_18_col153, output2_limb_19_col154, output2_limb_20_col155, output2_limb_21_col156,
            output2_limb_22_col157, output2_limb_23_col158, output2_limb_24_col159, output2_limb_25_col160,
            output2_limb_26_col161, output2_limb_27_col162, output2_limb_28_col163, output2_limb_29_col164,
            output2_limb_30_col165, output2_limb_31_col166, output2_limb_32_col167, output2_limb_33_col168,
            output2_limb_34_col169, output2_limb_35_col170, output2_limb_36_col171, output2_limb_37_col172,
            output2_limb_38_col173, output2_limb_39_col174, output2_limb_40_col175, output2_limb_41_col176
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // ===================== 10. PedersenAggregatorWindowBits18 PROVIDE =====================
    // Multiplicity = -multiplicity_0 (PROVIDE = negative)
    {
        m31 values[4] = {
            PEDERSEN_AGGREGATOR_WINDOW_BITS_18_RELATION_ID,
            input_limb_0_col0,
            input_limb_1_col1,
            input_limb_2_col2
        };
        m31 neg_mult = neg(multiplicity_0);
        qm31 multiplicity_ext = {{neg_mult, 0}, {0, 0}};
        cuda_evaluator.add_to_relation<4>(
            agg_eval->common_lookup_elements,
            multiplicity_ext,
            values
        );
    }

    // Store results
    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// =====================================================================
// Host Wrapper Function
// =====================================================================
extern "C"
void evaluate_pedersen_aggregator_window_bits_18(
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

    PedersenAggregator_WB18_Eval *device_agg_eval = cuda_malloc<PedersenAggregator_WB18_Eval>(1);
    cuda_mem_copy_host_to_device<PedersenAggregator_WB18_Eval>((PedersenAggregator_WB18_Eval*)eval, device_agg_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);

    int block_dim = eval_domain_size < PEDERSEN_AGGREGATOR_WB18_THREAD_COUNT_MAX ? eval_domain_size : PEDERSEN_AGGREGATOR_WB18_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_pedersen_aggregator_window_bits_18_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_agg_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_pedersen_aggregator_window_bits_18_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_agg_eval,
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
        g_should_accumulate_host
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_agg_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
