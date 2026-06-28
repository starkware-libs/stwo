/*
============================================
PoseidonAggregator CUDA Evaluator
============================================

Component: PoseidonAggregator (342 trace columns, 1 preprocessed column)
Translated from: cairo-air/src/components/poseidon_aggregator.rs

Flow:
1. Read preprocessed column (seq) from trace0
2. Read 342 trace columns from trace1
3. 3x ReadPositiveKnownIdNumBits252 (inlined: just MemoryIdToBig lookup each)
4. Compute 27 packed values (9 per state word): packed = val_0 + val_1 * 512 + val_2 * 262144
5. PoseidonHadesPermutation with 30 packed inputs and 197 intermediate columns
6. 3x Felt252UnpackFrom27 (compute intermediate unpacked values)
7. 3x MemoryIdToBig lookups for unpacked output states
8. 1x PoseidonAggregator PROVIDE relation (negative multiplicity)

============================================
*/

#include <cstdio>
#include <vector>

#include "evaluate_poseidon_aggregator.cuh"
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_common.cuh"
#include "relations.cuh"
#include "../evaluate_poseidon_hades_permutation.cuh"
#include "../evaluate_felt_252_unpack_from_27.cuh"

#define POSEIDON_AGGREGATOR_THREAD_COUNT_MAX 256

// =====================================================================
// Pre-Kernel: Read trace columns, evaluate constraints, build logup fractions
// =====================================================================
template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_poseidon_aggregator_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PoseidonAggregator_Eval *agg_eval,
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
    const m31 M31_262144 = m31(262144);

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

    // Evaluator for base trace (trace1) -- reads 342 trace columns + builds logup
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

    // ===================== Read all 342 trace columns =====================
    // col0-col5: input limbs (memory IDs for 3 input + 3 output state words)
    m31 input_limb_0_col0 = cuda_evaluator.next_trace_mask();
    m31 input_limb_1_col1 = cuda_evaluator.next_trace_mask();
    m31 input_limb_2_col2 = cuda_evaluator.next_trace_mask();
    m31 input_limb_3_col3 = cuda_evaluator.next_trace_mask();
    m31 input_limb_4_col4 = cuda_evaluator.next_trace_mask();
    m31 input_limb_5_col5 = cuda_evaluator.next_trace_mask();

    // col6-col33: value limbs for first state word (28 limbs)
    m31 value_limb_0_col6 = cuda_evaluator.next_trace_mask();
    m31 value_limb_1_col7 = cuda_evaluator.next_trace_mask();
    m31 value_limb_2_col8 = cuda_evaluator.next_trace_mask();
    m31 value_limb_3_col9 = cuda_evaluator.next_trace_mask();
    m31 value_limb_4_col10 = cuda_evaluator.next_trace_mask();
    m31 value_limb_5_col11 = cuda_evaluator.next_trace_mask();
    m31 value_limb_6_col12 = cuda_evaluator.next_trace_mask();
    m31 value_limb_7_col13 = cuda_evaluator.next_trace_mask();
    m31 value_limb_8_col14 = cuda_evaluator.next_trace_mask();
    m31 value_limb_9_col15 = cuda_evaluator.next_trace_mask();
    m31 value_limb_10_col16 = cuda_evaluator.next_trace_mask();
    m31 value_limb_11_col17 = cuda_evaluator.next_trace_mask();
    m31 value_limb_12_col18 = cuda_evaluator.next_trace_mask();
    m31 value_limb_13_col19 = cuda_evaluator.next_trace_mask();
    m31 value_limb_14_col20 = cuda_evaluator.next_trace_mask();
    m31 value_limb_15_col21 = cuda_evaluator.next_trace_mask();
    m31 value_limb_16_col22 = cuda_evaluator.next_trace_mask();
    m31 value_limb_17_col23 = cuda_evaluator.next_trace_mask();
    m31 value_limb_18_col24 = cuda_evaluator.next_trace_mask();
    m31 value_limb_19_col25 = cuda_evaluator.next_trace_mask();
    m31 value_limb_20_col26 = cuda_evaluator.next_trace_mask();
    m31 value_limb_21_col27 = cuda_evaluator.next_trace_mask();
    m31 value_limb_22_col28 = cuda_evaluator.next_trace_mask();
    m31 value_limb_23_col29 = cuda_evaluator.next_trace_mask();
    m31 value_limb_24_col30 = cuda_evaluator.next_trace_mask();
    m31 value_limb_25_col31 = cuda_evaluator.next_trace_mask();
    m31 value_limb_26_col32 = cuda_evaluator.next_trace_mask();
    m31 value_limb_27_col33 = cuda_evaluator.next_trace_mask();

    // col34-col61: value limbs for second state word (28 limbs)
    m31 value_limb_0_col34 = cuda_evaluator.next_trace_mask();
    m31 value_limb_1_col35 = cuda_evaluator.next_trace_mask();
    m31 value_limb_2_col36 = cuda_evaluator.next_trace_mask();
    m31 value_limb_3_col37 = cuda_evaluator.next_trace_mask();
    m31 value_limb_4_col38 = cuda_evaluator.next_trace_mask();
    m31 value_limb_5_col39 = cuda_evaluator.next_trace_mask();
    m31 value_limb_6_col40 = cuda_evaluator.next_trace_mask();
    m31 value_limb_7_col41 = cuda_evaluator.next_trace_mask();
    m31 value_limb_8_col42 = cuda_evaluator.next_trace_mask();
    m31 value_limb_9_col43 = cuda_evaluator.next_trace_mask();
    m31 value_limb_10_col44 = cuda_evaluator.next_trace_mask();
    m31 value_limb_11_col45 = cuda_evaluator.next_trace_mask();
    m31 value_limb_12_col46 = cuda_evaluator.next_trace_mask();
    m31 value_limb_13_col47 = cuda_evaluator.next_trace_mask();
    m31 value_limb_14_col48 = cuda_evaluator.next_trace_mask();
    m31 value_limb_15_col49 = cuda_evaluator.next_trace_mask();
    m31 value_limb_16_col50 = cuda_evaluator.next_trace_mask();
    m31 value_limb_17_col51 = cuda_evaluator.next_trace_mask();
    m31 value_limb_18_col52 = cuda_evaluator.next_trace_mask();
    m31 value_limb_19_col53 = cuda_evaluator.next_trace_mask();
    m31 value_limb_20_col54 = cuda_evaluator.next_trace_mask();
    m31 value_limb_21_col55 = cuda_evaluator.next_trace_mask();
    m31 value_limb_22_col56 = cuda_evaluator.next_trace_mask();
    m31 value_limb_23_col57 = cuda_evaluator.next_trace_mask();
    m31 value_limb_24_col58 = cuda_evaluator.next_trace_mask();
    m31 value_limb_25_col59 = cuda_evaluator.next_trace_mask();
    m31 value_limb_26_col60 = cuda_evaluator.next_trace_mask();
    m31 value_limb_27_col61 = cuda_evaluator.next_trace_mask();

    // col62-col89: value limbs for third state word (28 limbs)
    m31 value_limb_0_col62 = cuda_evaluator.next_trace_mask();
    m31 value_limb_1_col63 = cuda_evaluator.next_trace_mask();
    m31 value_limb_2_col64 = cuda_evaluator.next_trace_mask();
    m31 value_limb_3_col65 = cuda_evaluator.next_trace_mask();
    m31 value_limb_4_col66 = cuda_evaluator.next_trace_mask();
    m31 value_limb_5_col67 = cuda_evaluator.next_trace_mask();
    m31 value_limb_6_col68 = cuda_evaluator.next_trace_mask();
    m31 value_limb_7_col69 = cuda_evaluator.next_trace_mask();
    m31 value_limb_8_col70 = cuda_evaluator.next_trace_mask();
    m31 value_limb_9_col71 = cuda_evaluator.next_trace_mask();
    m31 value_limb_10_col72 = cuda_evaluator.next_trace_mask();
    m31 value_limb_11_col73 = cuda_evaluator.next_trace_mask();
    m31 value_limb_12_col74 = cuda_evaluator.next_trace_mask();
    m31 value_limb_13_col75 = cuda_evaluator.next_trace_mask();
    m31 value_limb_14_col76 = cuda_evaluator.next_trace_mask();
    m31 value_limb_15_col77 = cuda_evaluator.next_trace_mask();
    m31 value_limb_16_col78 = cuda_evaluator.next_trace_mask();
    m31 value_limb_17_col79 = cuda_evaluator.next_trace_mask();
    m31 value_limb_18_col80 = cuda_evaluator.next_trace_mask();
    m31 value_limb_19_col81 = cuda_evaluator.next_trace_mask();
    m31 value_limb_20_col82 = cuda_evaluator.next_trace_mask();
    m31 value_limb_21_col83 = cuda_evaluator.next_trace_mask();
    m31 value_limb_22_col84 = cuda_evaluator.next_trace_mask();
    m31 value_limb_23_col85 = cuda_evaluator.next_trace_mask();
    m31 value_limb_24_col86 = cuda_evaluator.next_trace_mask();
    m31 value_limb_25_col87 = cuda_evaluator.next_trace_mask();
    m31 value_limb_26_col88 = cuda_evaluator.next_trace_mask();
    m31 value_limb_27_col89 = cuda_evaluator.next_trace_mask();

    // col90-col286: PoseidonHadesPermutation columns (197 columns)
    m31 combination_limb_0_col90 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col91 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col92 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col93 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col94 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col95 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col96 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col97 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col98 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col99 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col100 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col101 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col102 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col103 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col104 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col105 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col106 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col107 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col108 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col109 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col110 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col111 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col112 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col113 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col114 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col115 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col116 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col117 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col118 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col119 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col120 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col121 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col122 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_0_col123 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_1_col124 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_2_col125 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_3_col126 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_4_col127 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_5_col128 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_6_col129 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_7_col130 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_8_col131 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_9_col132 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_10_col133 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_11_col134 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_12_col135 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_13_col136 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_14_col137 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_15_col138 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_16_col139 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_17_col140 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_18_col141 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_19_col142 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_20_col143 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_21_col144 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_22_col145 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_23_col146 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_24_col147 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_25_col148 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_26_col149 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_27_col150 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_28_col151 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_29_col152 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_0_col153 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_1_col154 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_2_col155 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_3_col156 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_4_col157 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_5_col158 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_6_col159 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_7_col160 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_8_col161 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_9_col162 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col163 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col164 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col165 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col166 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col167 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col168 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col169 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col170 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col171 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col172 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col173 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_0_col174 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_1_col175 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_2_col176 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_3_col177 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_4_col178 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_5_col179 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_6_col180 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_7_col181 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_8_col182 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_9_col183 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col184 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col185 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col186 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col187 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col188 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col189 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col190 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col191 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col192 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col193 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col194 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_0_col195 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_1_col196 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_2_col197 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_3_col198 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_4_col199 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_5_col200 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_6_col201 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_7_col202 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_8_col203 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_9_col204 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_10_col205 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_11_col206 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_12_col207 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_13_col208 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_14_col209 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_15_col210 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_16_col211 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_17_col212 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_18_col213 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_19_col214 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_20_col215 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_21_col216 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_22_col217 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_23_col218 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_24_col219 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_25_col220 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_26_col221 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_27_col222 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_28_col223 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_29_col224 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_30_col225 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_31_col226 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_32_col227 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_33_col228 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_34_col229 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_35_col230 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_36_col231 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_37_col232 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_38_col233 = cuda_evaluator.next_trace_mask();
    m31 poseidon_3_partial_rounds_chain_output_limb_39_col234 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col235 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col236 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col237 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col238 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col239 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col240 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col241 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col242 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col243 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col244 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col245 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col246 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col247 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col248 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col249 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col250 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col251 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col252 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col253 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col254 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col255 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col256 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_0_col257 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_1_col258 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_2_col259 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_3_col260 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_4_col261 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_5_col262 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_6_col263 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_7_col264 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_8_col265 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_9_col266 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_10_col267 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_11_col268 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_12_col269 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_13_col270 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_14_col271 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_15_col272 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_16_col273 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_17_col274 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_18_col275 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_19_col276 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_20_col277 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_21_col278 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_22_col279 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_23_col280 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_24_col281 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_25_col282 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_26_col283 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_27_col284 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_28_col285 = cuda_evaluator.next_trace_mask();
    m31 poseidon_full_round_chain_output_limb_29_col286 = cuda_evaluator.next_trace_mask();

    // col287-col304: Felt252UnpackFrom27 #1 unpacked limbs (18 columns)
    m31 unpacked_limb_0_col287 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_1_col288 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_3_col289 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_4_col290 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_6_col291 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_7_col292 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_9_col293 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_10_col294 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_12_col295 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_13_col296 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_15_col297 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_16_col298 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_18_col299 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_19_col300 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_21_col301 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_22_col302 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_24_col303 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_25_col304 = cuda_evaluator.next_trace_mask();

    // col305-col322: Felt252UnpackFrom27 #2 unpacked limbs (18 columns)
    m31 unpacked_limb_0_col305 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_1_col306 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_3_col307 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_4_col308 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_6_col309 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_7_col310 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_9_col311 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_10_col312 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_12_col313 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_13_col314 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_15_col315 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_16_col316 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_18_col317 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_19_col318 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_21_col319 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_22_col320 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_24_col321 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_25_col322 = cuda_evaluator.next_trace_mask();

    // col323-col340: Felt252UnpackFrom27 #3 unpacked limbs (18 columns)
    m31 unpacked_limb_0_col323 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_1_col324 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_3_col325 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_4_col326 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_6_col327 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_7_col328 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_9_col329 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_10_col330 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_12_col331 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_13_col332 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_15_col333 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_16_col334 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_18_col335 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_19_col336 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_21_col337 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_22_col338 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_24_col339 = cuda_evaluator.next_trace_mask();
    m31 unpacked_limb_25_col340 = cuda_evaluator.next_trace_mask();

    // col341: multiplicity
    m31 multiplicity_0 = cuda_evaluator.next_trace_mask();

    // ===================== 1. ReadPositiveKnownIdNumBits252 #1 =====================
    // The "known ID" variant just does a MemoryIdToBig lookup (no MemoryAddressToId).
    // Input: input_limb_0_col0 (known ID), value limbs col6-col33
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_0_col0,
            value_limb_0_col6, value_limb_1_col7, value_limb_2_col8, value_limb_3_col9,
            value_limb_4_col10, value_limb_5_col11, value_limb_6_col12, value_limb_7_col13,
            value_limb_8_col14, value_limb_9_col15, value_limb_10_col16, value_limb_11_col17,
            value_limb_12_col18, value_limb_13_col19, value_limb_14_col20, value_limb_15_col21,
            value_limb_16_col22, value_limb_17_col23, value_limb_18_col24, value_limb_19_col25,
            value_limb_20_col26, value_limb_21_col27, value_limb_22_col28, value_limb_23_col29,
            value_limb_24_col30, value_limb_25_col31, value_limb_26_col32, value_limb_27_col33
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // Compute 9 packed values for state word 0: packed = val_0 + val_1 * 512 + val_2 * 262144
    m31 packed_0_0 = add(add(value_limb_0_col6, mul(value_limb_1_col7, M31_512)), mul(value_limb_2_col8, M31_262144));
    m31 packed_0_1 = add(add(value_limb_3_col9, mul(value_limb_4_col10, M31_512)), mul(value_limb_5_col11, M31_262144));
    m31 packed_0_2 = add(add(value_limb_6_col12, mul(value_limb_7_col13, M31_512)), mul(value_limb_8_col14, M31_262144));
    m31 packed_0_3 = add(add(value_limb_9_col15, mul(value_limb_10_col16, M31_512)), mul(value_limb_11_col17, M31_262144));
    m31 packed_0_4 = add(add(value_limb_12_col18, mul(value_limb_13_col19, M31_512)), mul(value_limb_14_col20, M31_262144));
    m31 packed_0_5 = add(add(value_limb_15_col21, mul(value_limb_16_col22, M31_512)), mul(value_limb_17_col23, M31_262144));
    m31 packed_0_6 = add(add(value_limb_18_col24, mul(value_limb_19_col25, M31_512)), mul(value_limb_20_col26, M31_262144));
    m31 packed_0_7 = add(add(value_limb_21_col27, mul(value_limb_22_col28, M31_512)), mul(value_limb_23_col29, M31_262144));
    m31 packed_0_8 = add(add(value_limb_24_col30, mul(value_limb_25_col31, M31_512)), mul(value_limb_26_col32, M31_262144));

    // Note: add_intermediate in Rust is a no-op identity function in CUDA - no constraints added.

    // ===================== 2. ReadPositiveKnownIdNumBits252 #2 =====================
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_1_col1,
            value_limb_0_col34, value_limb_1_col35, value_limb_2_col36, value_limb_3_col37,
            value_limb_4_col38, value_limb_5_col39, value_limb_6_col40, value_limb_7_col41,
            value_limb_8_col42, value_limb_9_col43, value_limb_10_col44, value_limb_11_col45,
            value_limb_12_col46, value_limb_13_col47, value_limb_14_col48, value_limb_15_col49,
            value_limb_16_col50, value_limb_17_col51, value_limb_18_col52, value_limb_19_col53,
            value_limb_20_col54, value_limb_21_col55, value_limb_22_col56, value_limb_23_col57,
            value_limb_24_col58, value_limb_25_col59, value_limb_26_col60, value_limb_27_col61
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // Compute 9 packed values for state word 1
    m31 packed_1_0 = add(add(value_limb_0_col34, mul(value_limb_1_col35, M31_512)), mul(value_limb_2_col36, M31_262144));
    m31 packed_1_1 = add(add(value_limb_3_col37, mul(value_limb_4_col38, M31_512)), mul(value_limb_5_col39, M31_262144));
    m31 packed_1_2 = add(add(value_limb_6_col40, mul(value_limb_7_col41, M31_512)), mul(value_limb_8_col42, M31_262144));
    m31 packed_1_3 = add(add(value_limb_9_col43, mul(value_limb_10_col44, M31_512)), mul(value_limb_11_col45, M31_262144));
    m31 packed_1_4 = add(add(value_limb_12_col46, mul(value_limb_13_col47, M31_512)), mul(value_limb_14_col48, M31_262144));
    m31 packed_1_5 = add(add(value_limb_15_col49, mul(value_limb_16_col50, M31_512)), mul(value_limb_17_col51, M31_262144));
    m31 packed_1_6 = add(add(value_limb_18_col52, mul(value_limb_19_col53, M31_512)), mul(value_limb_20_col54, M31_262144));
    m31 packed_1_7 = add(add(value_limb_21_col55, mul(value_limb_22_col56, M31_512)), mul(value_limb_23_col57, M31_262144));
    m31 packed_1_8 = add(add(value_limb_24_col58, mul(value_limb_25_col59, M31_512)), mul(value_limb_26_col60, M31_262144));

    // Note: add_intermediate in Rust is a no-op identity function in CUDA - no constraints added.

    // ===================== 3. ReadPositiveKnownIdNumBits252 #3 =====================
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_2_col2,
            value_limb_0_col62, value_limb_1_col63, value_limb_2_col64, value_limb_3_col65,
            value_limb_4_col66, value_limb_5_col67, value_limb_6_col68, value_limb_7_col69,
            value_limb_8_col70, value_limb_9_col71, value_limb_10_col72, value_limb_11_col73,
            value_limb_12_col74, value_limb_13_col75, value_limb_14_col76, value_limb_15_col77,
            value_limb_16_col78, value_limb_17_col79, value_limb_18_col80, value_limb_19_col81,
            value_limb_20_col82, value_limb_21_col83, value_limb_22_col84, value_limb_23_col85,
            value_limb_24_col86, value_limb_25_col87, value_limb_26_col88, value_limb_27_col89
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // Compute 9 packed values for state word 2
    m31 packed_2_0 = add(add(value_limb_0_col62, mul(value_limb_1_col63, M31_512)), mul(value_limb_2_col64, M31_262144));
    m31 packed_2_1 = add(add(value_limb_3_col65, mul(value_limb_4_col66, M31_512)), mul(value_limb_5_col67, M31_262144));
    m31 packed_2_2 = add(add(value_limb_6_col68, mul(value_limb_7_col69, M31_512)), mul(value_limb_8_col70, M31_262144));
    m31 packed_2_3 = add(add(value_limb_9_col71, mul(value_limb_10_col72, M31_512)), mul(value_limb_11_col73, M31_262144));
    m31 packed_2_4 = add(add(value_limb_12_col74, mul(value_limb_13_col75, M31_512)), mul(value_limb_14_col76, M31_262144));
    m31 packed_2_5 = add(add(value_limb_15_col77, mul(value_limb_16_col78, M31_512)), mul(value_limb_17_col79, M31_262144));
    m31 packed_2_6 = add(add(value_limb_18_col80, mul(value_limb_19_col81, M31_512)), mul(value_limb_20_col82, M31_262144));
    m31 packed_2_7 = add(add(value_limb_21_col83, mul(value_limb_22_col84, M31_512)), mul(value_limb_23_col85, M31_262144));
    m31 packed_2_8 = add(add(value_limb_24_col86, mul(value_limb_25_col87, M31_512)), mul(value_limb_26_col88, M31_262144));

    // Note: add_intermediate in Rust is a no-op identity function in CUDA - no constraints added.

    // ===================== 4. PoseidonHadesPermutation =====================
    // Input: 30 packed limbs (9 packed + col33 per state word)
    // Columns: col90-col286 (197 columns mapped to subroutine col0-col196)
    poseidon_hades_permutation_evaluate(
        // 30 input limbs: state0 (9+1), state1 (9+1), state2 (9+1)
        packed_0_0, packed_0_1, packed_0_2, packed_0_3, packed_0_4,
        packed_0_5, packed_0_6, packed_0_7, packed_0_8, value_limb_27_col33,
        packed_1_0, packed_1_1, packed_1_2, packed_1_3, packed_1_4,
        packed_1_5, packed_1_6, packed_1_7, packed_1_8, value_limb_27_col61,
        packed_2_0, packed_2_1, packed_2_2, packed_2_3, packed_2_4,
        packed_2_5, packed_2_6, packed_2_7, packed_2_8, value_limb_27_col89,
        // col0-col10: First combination
        combination_limb_0_col90, combination_limb_1_col91, combination_limb_2_col92,
        combination_limb_3_col93, combination_limb_4_col94, combination_limb_5_col95,
        combination_limb_6_col96, combination_limb_7_col97, combination_limb_8_col98,
        combination_limb_9_col99, p_coef_col100,
        // col11-col21: Second combination
        combination_limb_0_col101, combination_limb_1_col102, combination_limb_2_col103,
        combination_limb_3_col104, combination_limb_4_col105, combination_limb_5_col106,
        combination_limb_6_col107, combination_limb_7_col108, combination_limb_8_col109,
        combination_limb_9_col110, p_coef_col111,
        // col22-col32: Third combination
        combination_limb_0_col112, combination_limb_1_col113, combination_limb_2_col114,
        combination_limb_3_col115, combination_limb_4_col116, combination_limb_5_col117,
        combination_limb_6_col118, combination_limb_7_col119, combination_limb_8_col120,
        combination_limb_9_col121, p_coef_col122,
        // col33-col62: First full round chain output (30 limbs)
        poseidon_full_round_chain_output_limb_0_col123, poseidon_full_round_chain_output_limb_1_col124,
        poseidon_full_round_chain_output_limb_2_col125, poseidon_full_round_chain_output_limb_3_col126,
        poseidon_full_round_chain_output_limb_4_col127, poseidon_full_round_chain_output_limb_5_col128,
        poseidon_full_round_chain_output_limb_6_col129, poseidon_full_round_chain_output_limb_7_col130,
        poseidon_full_round_chain_output_limb_8_col131, poseidon_full_round_chain_output_limb_9_col132,
        poseidon_full_round_chain_output_limb_10_col133, poseidon_full_round_chain_output_limb_11_col134,
        poseidon_full_round_chain_output_limb_12_col135, poseidon_full_round_chain_output_limb_13_col136,
        poseidon_full_round_chain_output_limb_14_col137, poseidon_full_round_chain_output_limb_15_col138,
        poseidon_full_round_chain_output_limb_16_col139, poseidon_full_round_chain_output_limb_17_col140,
        poseidon_full_round_chain_output_limb_18_col141, poseidon_full_round_chain_output_limb_19_col142,
        poseidon_full_round_chain_output_limb_20_col143, poseidon_full_round_chain_output_limb_21_col144,
        poseidon_full_round_chain_output_limb_22_col145, poseidon_full_round_chain_output_limb_23_col146,
        poseidon_full_round_chain_output_limb_24_col147, poseidon_full_round_chain_output_limb_25_col148,
        poseidon_full_round_chain_output_limb_26_col149, poseidon_full_round_chain_output_limb_27_col150,
        poseidon_full_round_chain_output_limb_28_col151, poseidon_full_round_chain_output_limb_29_col152,
        // col63-col72: First cube output
        cube_252_output_limb_0_col153, cube_252_output_limb_1_col154,
        cube_252_output_limb_2_col155, cube_252_output_limb_3_col156,
        cube_252_output_limb_4_col157, cube_252_output_limb_5_col158,
        cube_252_output_limb_6_col159, cube_252_output_limb_7_col160,
        cube_252_output_limb_8_col161, cube_252_output_limb_9_col162,
        // col73-col83: Fourth combination
        combination_limb_0_col163, combination_limb_1_col164, combination_limb_2_col165,
        combination_limb_3_col166, combination_limb_4_col167, combination_limb_5_col168,
        combination_limb_6_col169, combination_limb_7_col170, combination_limb_8_col171,
        combination_limb_9_col172, p_coef_col173,
        // col84-col93: Second cube output
        cube_252_output_limb_0_col174, cube_252_output_limb_1_col175,
        cube_252_output_limb_2_col176, cube_252_output_limb_3_col177,
        cube_252_output_limb_4_col178, cube_252_output_limb_5_col179,
        cube_252_output_limb_6_col180, cube_252_output_limb_7_col181,
        cube_252_output_limb_8_col182, cube_252_output_limb_9_col183,
        // col94-col104: Fifth combination
        combination_limb_0_col184, combination_limb_1_col185, combination_limb_2_col186,
        combination_limb_3_col187, combination_limb_4_col188, combination_limb_5_col189,
        combination_limb_6_col190, combination_limb_7_col191, combination_limb_8_col192,
        combination_limb_9_col193, p_coef_col194,
        // col105-col144: Partial rounds chain output (40 limbs)
        poseidon_3_partial_rounds_chain_output_limb_0_col195, poseidon_3_partial_rounds_chain_output_limb_1_col196,
        poseidon_3_partial_rounds_chain_output_limb_2_col197, poseidon_3_partial_rounds_chain_output_limb_3_col198,
        poseidon_3_partial_rounds_chain_output_limb_4_col199, poseidon_3_partial_rounds_chain_output_limb_5_col200,
        poseidon_3_partial_rounds_chain_output_limb_6_col201, poseidon_3_partial_rounds_chain_output_limb_7_col202,
        poseidon_3_partial_rounds_chain_output_limb_8_col203, poseidon_3_partial_rounds_chain_output_limb_9_col204,
        poseidon_3_partial_rounds_chain_output_limb_10_col205, poseidon_3_partial_rounds_chain_output_limb_11_col206,
        poseidon_3_partial_rounds_chain_output_limb_12_col207, poseidon_3_partial_rounds_chain_output_limb_13_col208,
        poseidon_3_partial_rounds_chain_output_limb_14_col209, poseidon_3_partial_rounds_chain_output_limb_15_col210,
        poseidon_3_partial_rounds_chain_output_limb_16_col211, poseidon_3_partial_rounds_chain_output_limb_17_col212,
        poseidon_3_partial_rounds_chain_output_limb_18_col213, poseidon_3_partial_rounds_chain_output_limb_19_col214,
        poseidon_3_partial_rounds_chain_output_limb_20_col215, poseidon_3_partial_rounds_chain_output_limb_21_col216,
        poseidon_3_partial_rounds_chain_output_limb_22_col217, poseidon_3_partial_rounds_chain_output_limb_23_col218,
        poseidon_3_partial_rounds_chain_output_limb_24_col219, poseidon_3_partial_rounds_chain_output_limb_25_col220,
        poseidon_3_partial_rounds_chain_output_limb_26_col221, poseidon_3_partial_rounds_chain_output_limb_27_col222,
        poseidon_3_partial_rounds_chain_output_limb_28_col223, poseidon_3_partial_rounds_chain_output_limb_29_col224,
        poseidon_3_partial_rounds_chain_output_limb_30_col225, poseidon_3_partial_rounds_chain_output_limb_31_col226,
        poseidon_3_partial_rounds_chain_output_limb_32_col227, poseidon_3_partial_rounds_chain_output_limb_33_col228,
        poseidon_3_partial_rounds_chain_output_limb_34_col229, poseidon_3_partial_rounds_chain_output_limb_35_col230,
        poseidon_3_partial_rounds_chain_output_limb_36_col231, poseidon_3_partial_rounds_chain_output_limb_37_col232,
        poseidon_3_partial_rounds_chain_output_limb_38_col233, poseidon_3_partial_rounds_chain_output_limb_39_col234,
        // col145-col155: Sixth combination
        combination_limb_0_col235, combination_limb_1_col236, combination_limb_2_col237,
        combination_limb_3_col238, combination_limb_4_col239, combination_limb_5_col240,
        combination_limb_6_col241, combination_limb_7_col242, combination_limb_8_col243,
        combination_limb_9_col244, p_coef_col245,
        // col156-col166: Seventh combination
        combination_limb_0_col246, combination_limb_1_col247, combination_limb_2_col248,
        combination_limb_3_col249, combination_limb_4_col250, combination_limb_5_col251,
        combination_limb_6_col252, combination_limb_7_col253, combination_limb_8_col254,
        combination_limb_9_col255, p_coef_col256,
        // col167-col196: Second full round chain output (30 limbs)
        poseidon_full_round_chain_output_limb_0_col257, poseidon_full_round_chain_output_limb_1_col258,
        poseidon_full_round_chain_output_limb_2_col259, poseidon_full_round_chain_output_limb_3_col260,
        poseidon_full_round_chain_output_limb_4_col261, poseidon_full_round_chain_output_limb_5_col262,
        poseidon_full_round_chain_output_limb_6_col263, poseidon_full_round_chain_output_limb_7_col264,
        poseidon_full_round_chain_output_limb_8_col265, poseidon_full_round_chain_output_limb_9_col266,
        poseidon_full_round_chain_output_limb_10_col267, poseidon_full_round_chain_output_limb_11_col268,
        poseidon_full_round_chain_output_limb_12_col269, poseidon_full_round_chain_output_limb_13_col270,
        poseidon_full_round_chain_output_limb_14_col271, poseidon_full_round_chain_output_limb_15_col272,
        poseidon_full_round_chain_output_limb_16_col273, poseidon_full_round_chain_output_limb_17_col274,
        poseidon_full_round_chain_output_limb_18_col275, poseidon_full_round_chain_output_limb_19_col276,
        poseidon_full_round_chain_output_limb_20_col277, poseidon_full_round_chain_output_limb_21_col278,
        poseidon_full_round_chain_output_limb_22_col279, poseidon_full_round_chain_output_limb_23_col280,
        poseidon_full_round_chain_output_limb_24_col281, poseidon_full_round_chain_output_limb_25_col282,
        poseidon_full_round_chain_output_limb_26_col283, poseidon_full_round_chain_output_limb_27_col284,
        poseidon_full_round_chain_output_limb_28_col285, poseidon_full_round_chain_output_limb_29_col286,
        // Lookup elements and seq
        agg_eval->common_lookup_elements,
        seq,
        &cuda_evaluator
    );

    // ===================== 5. Felt252UnpackFrom27 #1 =====================
    // Input: second full round chain output limbs 0-9 (col257-col266)
    // Unpacked limbs: col287-col304
    m31 unpack_output_1[10];
    felt_252_unpack_from_27_evaluate(
        poseidon_full_round_chain_output_limb_0_col257,
        poseidon_full_round_chain_output_limb_1_col258,
        poseidon_full_round_chain_output_limb_2_col259,
        poseidon_full_round_chain_output_limb_3_col260,
        poseidon_full_round_chain_output_limb_4_col261,
        poseidon_full_round_chain_output_limb_5_col262,
        poseidon_full_round_chain_output_limb_6_col263,
        poseidon_full_round_chain_output_limb_7_col264,
        poseidon_full_round_chain_output_limb_8_col265,
        poseidon_full_round_chain_output_limb_9_col266,
        unpacked_limb_0_col287, unpacked_limb_1_col288,
        unpacked_limb_3_col289, unpacked_limb_4_col290,
        unpacked_limb_6_col291, unpacked_limb_7_col292,
        unpacked_limb_9_col293, unpacked_limb_10_col294,
        unpacked_limb_12_col295, unpacked_limb_13_col296,
        unpacked_limb_15_col297, unpacked_limb_16_col298,
        unpacked_limb_18_col299, unpacked_limb_19_col300,
        unpacked_limb_21_col301, unpacked_limb_22_col302,
        unpacked_limb_24_col303, unpacked_limb_25_col304,
        unpack_output_1,
        &cuda_evaluator
    );

    // ===================== 6. MemoryIdToBig #1 (for output state 0) =====================
    // 30 values: [RELATION_ID, input_limb_3_col3, unpacked_0, unpacked_1, computed_2, ...]
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_3_col3,
            unpacked_limb_0_col287, unpacked_limb_1_col288, unpack_output_1[0],
            unpacked_limb_3_col289, unpacked_limb_4_col290, unpack_output_1[1],
            unpacked_limb_6_col291, unpacked_limb_7_col292, unpack_output_1[2],
            unpacked_limb_9_col293, unpacked_limb_10_col294, unpack_output_1[3],
            unpacked_limb_12_col295, unpacked_limb_13_col296, unpack_output_1[4],
            unpacked_limb_15_col297, unpacked_limb_16_col298, unpack_output_1[5],
            unpacked_limb_18_col299, unpacked_limb_19_col300, unpack_output_1[6],
            unpacked_limb_21_col301, unpacked_limb_22_col302, unpack_output_1[7],
            unpacked_limb_24_col303, unpacked_limb_25_col304, unpack_output_1[8],
            poseidon_full_round_chain_output_limb_9_col266
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // ===================== 7. Felt252UnpackFrom27 #2 =====================
    // Input: second full round chain output limbs 10-19 (col267-col276)
    // Unpacked limbs: col305-col322
    m31 unpack_output_2[10];
    felt_252_unpack_from_27_evaluate(
        poseidon_full_round_chain_output_limb_10_col267,
        poseidon_full_round_chain_output_limb_11_col268,
        poseidon_full_round_chain_output_limb_12_col269,
        poseidon_full_round_chain_output_limb_13_col270,
        poseidon_full_round_chain_output_limb_14_col271,
        poseidon_full_round_chain_output_limb_15_col272,
        poseidon_full_round_chain_output_limb_16_col273,
        poseidon_full_round_chain_output_limb_17_col274,
        poseidon_full_round_chain_output_limb_18_col275,
        poseidon_full_round_chain_output_limb_19_col276,
        unpacked_limb_0_col305, unpacked_limb_1_col306,
        unpacked_limb_3_col307, unpacked_limb_4_col308,
        unpacked_limb_6_col309, unpacked_limb_7_col310,
        unpacked_limb_9_col311, unpacked_limb_10_col312,
        unpacked_limb_12_col313, unpacked_limb_13_col314,
        unpacked_limb_15_col315, unpacked_limb_16_col316,
        unpacked_limb_18_col317, unpacked_limb_19_col318,
        unpacked_limb_21_col319, unpacked_limb_22_col320,
        unpacked_limb_24_col321, unpacked_limb_25_col322,
        unpack_output_2,
        &cuda_evaluator
    );

    // ===================== 8. MemoryIdToBig #2 (for output state 1) =====================
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_4_col4,
            unpacked_limb_0_col305, unpacked_limb_1_col306, unpack_output_2[0],
            unpacked_limb_3_col307, unpacked_limb_4_col308, unpack_output_2[1],
            unpacked_limb_6_col309, unpacked_limb_7_col310, unpack_output_2[2],
            unpacked_limb_9_col311, unpacked_limb_10_col312, unpack_output_2[3],
            unpacked_limb_12_col313, unpacked_limb_13_col314, unpack_output_2[4],
            unpacked_limb_15_col315, unpacked_limb_16_col316, unpack_output_2[5],
            unpacked_limb_18_col317, unpacked_limb_19_col318, unpack_output_2[6],
            unpacked_limb_21_col319, unpacked_limb_22_col320, unpack_output_2[7],
            unpacked_limb_24_col321, unpacked_limb_25_col322, unpack_output_2[8],
            poseidon_full_round_chain_output_limb_19_col276
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // ===================== 9. Felt252UnpackFrom27 #3 =====================
    // Input: second full round chain output limbs 20-29 (col277-col286)
    // Unpacked limbs: col323-col340
    m31 unpack_output_3[10];
    felt_252_unpack_from_27_evaluate(
        poseidon_full_round_chain_output_limb_20_col277,
        poseidon_full_round_chain_output_limb_21_col278,
        poseidon_full_round_chain_output_limb_22_col279,
        poseidon_full_round_chain_output_limb_23_col280,
        poseidon_full_round_chain_output_limb_24_col281,
        poseidon_full_round_chain_output_limb_25_col282,
        poseidon_full_round_chain_output_limb_26_col283,
        poseidon_full_round_chain_output_limb_27_col284,
        poseidon_full_round_chain_output_limb_28_col285,
        poseidon_full_round_chain_output_limb_29_col286,
        unpacked_limb_0_col323, unpacked_limb_1_col324,
        unpacked_limb_3_col325, unpacked_limb_4_col326,
        unpacked_limb_6_col327, unpacked_limb_7_col328,
        unpacked_limb_9_col329, unpacked_limb_10_col330,
        unpacked_limb_12_col331, unpacked_limb_13_col332,
        unpacked_limb_15_col333, unpacked_limb_16_col334,
        unpacked_limb_18_col335, unpacked_limb_19_col336,
        unpacked_limb_21_col337, unpacked_limb_22_col338,
        unpacked_limb_24_col339, unpacked_limb_25_col340,
        unpack_output_3,
        &cuda_evaluator
    );

    // ===================== 10. MemoryIdToBig #3 (for output state 2) =====================
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            input_limb_5_col5,
            unpacked_limb_0_col323, unpacked_limb_1_col324, unpack_output_3[0],
            unpacked_limb_3_col325, unpacked_limb_4_col326, unpack_output_3[1],
            unpacked_limb_6_col327, unpacked_limb_7_col328, unpack_output_3[2],
            unpacked_limb_9_col329, unpacked_limb_10_col330, unpack_output_3[3],
            unpacked_limb_12_col331, unpacked_limb_13_col332, unpack_output_3[4],
            unpacked_limb_15_col333, unpacked_limb_16_col334, unpack_output_3[5],
            unpacked_limb_18_col335, unpacked_limb_19_col336, unpack_output_3[6],
            unpacked_limb_21_col337, unpacked_limb_22_col338, unpack_output_3[7],
            unpacked_limb_24_col339, unpacked_limb_25_col340, unpack_output_3[8],
            poseidon_full_round_chain_output_limb_29_col286
        };
        cuda_evaluator.add_to_relation<30>(agg_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // ===================== 11. PoseidonAggregator PROVIDE =====================
    // Multiplicity: -multiplicity_0 (PROVIDE = negative)
    {
        m31 values[7] = {
            POSEIDON_AGGREGATOR_RELATION_ID,
            input_limb_0_col0,
            input_limb_1_col1,
            input_limb_2_col2,
            input_limb_3_col3,
            input_limb_4_col4,
            input_limb_5_col5
        };
        m31 neg_mult = neg(multiplicity_0);
        qm31 multiplicity_ext = {{neg_mult, 0}, {0, 0}};
        cuda_evaluator.add_to_relation<7>(
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
void evaluate_poseidon_aggregator(
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

    PoseidonAggregator_Eval *device_agg_eval = cuda_malloc<PoseidonAggregator_Eval>(1);
    cuda_mem_copy_host_to_device<PoseidonAggregator_Eval>((PoseidonAggregator_Eval*)eval, device_agg_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);

    int block_dim = eval_domain_size < POSEIDON_AGGREGATOR_THREAD_COUNT_MAX ? eval_domain_size : POSEIDON_AGGREGATOR_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_poseidon_aggregator_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
        evaluate_poseidon_aggregator_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
