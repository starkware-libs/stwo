/*
============================================
Poseidon3PartialRoundsChain CUDA Evaluator
============================================

Component: Poseidon3PartialRoundsChain
translated from: cairo-air/src/comptogethernts/poseidon_3_partial_rounds_chain.rs
AIR version: 54d95c0d

Functionality:
- Implements 3 partial rounds chain of Poseidon permutation
- Connects multiple partial rounds with round keys
- Uses lookup chaining for state progression

Inputs/Outputs:
- Trace Columns: 169
  - Columns 0-41: Input limbs (42 total)
  - Columns 42-71: PoseidonRoundKeys output (30 limbs)
  - Columns 72-103: First PoseidonPartialRound data (32 columns)
  - Columns 104-135: Second PoseidonPartialRound data (32 columns)
  - Columns 136-167: Third PoseidonPartialRound data (32 columns)
  - Column 168: Enabler (boolean)

Constraint Logic:
- 1 enabler boolean constraint (enabler² - enabler = 0)
- 1 PoseidonRoundKeys lookup
- 3 PoseidonPartialRound calls
- 2 Poseidon3PartialRoundsChain self-lookups (chaining with ±enabler)

Relation Lookups:
- PoseidonRoundKeys: 1 use
- Poseidon3PartialRoundsChain: 2 uses (self-chaining)
- Plus all lookups from 3× PoseidonPartialRound calls

============================================
*/

#include <cstdio>
#include <vector>

#include "evaluate_poseidon_3_partial_rounds_chain.cuh"
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_common.cuh"
#include "relations.cuh"
#include "../evaluate_poseidon_partial_round.cuh"

#define POSEIDON_3_PARTIAL_ROUNDS_CHAIN_THREAD_COUNT_MAX 256

// Pre-kernel: Evaluates constraints and relation lookups for poseidon_3_partial_rounds_chain
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_poseidon_3_partial_rounds_chain_pre_kernel(
    qm31 *numerators,
    m31 **trace_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    Poseidon3PartialRoundsChain_Eval *poseidon_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    unsigned eval_domain_size = 1U << eval_domain_log_size;
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= eval_domain_size) {
        return;
    }

    // Initialize evaluator
    EvaluatorT cuda_evaluator(
        trace_evaluations,
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

    // Constants
    const m31 M31_1 = m31(1);

    // Read all 169 trace columns
    // Columns 0-41: Input limbs
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
    m31 input_limb_35_col35 = cuda_evaluator.next_trace_mask();
    m31 input_limb_36_col36 = cuda_evaluator.next_trace_mask();
    m31 input_limb_37_col37 = cuda_evaluator.next_trace_mask();
    m31 input_limb_38_col38 = cuda_evaluator.next_trace_mask();
    m31 input_limb_39_col39 = cuda_evaluator.next_trace_mask();
    m31 input_limb_40_col40 = cuda_evaluator.next_trace_mask();
    m31 input_limb_41_col41 = cuda_evaluator.next_trace_mask();

    // Columns 42-71: PoseidonRoundKeys output (30 limbs)
    m31 poseidon_round_keys_output_limb_0_col42 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_1_col43 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_2_col44 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_3_col45 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_4_col46 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_5_col47 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_6_col48 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_7_col49 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_8_col50 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_9_col51 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_10_col52 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_11_col53 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_12_col54 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_13_col55 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_14_col56 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_15_col57 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_16_col58 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_17_col59 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_18_col60 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_19_col61 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_20_col62 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_21_col63 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_22_col64 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_23_col65 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_24_col66 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_25_col67 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_26_col68 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_27_col69 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_28_col70 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_29_col71 = cuda_evaluator.next_trace_mask();

    // Columns 72-103: First PoseidonPartialRound data
    m31 cube_252_output_limb_0_col72 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_1_col73 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_2_col74 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_3_col75 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_4_col76 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_5_col77 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_6_col78 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_7_col79 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_8_col80 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_9_col81 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col82 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col83 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col84 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col85 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col86 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col87 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col88 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col89 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col90 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col91 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col92 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col93 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col94 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col95 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col96 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col97 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col98 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col99 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col100 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col101 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col102 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col103 = cuda_evaluator.next_trace_mask();

    // Columns 104-135: Second PoseidonPartialRound data
    m31 cube_252_output_limb_0_col104 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_1_col105 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_2_col106 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_3_col107 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_4_col108 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_5_col109 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_6_col110 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_7_col111 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_8_col112 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_9_col113 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col114 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col115 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col116 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col117 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col118 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col119 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col120 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col121 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col122 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col123 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col124 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col125 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col126 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col127 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col128 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col129 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col130 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col131 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col132 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col133 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col134 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col135 = cuda_evaluator.next_trace_mask();

    // Columns 136-167: Third PoseidonPartialRound data
    m31 cube_252_output_limb_0_col136 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_1_col137 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_2_col138 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_3_col139 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_4_col140 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_5_col141 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_6_col142 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_7_col143 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_8_col144 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_9_col145 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col146 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col147 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col148 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col149 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col150 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col151 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col152 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col153 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col154 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col155 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col156 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_0_col157 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col158 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col159 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col160 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col161 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col162 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col163 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col164 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col165 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col166 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col167 = cuda_evaluator.next_trace_mask();

    // Column 168: Enabler
    m31 enabler = cuda_evaluator.next_trace_mask();

    // PoseidonRoundKeys lookup (32 values: RELATION_ID + input_limb_1_col1 + 30 output limbs)
    {
        m31 values[32];
        values[0] = POSEIDON_ROUND_KEYS_RELATION_ID;
        values[1] = input_limb_1_col1;
        values[2] = poseidon_round_keys_output_limb_0_col42;
        values[3] = poseidon_round_keys_output_limb_1_col43;
        values[4] = poseidon_round_keys_output_limb_2_col44;
        values[5] = poseidon_round_keys_output_limb_3_col45;
        values[6] = poseidon_round_keys_output_limb_4_col46;
        values[7] = poseidon_round_keys_output_limb_5_col47;
        values[8] = poseidon_round_keys_output_limb_6_col48;
        values[9] = poseidon_round_keys_output_limb_7_col49;
        values[10] = poseidon_round_keys_output_limb_8_col50;
        values[11] = poseidon_round_keys_output_limb_9_col51;
        values[12] = poseidon_round_keys_output_limb_10_col52;
        values[13] = poseidon_round_keys_output_limb_11_col53;
        values[14] = poseidon_round_keys_output_limb_12_col54;
        values[15] = poseidon_round_keys_output_limb_13_col55;
        values[16] = poseidon_round_keys_output_limb_14_col56;
        values[17] = poseidon_round_keys_output_limb_15_col57;
        values[18] = poseidon_round_keys_output_limb_16_col58;
        values[19] = poseidon_round_keys_output_limb_17_col59;
        values[20] = poseidon_round_keys_output_limb_18_col60;
        values[21] = poseidon_round_keys_output_limb_19_col61;
        values[22] = poseidon_round_keys_output_limb_20_col62;
        values[23] = poseidon_round_keys_output_limb_21_col63;
        values[24] = poseidon_round_keys_output_limb_22_col64;
        values[25] = poseidon_round_keys_output_limb_23_col65;
        values[26] = poseidon_round_keys_output_limb_24_col66;
        values[27] = poseidon_round_keys_output_limb_25_col67;
        values[28] = poseidon_round_keys_output_limb_26_col68;
        values[29] = poseidon_round_keys_output_limb_27_col69;
        values[30] = poseidon_round_keys_output_limb_28_col70;
        values[31] = poseidon_round_keys_output_limb_29_col71;

        cuda_evaluator.add_to_relation<32>(poseidon_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // First PoseidonPartialRound call
    // Input[50]: columns 2-41 (40 limbs) + columns 42-51 (10 limbs)
    // Cube output: columns 72-81
    // Combination 1: columns 82-92
    // Combination 2: columns 93-103
    poseidon_partial_round_evaluate(
        // z0_3 (input[0-9]): columns 2-11
        input_limb_2_col2, input_limb_3_col3, input_limb_4_col4, input_limb_5_col5,
        input_limb_6_col6, input_limb_7_col7, input_limb_8_col8, input_limb_9_col9,
        input_limb_10_col10, input_limb_11_col11,
        // z1 (input[10-19]): columns 12-21
        input_limb_12_col12, input_limb_13_col13, input_limb_14_col14, input_limb_15_col15,
        input_limb_16_col16, input_limb_17_col17, input_limb_18_col18, input_limb_19_col19,
        input_limb_20_col20, input_limb_21_col21,
        // z1_3 (input[20-29]): columns 22-31
        input_limb_22_col22, input_limb_23_col23, input_limb_24_col24, input_limb_25_col25,
        input_limb_26_col26, input_limb_27_col27, input_limb_28_col28, input_limb_29_col29,
        input_limb_30_col30, input_limb_31_col31,
        // z2 (input[30-39]): columns 32-41
        input_limb_32_col32, input_limb_33_col33, input_limb_34_col34, input_limb_35_col35,
        input_limb_36_col36, input_limb_37_col37, input_limb_38_col38, input_limb_39_col39,
        input_limb_40_col40, input_limb_41_col41,
        // half_key (input[40-49]): columns 42-51
        poseidon_round_keys_output_limb_0_col42, poseidon_round_keys_output_limb_1_col43,
        poseidon_round_keys_output_limb_2_col44, poseidon_round_keys_output_limb_3_col45,
        poseidon_round_keys_output_limb_4_col46, poseidon_round_keys_output_limb_5_col47,
        poseidon_round_keys_output_limb_6_col48, poseidon_round_keys_output_limb_7_col49,
        poseidon_round_keys_output_limb_8_col50, poseidon_round_keys_output_limb_9_col51,
        // Cube252 output: columns 72-81
        cube_252_output_limb_0_col72, cube_252_output_limb_1_col73, cube_252_output_limb_2_col74,
        cube_252_output_limb_3_col75, cube_252_output_limb_4_col76, cube_252_output_limb_5_col77,
        cube_252_output_limb_6_col78, cube_252_output_limb_7_col79, cube_252_output_limb_8_col80,
        cube_252_output_limb_9_col81,
        // First combination: columns 82-92
        combination_limb_0_col82, combination_limb_1_col83, combination_limb_2_col84,
        combination_limb_3_col85, combination_limb_4_col86, combination_limb_5_col87,
        combination_limb_6_col88, combination_limb_7_col89, combination_limb_8_col90,
        combination_limb_9_col91, p_coef_col92,
        // Second combination: columns 93-103
        combination_limb_0_col93, combination_limb_1_col94, combination_limb_2_col95,
        combination_limb_3_col96, combination_limb_4_col97, combination_limb_5_col98,
        combination_limb_6_col99, combination_limb_7_col100, combination_limb_8_col101,
        combination_limb_9_col102, p_coef_col103,
        // Lookup elements
        poseidon_eval->common_lookup_elements,
        // Evaluator
        &cuda_evaluator
    );

    // Second PoseidonPartialRound call
    // Input[50]: columns 22-41 (20 limbs) + columns 72-81 (10 limbs) + columns 93-102 (10 limbs) + columns 52-61 (10 limbs)
    // Cube output: columns 104-113
    // Combination 1: columns 114-124
    // Combination 2: columns 125-135
    poseidon_partial_round_evaluate(
        // z0_3 (input[0-9]): columns 22-31
        input_limb_22_col22, input_limb_23_col23, input_limb_24_col24, input_limb_25_col25,
        input_limb_26_col26, input_limb_27_col27, input_limb_28_col28, input_limb_29_col29,
        input_limb_30_col30, input_limb_31_col31,
        // z1 (input[10-19]): columns 32-41
        input_limb_32_col32, input_limb_33_col33, input_limb_34_col34, input_limb_35_col35,
        input_limb_36_col36, input_limb_37_col37, input_limb_38_col38, input_limb_39_col39,
        input_limb_40_col40, input_limb_41_col41,
        // z1_3 (input[20-29]): columns 72-81
        cube_252_output_limb_0_col72, cube_252_output_limb_1_col73, cube_252_output_limb_2_col74,
        cube_252_output_limb_3_col75, cube_252_output_limb_4_col76, cube_252_output_limb_5_col77,
        cube_252_output_limb_6_col78, cube_252_output_limb_7_col79, cube_252_output_limb_8_col80,
        cube_252_output_limb_9_col81,
        // z2 (input[30-39]): columns 93-102
        combination_limb_0_col93, combination_limb_1_col94, combination_limb_2_col95,
        combination_limb_3_col96, combination_limb_4_col97, combination_limb_5_col98,
        combination_limb_6_col99, combination_limb_7_col100, combination_limb_8_col101,
        combination_limb_9_col102,
        // half_key (input[40-49]): columns 52-61
        poseidon_round_keys_output_limb_10_col52, poseidon_round_keys_output_limb_11_col53,
        poseidon_round_keys_output_limb_12_col54, poseidon_round_keys_output_limb_13_col55,
        poseidon_round_keys_output_limb_14_col56, poseidon_round_keys_output_limb_15_col57,
        poseidon_round_keys_output_limb_16_col58, poseidon_round_keys_output_limb_17_col59,
        poseidon_round_keys_output_limb_18_col60, poseidon_round_keys_output_limb_19_col61,
        // Cube252 output: columns 104-113
        cube_252_output_limb_0_col104, cube_252_output_limb_1_col105, cube_252_output_limb_2_col106,
        cube_252_output_limb_3_col107, cube_252_output_limb_4_col108, cube_252_output_limb_5_col109,
        cube_252_output_limb_6_col110, cube_252_output_limb_7_col111, cube_252_output_limb_8_col112,
        cube_252_output_limb_9_col113,
        // First combination: columns 114-124
        combination_limb_0_col114, combination_limb_1_col115, combination_limb_2_col116,
        combination_limb_3_col117, combination_limb_4_col118, combination_limb_5_col119,
        combination_limb_6_col120, combination_limb_7_col121, combination_limb_8_col122,
        combination_limb_9_col123, p_coef_col124,
        // Second combination: columns 125-135
        combination_limb_0_col125, combination_limb_1_col126, combination_limb_2_col127,
        combination_limb_3_col128, combination_limb_4_col129, combination_limb_5_col130,
        combination_limb_6_col131, combination_limb_7_col132, combination_limb_8_col133,
        combination_limb_9_col134, p_coef_col135,
        // Lookup elements
        poseidon_eval->common_lookup_elements,
        // Evaluator
        &cuda_evaluator
    );

    // Third PoseidonPartialRound call
    // Input[50]: columns 72-81 (10 limbs) + columns 93-102 (10 limbs) + columns 104-113 (10 limbs) + columns 125-134 (10 limbs) + columns 62-71 (10 limbs)
    // Cube output: columns 136-145
    // Combination 1: columns 146-156
    // Combination 2: columns 157-167
    poseidon_partial_round_evaluate(
        // z0_3 (input[0-9]): columns 72-81
        cube_252_output_limb_0_col72, cube_252_output_limb_1_col73, cube_252_output_limb_2_col74,
        cube_252_output_limb_3_col75, cube_252_output_limb_4_col76, cube_252_output_limb_5_col77,
        cube_252_output_limb_6_col78, cube_252_output_limb_7_col79, cube_252_output_limb_8_col80,
        cube_252_output_limb_9_col81,
        // z1 (input[10-19]): columns 93-102
        combination_limb_0_col93, combination_limb_1_col94, combination_limb_2_col95,
        combination_limb_3_col96, combination_limb_4_col97, combination_limb_5_col98,
        combination_limb_6_col99, combination_limb_7_col100, combination_limb_8_col101,
        combination_limb_9_col102,
        // z1_3 (input[20-29]): columns 104-113
        cube_252_output_limb_0_col104, cube_252_output_limb_1_col105, cube_252_output_limb_2_col106,
        cube_252_output_limb_3_col107, cube_252_output_limb_4_col108, cube_252_output_limb_5_col109,
        cube_252_output_limb_6_col110, cube_252_output_limb_7_col111, cube_252_output_limb_8_col112,
        cube_252_output_limb_9_col113,
        // z2 (input[30-39]): columns 125-134
        combination_limb_0_col125, combination_limb_1_col126, combination_limb_2_col127,
        combination_limb_3_col128, combination_limb_4_col129, combination_limb_5_col130,
        combination_limb_6_col131, combination_limb_7_col132, combination_limb_8_col133,
        combination_limb_9_col134,
        // half_key (input[40-49]): columns 62-71
        poseidon_round_keys_output_limb_20_col62, poseidon_round_keys_output_limb_21_col63,
        poseidon_round_keys_output_limb_22_col64, poseidon_round_keys_output_limb_23_col65,
        poseidon_round_keys_output_limb_24_col66, poseidon_round_keys_output_limb_25_col67,
        poseidon_round_keys_output_limb_26_col68, poseidon_round_keys_output_limb_27_col69,
        poseidon_round_keys_output_limb_28_col70, poseidon_round_keys_output_limb_29_col71,
        // Cube252 output: columns 136-145
        cube_252_output_limb_0_col136, cube_252_output_limb_1_col137, cube_252_output_limb_2_col138,
        cube_252_output_limb_3_col139, cube_252_output_limb_4_col140, cube_252_output_limb_5_col141,
        cube_252_output_limb_6_col142, cube_252_output_limb_7_col143, cube_252_output_limb_8_col144,
        cube_252_output_limb_9_col145,
        // First combination: columns 146-156
        combination_limb_0_col146, combination_limb_1_col147, combination_limb_2_col148,
        combination_limb_3_col149, combination_limb_4_col150, combination_limb_5_col151,
        combination_limb_6_col152, combination_limb_7_col153, combination_limb_8_col154,
        combination_limb_9_col155, p_coef_col156,
        // Second combination: columns 157-167
        combination_limb_0_col157, combination_limb_1_col158, combination_limb_2_col159,
        combination_limb_3_col160, combination_limb_4_col161, combination_limb_5_col162,
        combination_limb_6_col163, combination_limb_7_col164, combination_limb_8_col165,
        combination_limb_9_col166, p_coef_col167,
        // Lookup elements
        poseidon_eval->common_lookup_elements,
        // Evaluator
        &cuda_evaluator
    );

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // First self-lookup: positive multiplicity (enabler)
    // Represents input state for this partial rounds chain
    {
        m31 values[43];
        values[0] = POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID;
        values[1] = input_limb_0_col0;
        values[2] = input_limb_1_col1;
        values[3] = input_limb_2_col2;
        values[4] = input_limb_3_col3;
        values[5] = input_limb_4_col4;
        values[6] = input_limb_5_col5;
        values[7] = input_limb_6_col6;
        values[8] = input_limb_7_col7;
        values[9] = input_limb_8_col8;
        values[10] = input_limb_9_col9;
        values[11] = input_limb_10_col10;
        values[12] = input_limb_11_col11;
        values[13] = input_limb_12_col12;
        values[14] = input_limb_13_col13;
        values[15] = input_limb_14_col14;
        values[16] = input_limb_15_col15;
        values[17] = input_limb_16_col16;
        values[18] = input_limb_17_col17;
        values[19] = input_limb_18_col18;
        values[20] = input_limb_19_col19;
        values[21] = input_limb_20_col20;
        values[22] = input_limb_21_col21;
        values[23] = input_limb_22_col22;
        values[24] = input_limb_23_col23;
        values[25] = input_limb_24_col24;
        values[26] = input_limb_25_col25;
        values[27] = input_limb_26_col26;
        values[28] = input_limb_27_col27;
        values[29] = input_limb_28_col28;
        values[30] = input_limb_29_col29;
        values[31] = input_limb_30_col30;
        values[32] = input_limb_31_col31;
        values[33] = input_limb_32_col32;
        values[34] = input_limb_33_col33;
        values[35] = input_limb_34_col34;
        values[36] = input_limb_35_col35;
        values[37] = input_limb_36_col36;
        values[38] = input_limb_37_col37;
        values[39] = input_limb_38_col38;
        values[40] = input_limb_39_col39;
        values[41] = input_limb_40_col40;
        values[42] = input_limb_41_col41;

        qm31 multiplicity = qm31{{enabler, 0}, {0, 0}};
        cuda_evaluator.add_to_relation<43>(poseidon_eval->common_lookup_elements, multiplicity, values);
    }

    // Second self-lookup: negative multiplicity (-enabler)
    // Represents output state after this partial rounds chain (input to next chain)
    {
        m31 values[43];
        values[0] = POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID;
        values[1] = input_limb_0_col0;
        values[2] = add(input_limb_1_col1, M31_1);  // input_limb_1_col1 + 1
        values[3] = cube_252_output_limb_0_col104;
        values[4] = cube_252_output_limb_1_col105;
        values[5] = cube_252_output_limb_2_col106;
        values[6] = cube_252_output_limb_3_col107;
        values[7] = cube_252_output_limb_4_col108;
        values[8] = cube_252_output_limb_5_col109;
        values[9] = cube_252_output_limb_6_col110;
        values[10] = cube_252_output_limb_7_col111;
        values[11] = cube_252_output_limb_8_col112;
        values[12] = cube_252_output_limb_9_col113;
        values[13] = combination_limb_0_col125;
        values[14] = combination_limb_1_col126;
        values[15] = combination_limb_2_col127;
        values[16] = combination_limb_3_col128;
        values[17] = combination_limb_4_col129;
        values[18] = combination_limb_5_col130;
        values[19] = combination_limb_6_col131;
        values[20] = combination_limb_7_col132;
        values[21] = combination_limb_8_col133;
        values[22] = combination_limb_9_col134;
        values[23] = cube_252_output_limb_0_col136;
        values[24] = cube_252_output_limb_1_col137;
        values[25] = cube_252_output_limb_2_col138;
        values[26] = cube_252_output_limb_3_col139;
        values[27] = cube_252_output_limb_4_col140;
        values[28] = cube_252_output_limb_5_col141;
        values[29] = cube_252_output_limb_6_col142;
        values[30] = cube_252_output_limb_7_col143;
        values[31] = cube_252_output_limb_8_col144;
        values[32] = cube_252_output_limb_9_col145;
        values[33] = combination_limb_0_col157;
        values[34] = combination_limb_1_col158;
        values[35] = combination_limb_2_col159;
        values[36] = combination_limb_3_col160;
        values[37] = combination_limb_4_col161;
        values[38] = combination_limb_5_col162;
        values[39] = combination_limb_6_col163;
        values[40] = combination_limb_7_col164;
        values[41] = combination_limb_8_col165;
        values[42] = combination_limb_9_col166;

        qm31 multiplicity = qm31{{neg(enabler), 0}, {0, 0}};
        cuda_evaluator.add_to_relation<43>(poseidon_eval->common_lookup_elements, multiplicity, values);
    }

    // Store results back to global memory
    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// Host function: Evaluates poseidon_3_partial_rounds_chain component
extern "C"
void evaluate_poseidon_3_partial_rounds_chain(
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

    Poseidon3PartialRoundsChain_Eval *device_poseidon_eval = cuda_malloc<Poseidon3PartialRoundsChain_Eval>(1);
    cuda_mem_copy_host_to_device<Poseidon3PartialRoundsChain_Eval>((Poseidon3PartialRoundsChain_Eval*)eval, device_poseidon_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);

    int block_dim = eval_domain_size < POSEIDON_3_PARTIAL_ROUNDS_CHAIN_THREAD_COUNT_MAX ? eval_domain_size : POSEIDON_3_PARTIAL_ROUNDS_CHAIN_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    // This component uses trace columns (trace1), not preprocessed columns (trace0)
    if (use_assert_evaluator) {
        evaluate_poseidon_3_partial_rounds_chain_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_poseidon_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_poseidon_3_partial_rounds_chain_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_poseidon_eval,
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

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_poseidon_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
