/*
============================================
PoseidonFullRoundChain CUDA Evaluator
============================================

Component: PoseidonFullRoundChain
translated from: cairo-air/src/components/poseidon_full_round_chain.rs
AIR version: 54d95c0d

Functionality:
- Evaluates one full round of Poseidon hash permutation
- Processes 3 field elements through cubing and linear combination
- Updates round counter and applies round keys

Inputs/Outputs:
- Trace Columns: 126
  - input_limb[0-31]: 32 input state limbs
  - cube_252_output_limb[32-61]: 30 cubed value limbs (3 groups of 10)
  - poseidon_round_keys_output_limb[62-91]: 30 round key limbs
  - combination_limb[92-124]: 33 combination result limbs (3 sets of 10 + p_coef)
  - enabler: Boolean flag (column 125)

Constraint Logic:
- 1 enabler boolean constraint: enabler^2 - enabler = 0
- 3 Cube252 lookups (cubing 3 field elements)
- 1 PoseidonRoundKeys lookup (retrieving round keys)
- 3 LinearCombination calls (adding cubed values and round keys)
- 2 PoseidonFullRoundChain self-lookups (chaining rounds)

Relation Lookups:
- Cube252: 3 uses (cubing input[2-11], [12-21], [22-31])
- PoseidonRoundKeys: 1 use (retrieving 30 round keys)
- RangeCheck_3_3_3_3_3: 6 uses (via 3 LinearCombination calls)
- PoseidonFullRoundChain: 2 uses (self-chaining with ±enabler)

============================================
*/

#include <cstdio>
#include <vector>

#include "evaluate_poseidon_full_round_chain.cuh"
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_common.cuh"
#include "relations.cuh"
#include "evaluate_linear_combination_n_4_coefs_3_1_1_1.cuh"
#include "evaluate_linear_combination_n_4_coefs_1_m1_1_1.cuh"
#include "evaluate_linear_combination_n_4_coefs_1_1_m2_1.cuh"

#define POSEIDON_FULL_ROUND_CHAIN_THREAD_COUNT_MAX 256

// Pre-kernel: Evaluates constraints and relation lookups for poseidon_full_round_chain
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_poseidon_full_round_chain_pre_kernel(
    qm31 *numerators,
    m31 **trace_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PoseidonFullRoundChain_Eval *poseidon_eval,
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

    // Read all 126 trace columns
    // Columns 0-31: input_limb
    m31 input_limb_0 = cuda_evaluator.next_trace_mask();
    m31 input_limb_1 = cuda_evaluator.next_trace_mask();
    m31 input_limb_2 = cuda_evaluator.next_trace_mask();
    m31 input_limb_3 = cuda_evaluator.next_trace_mask();
    m31 input_limb_4 = cuda_evaluator.next_trace_mask();
    m31 input_limb_5 = cuda_evaluator.next_trace_mask();
    m31 input_limb_6 = cuda_evaluator.next_trace_mask();
    m31 input_limb_7 = cuda_evaluator.next_trace_mask();
    m31 input_limb_8 = cuda_evaluator.next_trace_mask();
    m31 input_limb_9 = cuda_evaluator.next_trace_mask();
    m31 input_limb_10 = cuda_evaluator.next_trace_mask();
    m31 input_limb_11 = cuda_evaluator.next_trace_mask();
    m31 input_limb_12 = cuda_evaluator.next_trace_mask();
    m31 input_limb_13 = cuda_evaluator.next_trace_mask();
    m31 input_limb_14 = cuda_evaluator.next_trace_mask();
    m31 input_limb_15 = cuda_evaluator.next_trace_mask();
    m31 input_limb_16 = cuda_evaluator.next_trace_mask();
    m31 input_limb_17 = cuda_evaluator.next_trace_mask();
    m31 input_limb_18 = cuda_evaluator.next_trace_mask();
    m31 input_limb_19 = cuda_evaluator.next_trace_mask();
    m31 input_limb_20 = cuda_evaluator.next_trace_mask();
    m31 input_limb_21 = cuda_evaluator.next_trace_mask();
    m31 input_limb_22 = cuda_evaluator.next_trace_mask();
    m31 input_limb_23 = cuda_evaluator.next_trace_mask();
    m31 input_limb_24 = cuda_evaluator.next_trace_mask();
    m31 input_limb_25 = cuda_evaluator.next_trace_mask();
    m31 input_limb_26 = cuda_evaluator.next_trace_mask();
    m31 input_limb_27 = cuda_evaluator.next_trace_mask();
    m31 input_limb_28 = cuda_evaluator.next_trace_mask();
    m31 input_limb_29 = cuda_evaluator.next_trace_mask();
    m31 input_limb_30 = cuda_evaluator.next_trace_mask();
    m31 input_limb_31 = cuda_evaluator.next_trace_mask();

    // Columns 32-61: cube_252_output_limb (3 groups of 10 limbs)
    m31 cube_252_output_limb_0_col32 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_1_col33 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_2_col34 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_3_col35 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_4_col36 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_5_col37 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_6_col38 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_7_col39 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_8_col40 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_9_col41 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_0_col42 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_1_col43 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_2_col44 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_3_col45 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_4_col46 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_5_col47 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_6_col48 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_7_col49 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_8_col50 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_9_col51 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_0_col52 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_1_col53 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_2_col54 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_3_col55 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_4_col56 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_5_col57 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_6_col58 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_7_col59 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_8_col60 = cuda_evaluator.next_trace_mask();
    m31 cube_252_output_limb_9_col61 = cuda_evaluator.next_trace_mask();

    // Columns 62-91: poseidon_round_keys_output_limb (30 limbs)
    m31 poseidon_round_keys_output_limb_0_col62 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_1_col63 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_2_col64 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_3_col65 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_4_col66 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_5_col67 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_6_col68 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_7_col69 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_8_col70 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_9_col71 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_10_col72 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_11_col73 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_12_col74 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_13_col75 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_14_col76 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_15_col77 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_16_col78 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_17_col79 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_18_col80 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_19_col81 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_20_col82 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_21_col83 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_22_col84 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_23_col85 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_24_col86 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_25_col87 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_26_col88 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_27_col89 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_28_col90 = cuda_evaluator.next_trace_mask();
    m31 poseidon_round_keys_output_limb_29_col91 = cuda_evaluator.next_trace_mask();

    // Columns 92-124: combination results (3 sets of 10 limbs + p_coef)
    // First combination: columns 92-102
    m31 combination_limb_0_col92 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col93 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col94 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col95 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col96 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col97 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col98 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col99 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col100 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col101 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col102 = cuda_evaluator.next_trace_mask();

    // Second combination: columns 103-113
    m31 combination_limb_0_col103 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_1_col104 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_2_col105 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_3_col106 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_4_col107 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_5_col108 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_6_col109 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_7_col110 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_8_col111 = cuda_evaluator.next_trace_mask();
    m31 combination_limb_9_col112 = cuda_evaluator.next_trace_mask();
    m31 p_coef_col113 = cuda_evaluator.next_trace_mask();

    // Third combination: columns 114-124
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

    // Column 125: enabler
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Constraint: enabler^2 - enabler = 0 (boolean constraint)
    m31 enabler_constraint = sub(mul(enabler, enabler), enabler);
    cuda_evaluator.add_constraint(enabler_constraint);

    // ========================================
    // Cube252 Lookups (3 uses)
    // ========================================

    // First Cube252 lookup: input[2-11] → cube_output[32-41]
    {
        m31 values[21];
        values[0] = CUBE_252_RELATION_ID;
        values[1] = input_limb_2;
        values[2] = input_limb_3;
        values[3] = input_limb_4;
        values[4] = input_limb_5;
        values[5] = input_limb_6;
        values[6] = input_limb_7;
        values[7] = input_limb_8;
        values[8] = input_limb_9;
        values[9] = input_limb_10;
        values[10] = input_limb_11;
        values[11] = cube_252_output_limb_0_col32;
        values[12] = cube_252_output_limb_1_col33;
        values[13] = cube_252_output_limb_2_col34;
        values[14] = cube_252_output_limb_3_col35;
        values[15] = cube_252_output_limb_4_col36;
        values[16] = cube_252_output_limb_5_col37;
        values[17] = cube_252_output_limb_6_col38;
        values[18] = cube_252_output_limb_7_col39;
        values[19] = cube_252_output_limb_8_col40;
        values[20] = cube_252_output_limb_9_col41;

        cuda_evaluator.add_to_relation<21>(poseidon_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Second Cube252 lookup: input[12-21] → cube_output[42-51]
    {
        m31 values[21];
        values[0] = CUBE_252_RELATION_ID;
        values[1] = input_limb_12;
        values[2] = input_limb_13;
        values[3] = input_limb_14;
        values[4] = input_limb_15;
        values[5] = input_limb_16;
        values[6] = input_limb_17;
        values[7] = input_limb_18;
        values[8] = input_limb_19;
        values[9] = input_limb_20;
        values[10] = input_limb_21;
        values[11] = cube_252_output_limb_0_col42;
        values[12] = cube_252_output_limb_1_col43;
        values[13] = cube_252_output_limb_2_col44;
        values[14] = cube_252_output_limb_3_col45;
        values[15] = cube_252_output_limb_4_col46;
        values[16] = cube_252_output_limb_5_col47;
        values[17] = cube_252_output_limb_6_col48;
        values[18] = cube_252_output_limb_7_col49;
        values[19] = cube_252_output_limb_8_col50;
        values[20] = cube_252_output_limb_9_col51;

        cuda_evaluator.add_to_relation<21>(poseidon_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Third Cube252 lookup: input[22-31] → cube_output[52-61]
    {
        m31 values[21];
        values[0] = CUBE_252_RELATION_ID;
        values[1] = input_limb_22;
        values[2] = input_limb_23;
        values[3] = input_limb_24;
        values[4] = input_limb_25;
        values[5] = input_limb_26;
        values[6] = input_limb_27;
        values[7] = input_limb_28;
        values[8] = input_limb_29;
        values[9] = input_limb_30;
        values[10] = input_limb_31;
        values[11] = cube_252_output_limb_0_col52;
        values[12] = cube_252_output_limb_1_col53;
        values[13] = cube_252_output_limb_2_col54;
        values[14] = cube_252_output_limb_3_col55;
        values[15] = cube_252_output_limb_4_col56;
        values[16] = cube_252_output_limb_5_col57;
        values[17] = cube_252_output_limb_6_col58;
        values[18] = cube_252_output_limb_7_col59;
        values[19] = cube_252_output_limb_8_col60;
        values[20] = cube_252_output_limb_9_col61;

        cuda_evaluator.add_to_relation<21>(poseidon_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // ========================================
    // PoseidonRoundKeys Lookup (1 use)
    // ========================================

    {
        m31 values[32];
        values[0] = POSEIDON_ROUND_KEYS_RELATION_ID;
        values[1] = input_limb_1;  // Round index
        values[2] = poseidon_round_keys_output_limb_0_col62;
        values[3] = poseidon_round_keys_output_limb_1_col63;
        values[4] = poseidon_round_keys_output_limb_2_col64;
        values[5] = poseidon_round_keys_output_limb_3_col65;
        values[6] = poseidon_round_keys_output_limb_4_col66;
        values[7] = poseidon_round_keys_output_limb_5_col67;
        values[8] = poseidon_round_keys_output_limb_6_col68;
        values[9] = poseidon_round_keys_output_limb_7_col69;
        values[10] = poseidon_round_keys_output_limb_8_col70;
        values[11] = poseidon_round_keys_output_limb_9_col71;
        values[12] = poseidon_round_keys_output_limb_10_col72;
        values[13] = poseidon_round_keys_output_limb_11_col73;
        values[14] = poseidon_round_keys_output_limb_12_col74;
        values[15] = poseidon_round_keys_output_limb_13_col75;
        values[16] = poseidon_round_keys_output_limb_14_col76;
        values[17] = poseidon_round_keys_output_limb_15_col77;
        values[18] = poseidon_round_keys_output_limb_16_col78;
        values[19] = poseidon_round_keys_output_limb_17_col79;
        values[20] = poseidon_round_keys_output_limb_18_col80;
        values[21] = poseidon_round_keys_output_limb_19_col81;
        values[22] = poseidon_round_keys_output_limb_20_col82;
        values[23] = poseidon_round_keys_output_limb_21_col83;
        values[24] = poseidon_round_keys_output_limb_22_col84;
        values[25] = poseidon_round_keys_output_limb_23_col85;
        values[26] = poseidon_round_keys_output_limb_24_col86;
        values[27] = poseidon_round_keys_output_limb_25_col87;
        values[28] = poseidon_round_keys_output_limb_26_col88;
        values[29] = poseidon_round_keys_output_limb_27_col89;
        values[30] = poseidon_round_keys_output_limb_28_col90;
        values[31] = poseidon_round_keys_output_limb_29_col91;

        cuda_evaluator.add_to_relation<32>(poseidon_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // ========================================
    // LinearCombination Subroutine Calls (3 uses)
    // ========================================

    // First LinearCombination: LinearCombinationN4Coefs3111
    // Input: cube_outputs[32-61] + round_keys[62-71]
    // Output: combination[92-101] + p_coef[102]
    linear_combination_n_4_coefs_3_1_1_1_evaluate(
        // 40 input limbs
        cube_252_output_limb_0_col32, cube_252_output_limb_1_col33,
        cube_252_output_limb_2_col34, cube_252_output_limb_3_col35,
        cube_252_output_limb_4_col36, cube_252_output_limb_5_col37,
        cube_252_output_limb_6_col38, cube_252_output_limb_7_col39,
        cube_252_output_limb_8_col40, cube_252_output_limb_9_col41,
        cube_252_output_limb_0_col42, cube_252_output_limb_1_col43,
        cube_252_output_limb_2_col44, cube_252_output_limb_3_col45,
        cube_252_output_limb_4_col46, cube_252_output_limb_5_col47,
        cube_252_output_limb_6_col48, cube_252_output_limb_7_col49,
        cube_252_output_limb_8_col50, cube_252_output_limb_9_col51,
        cube_252_output_limb_0_col52, cube_252_output_limb_1_col53,
        cube_252_output_limb_2_col54, cube_252_output_limb_3_col55,
        cube_252_output_limb_4_col56, cube_252_output_limb_5_col57,
        cube_252_output_limb_6_col58, cube_252_output_limb_7_col59,
        cube_252_output_limb_8_col60, cube_252_output_limb_9_col61,
        poseidon_round_keys_output_limb_0_col62, poseidon_round_keys_output_limb_1_col63,
        poseidon_round_keys_output_limb_2_col64, poseidon_round_keys_output_limb_3_col65,
        poseidon_round_keys_output_limb_4_col66, poseidon_round_keys_output_limb_5_col67,
        poseidon_round_keys_output_limb_6_col68, poseidon_round_keys_output_limb_7_col69,
        poseidon_round_keys_output_limb_8_col70, poseidon_round_keys_output_limb_9_col71,
        // 10 combination limbs
        combination_limb_0_col92, combination_limb_1_col93,
        combination_limb_2_col94, combination_limb_3_col95,
        combination_limb_4_col96, combination_limb_5_col97,
        combination_limb_6_col98, combination_limb_7_col99,
        combination_limb_8_col100, combination_limb_9_col101,
        // p_coef
        p_coef_col102,
        // Lookup elements
        poseidon_eval->common_lookup_elements,
        // Evaluator
        &cuda_evaluator
    );

    // Second LinearCombination: LinearCombinationN4Coefs1M111
    // Input: cube_outputs[32-61] + round_keys[72-81]
    // Output: combination[103-112] + p_coef[113]
    linear_combination_n_4_coefs_1_m1_1_1_evaluate(
        // 40 input limbs
        cube_252_output_limb_0_col32, cube_252_output_limb_1_col33,
        cube_252_output_limb_2_col34, cube_252_output_limb_3_col35,
        cube_252_output_limb_4_col36, cube_252_output_limb_5_col37,
        cube_252_output_limb_6_col38, cube_252_output_limb_7_col39,
        cube_252_output_limb_8_col40, cube_252_output_limb_9_col41,
        cube_252_output_limb_0_col42, cube_252_output_limb_1_col43,
        cube_252_output_limb_2_col44, cube_252_output_limb_3_col45,
        cube_252_output_limb_4_col46, cube_252_output_limb_5_col47,
        cube_252_output_limb_6_col48, cube_252_output_limb_7_col49,
        cube_252_output_limb_8_col50, cube_252_output_limb_9_col51,
        cube_252_output_limb_0_col52, cube_252_output_limb_1_col53,
        cube_252_output_limb_2_col54, cube_252_output_limb_3_col55,
        cube_252_output_limb_4_col56, cube_252_output_limb_5_col57,
        cube_252_output_limb_6_col58, cube_252_output_limb_7_col59,
        cube_252_output_limb_8_col60, cube_252_output_limb_9_col61,
        poseidon_round_keys_output_limb_10_col72, poseidon_round_keys_output_limb_11_col73,
        poseidon_round_keys_output_limb_12_col74, poseidon_round_keys_output_limb_13_col75,
        poseidon_round_keys_output_limb_14_col76, poseidon_round_keys_output_limb_15_col77,
        poseidon_round_keys_output_limb_16_col78, poseidon_round_keys_output_limb_17_col79,
        poseidon_round_keys_output_limb_18_col80, poseidon_round_keys_output_limb_19_col81,
        // 10 combination limbs
        combination_limb_0_col103, combination_limb_1_col104,
        combination_limb_2_col105, combination_limb_3_col106,
        combination_limb_4_col107, combination_limb_5_col108,
        combination_limb_6_col109, combination_limb_7_col110,
        combination_limb_8_col111, combination_limb_9_col112,
        // p_coef
        p_coef_col113,
        // Lookup elements
        poseidon_eval->common_lookup_elements,
        // Evaluator
        &cuda_evaluator
    );

    // Third LinearCombination: LinearCombinationN4Coefs11M21
    // Input: cube_outputs[32-61] + round_keys[82-91]
    // Output: combination[114-123] + p_coef[124]
    linear_combination_n_4_coefs_1_1_m2_1_evaluate(
        // 40 input limbs
        cube_252_output_limb_0_col32, cube_252_output_limb_1_col33,
        cube_252_output_limb_2_col34, cube_252_output_limb_3_col35,
        cube_252_output_limb_4_col36, cube_252_output_limb_5_col37,
        cube_252_output_limb_6_col38, cube_252_output_limb_7_col39,
        cube_252_output_limb_8_col40, cube_252_output_limb_9_col41,
        cube_252_output_limb_0_col42, cube_252_output_limb_1_col43,
        cube_252_output_limb_2_col44, cube_252_output_limb_3_col45,
        cube_252_output_limb_4_col46, cube_252_output_limb_5_col47,
        cube_252_output_limb_6_col48, cube_252_output_limb_7_col49,
        cube_252_output_limb_8_col50, cube_252_output_limb_9_col51,
        cube_252_output_limb_0_col52, cube_252_output_limb_1_col53,
        cube_252_output_limb_2_col54, cube_252_output_limb_3_col55,
        cube_252_output_limb_4_col56, cube_252_output_limb_5_col57,
        cube_252_output_limb_6_col58, cube_252_output_limb_7_col59,
        cube_252_output_limb_8_col60, cube_252_output_limb_9_col61,
        poseidon_round_keys_output_limb_20_col82, poseidon_round_keys_output_limb_21_col83,
        poseidon_round_keys_output_limb_22_col84, poseidon_round_keys_output_limb_23_col85,
        poseidon_round_keys_output_limb_24_col86, poseidon_round_keys_output_limb_25_col87,
        poseidon_round_keys_output_limb_26_col88, poseidon_round_keys_output_limb_27_col89,
        poseidon_round_keys_output_limb_28_col90, poseidon_round_keys_output_limb_29_col91,
        // 10 combination limbs
        combination_limb_0_col114, combination_limb_1_col115,
        combination_limb_2_col116, combination_limb_3_col117,
        combination_limb_4_col118, combination_limb_5_col119,
        combination_limb_6_col120, combination_limb_7_col121,
        combination_limb_8_col122, combination_limb_9_col123,
        // p_coef
        p_coef_col124,
        // Lookup elements
        poseidon_eval->common_lookup_elements,
        // Evaluator
        &cuda_evaluator
    );

    // ========================================
    // PoseidonFullRoundChain Self-Lookups (2 uses)
    // ========================================

    // First self-lookup: positive multiplicity (enabler)
    // Represents input state at current round
    {
        m31 values[33];
        values[0] = POSEIDON_FULL_ROUND_CHAIN_RELATION_ID;
        values[1] = input_limb_0;
        values[2] = input_limb_1;
        values[3] = input_limb_2;
        values[4] = input_limb_3;
        values[5] = input_limb_4;
        values[6] = input_limb_5;
        values[7] = input_limb_6;
        values[8] = input_limb_7;
        values[9] = input_limb_8;
        values[10] = input_limb_9;
        values[11] = input_limb_10;
        values[12] = input_limb_11;
        values[13] = input_limb_12;
        values[14] = input_limb_13;
        values[15] = input_limb_14;
        values[16] = input_limb_15;
        values[17] = input_limb_16;
        values[18] = input_limb_17;
        values[19] = input_limb_18;
        values[20] = input_limb_19;
        values[21] = input_limb_20;
        values[22] = input_limb_21;
        values[23] = input_limb_22;
        values[24] = input_limb_23;
        values[25] = input_limb_24;
        values[26] = input_limb_25;
        values[27] = input_limb_26;
        values[28] = input_limb_27;
        values[29] = input_limb_28;
        values[30] = input_limb_29;
        values[31] = input_limb_30;
        values[32] = input_limb_31;

        qm31 multiplicity = qm31{{enabler, 0}, {0, 0}};
        cuda_evaluator.add_to_relation<33>(poseidon_eval->common_lookup_elements, multiplicity, values);
    }

    // Second self-lookup: negative multiplicity (-enabler)
    // Represents output state after this round (input to next round)
    {
        m31 values[33];
        values[0] = POSEIDON_FULL_ROUND_CHAIN_RELATION_ID;
        values[1] = input_limb_0;  // State counter unchanged
        values[2] = add(input_limb_1, M31_1);  // Round counter incremented
        values[3] = combination_limb_0_col92;
        values[4] = combination_limb_1_col93;
        values[5] = combination_limb_2_col94;
        values[6] = combination_limb_3_col95;
        values[7] = combination_limb_4_col96;
        values[8] = combination_limb_5_col97;
        values[9] = combination_limb_6_col98;
        values[10] = combination_limb_7_col99;
        values[11] = combination_limb_8_col100;
        values[12] = combination_limb_9_col101;
        values[13] = combination_limb_0_col103;
        values[14] = combination_limb_1_col104;
        values[15] = combination_limb_2_col105;
        values[16] = combination_limb_3_col106;
        values[17] = combination_limb_4_col107;
        values[18] = combination_limb_5_col108;
        values[19] = combination_limb_6_col109;
        values[20] = combination_limb_7_col110;
        values[21] = combination_limb_8_col111;
        values[22] = combination_limb_9_col112;
        values[23] = combination_limb_0_col114;
        values[24] = combination_limb_1_col115;
        values[25] = combination_limb_2_col116;
        values[26] = combination_limb_3_col117;
        values[27] = combination_limb_4_col118;
        values[28] = combination_limb_5_col119;
        values[29] = combination_limb_6_col120;
        values[30] = combination_limb_7_col121;
        values[31] = combination_limb_8_col122;
        values[32] = combination_limb_9_col123;

        qm31 multiplicity = qm31{{neg(enabler), 0}, {0, 0}};
        cuda_evaluator.add_to_relation<33>(poseidon_eval->common_lookup_elements, multiplicity, values);
    }

    // Store results back to global memory
    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// Host function: Evaluates poseidon_full_round_chain component
extern "C"
void evaluate_poseidon_full_round_chain(
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

    PoseidonFullRoundChain_Eval *device_poseidon_eval = cuda_malloc<PoseidonFullRoundChain_Eval>(1);
    cuda_mem_copy_host_to_device<PoseidonFullRoundChain_Eval>((PoseidonFullRoundChain_Eval*)eval, device_poseidon_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);

    int block_dim = eval_domain_size < POSEIDON_FULL_ROUND_CHAIN_THREAD_COUNT_MAX ? eval_domain_size : POSEIDON_FULL_ROUND_CHAIN_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    // This component uses trace columns (trace1), not preprocessed columns (trace0)
    if (use_assert_evaluator) {
        evaluate_poseidon_full_round_chain_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
        evaluate_poseidon_full_round_chain_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
