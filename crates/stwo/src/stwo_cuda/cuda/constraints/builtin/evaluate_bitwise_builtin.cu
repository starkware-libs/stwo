// ============================================================================
// Bitwise Builtin CUDA Evaluator
// ============================================================================
//
// CUDA version of Bitwise builtin AIR constraint evaluator
// translated from cairo-air/src/components/bitwise_builtin.rs
//
// ## Functionality Overview
// Bitwise builtin implements 252-bit integer bitwise operations (AND, OR, XOR)
//
// ## Data Structure
// - 89 trace columns:
// - op0: first operand (1 id + 28 9-bit limbs = 252 bits)
// - op1: second operand (1 id + 28 9-bit limbs = 252 bits)
// - xor: XOR result (28 9-bit limbs)
// - and_id, xor_id, or_id: result memory IDs (3 columns)
//
// ## Memory Layout
// Each bitwise operation occupies 5 consecutive memory cells:
//   [base + seq*5 + 0]: op0 (252-bit)
//   [base + seq*5 + 1]: op1 (252-bit)
//   [base + seq*5 + 2]: AND result (252-bit)
//   [base + seq*5 + 3]: XOR result (252-bit)
//   [base + seq*5 + 4]: OR result (252-bit)
//
// ## Constraint Logic
// 1. Verify memory read of op0 and op1 (MemoryAddressToId + MemoryIdToBig)
// 2. Verify XOR operation per limb (VerifyBitwiseXor_9 × 27 + VerifyBitwiseXor_8 × 1)
// 3. Compute AND = (op0 + op1 - XOR) / 2
// 4. Compute OR = AND + XOR
// 5. Verify memory write of AND, XOR, OR results
//
// ## Relation Lookups
// - MemoryAddressToId: 5 times (op0, op1, and, xor, or)
// - MemoryIdToBig: 5 times (each memory access corresponds to a big number)
// - VerifyBitwiseXor_9: 27 times (first 27 limbs, each 9 bits)
// - VerifyBitwiseXor_8: 1 time (last limb, 8 bits)
//
// ## Key Algorithms
// - XOR via table lookup verification: verify_bitwise_xor_N(op0_limb, op1_limb, xor_limb)
// - AND via algebraic computation: and_limb = (op0_limb + op1_limb - xor_limb) / 2
// - OR via algebraic computation: or_limb = and_limb + xor_limb
// - In M31 field, division by 2 is equivalent to multiplication by 2^30 = 1073741824
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
#include "evaluate_bitwise_builtin.cuh"
#include "evaluate_common.cuh"

// ============================================================================
// Bitwise Pre-Kernel: Main Constraint Evaluation
// ============================================================================
// This kernel is responsible for:
// 1. Reading all 89 trace columns
// 2. Executing memory lookup verification (reading op0, op1)
// 3. Executing per-limb XOR verification (table lookup)
// 4. Computing AND result (algebraic computation)
// 5. Executing memory lookup verification (writing and, xor, or)
// 6. Accumulating all relations into intermediate_fractions
template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_bitwise_builtin_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    BitwiseBuiltin_Eval *bitwise_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    // ===================== Constant Definitions =====================
    // M31_1073741824 = 2^30 in M31 field, for division by 2
    // Because 2 * 2^30 ≡ 1 (mod P), where P = 2^31 - 1
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_5 = m31(5); // Each bitwise instance occupies 5 memory cells
    const m31 M31_1073741824 = m31(1073741824); // 2^30, for implementing division by 2 in M31 field

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

    // ===================== Preprocessed Column: Sequence Number =====================
    // seq is used to compute memory address offset for each instance
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

    // ===================== Read 89 Trace Columns =====================
    // Read all trace columns in the same order as CPU bitwise_builtin::Eval::evaluate
    //
    // Column structure:
    // 0: op0_id (first operand memory ID)
    // 1-28: op0_limbs[0-27] (first operand 28 limbs, each 9 bits)
    // 29: op1_id (second operand memory ID)
    // 30-57: op1_limbs[0-27] (second operand 28 limbs)
    // 58-85: xor_limbs[0-27] (XOR result 28 limbs)
    // 86: and_id (AND result memory ID)
    // 87: xor_id (XOR result memory ID)
    // 88: or_id (OR result memory ID)
    //
    // Note: The first 27 limbs are full 9-bit, the last limb (27) is 8-bit
    // Because 27 × 9 + 8 = 251, but we need 252 bits, so actually it's 28 × 9 = 252

    // column 0-28: op0 (memory id + 28 limbs = 252 bits)
    m31 op0_id = cuda_evaluator1.next_trace_mask();
    m31 op0_limbs[28];
    for (int i = 0; i < 28; i++) {
        op0_limbs[i] = cuda_evaluator1.next_trace_mask();
    }

    // column 29-57: op1 (memory id + 28 limbs = 252 bits)
    m31 op1_id = cuda_evaluator1.next_trace_mask();
    m31 op1_limbs[28];
    for (int i = 0; i < 28; i++) {
        op1_limbs[i] = cuda_evaluator1.next_trace_mask();
    }

    // column 58-85: XOR result (28 limbs)
    m31 xor_limbs[28];
    for (int i = 0; i < 28; i++) {
        xor_limbs[i] = cuda_evaluator1.next_trace_mask();
    }

    // column 86-88: result memory IDs
    m31 and_id = cuda_evaluator1.next_trace_mask();
    m31 xor_id = cuda_evaluator1.next_trace_mask();
    m31 or_id = cuda_evaluator1.next_trace_mask();

    // ===================== op0 Memory Lookup (First Operand) =====================
    // Memory address: bitwise_builtin_segment_start + seq * 5
    // Verify reading 252-bit number from specified address to op0
    {
        m31 seq_times_5 = mul(seq, M31_5);
        m31 op0_address = add(m31(bitwise_eval->Claim.bitwise_builtin_segment_start), seq_times_5);

        // Debug: Print op0_address for row 0
        if (row == 0) {
            printf("[CUDA bitwise_builtin pre-kernel] row=0: bitwise_builtin_segment_start=%u, seq=%u, seq*5=%u, op0_address=%u\n",
                bitwise_eval->Claim.bitwise_builtin_segment_start, seq, seq_times_5, op0_address);
        }

        // 1. MemoryAddressToId lookup: verify address -> ID mapping
        // Ensure memory cell at op0_address has ID op0_id
        m31 addr_values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, op0_address, op0_id};
        cuda_evaluator1.add_to_relation<3>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, addr_values);

        // Debug: Print first fraction for row 0
        if (row == 0) {
            Fraction first_frac = intermediate_fractions[0];
            printf("[CUDA bitwise_builtin pre-kernel] row=0: first fraction num=(%u + %ui) + (%u + %ui)u, denom=(%u + %ui) + (%u + %ui)u\n",
                first_frac.numerator.a.a, first_frac.numerator.a.b, first_frac.numerator.b.a, first_frac.numerator.b.b,
                first_frac.denominator.a.a, first_frac.denominator.a.b, first_frac.denominator.b.a, first_frac.denominator.b.b);
        }

        // 2. MemoryIdToBig lookup: verify ID -> Big Number mapping
        // Ensure memory cell op0_id contains 252-bit number composed of 28 limbs
        m31 id_big_values[30];
        id_big_values[0] = MEMORY_ID_TO_BIG_RELATION_ID;
        id_big_values[1] = op0_id;
        for (int i = 0; i < 28; i++) {
            id_big_values[i + 2] = op0_limbs[i];
        }
        cuda_evaluator1.add_to_relation<30>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, id_big_values);
    }

    // ===================== op1 Memory Lookup (Second Operand) =====================
    // Memory address: bitwise_builtin_segment_start + seq * 5 + 1
    // Verify reading 252-bit number from specified address to op1
    {
        // Use proper M31 field arithmetic to avoid integer overflow
        m31 op1_address = add(add(m31(bitwise_eval->Claim.bitwise_builtin_segment_start), mul(seq, M31_5)), M31_1);

        // 1. MemoryAddressToId lookup
        m31 addr_values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, op1_address, op1_id};
        cuda_evaluator1.add_to_relation<3>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, addr_values);

        // 2. MemoryIdToBig lookup
        m31 id_big_values[30];
        id_big_values[0] = MEMORY_ID_TO_BIG_RELATION_ID;
        id_big_values[1] = op1_id;
        for (int i = 0; i < 28; i++) {
            id_big_values[i + 2] = op1_limbs[i];
        }
        cuda_evaluator1.add_to_relation<30>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, id_big_values);
    }

    // ===================== XOR Verification and AND Computation =====================
    //
    // Core bitwise algorithms:
    // 1. XOR via table lookup verification (pre-computed truth value table)
    // 2. AND via algebraic relation computation: AND = (op0 + op1 - XOR) / 2
    // 3. OR via algebraic relation computation: OR = AND + XOR
    //
    // Mathematical principle:
    // For any bits a, b:
    //   - a XOR b = a + b - 2(a AND b)  →  a AND b = (a + b - a XOR b) / 2
    //   - a OR b = a + b - (a AND b)     →  a OR b = (a AND b) + (a XOR b)
    //
    // At the limb level, these relations hold the same (as long as limbs don't overflow)
    m31 and_limbs[28];

    // Process first 27 limbs (each 9 bits)
    // Use VerifyBitwiseXor_9 table lookup to verify XOR correctness
    for (int i = 0; i < 27; i++) {
        // VerifyBitwiseXor_9 lookup: verify op0_limb XOR op1_limb = xor_limb
        m31 xor_values[4] = {VERIFY_BITWISE_XOR_9_RELATION_ID, op0_limbs[i], op1_limbs[i], xor_limbs[i]};
        cuda_evaluator1.add_to_relation<4>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, xor_values);

        // Compute AND: and_limb = (op0_limb + op1_limb - xor_limb) / 2
        // In M31 field, division by 2 is equivalent to multiplication by 2^30 = 1073741824
        and_limbs[i] = mul(M31_1073741824, sub(add(op0_limbs[i], op1_limbs[i]), xor_limbs[i]));
    }

    // Process last limb (limb 27), which is 8 bits instead of 9 bits
    // Because we need a total of 252 bits = 27×9 + 9, but the last one actually only needs 8 bits
    // Use VerifyBitwiseXor_8 table lookup
    {
        m31 xor_values[4] = {VERIFY_BITWISE_XOR_8_RELATION_ID, op0_limbs[27], op1_limbs[27], xor_limbs[27]};
        cuda_evaluator1.add_to_relation<4>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, xor_values);

        and_limbs[27] = mul(M31_1073741824, sub(add(op0_limbs[27], op1_limbs[27]), xor_limbs[27]));
    }

    // ===================== AND Result Memory Verification =====================
    // Memory address: bitwise_builtin_segment_start + seq * 5 + 2
    // Verify writing the computed AND result to memory
    {
        // Use proper M31 field arithmetic to avoid integer overflow
        m31 and_address = add(add(m31(bitwise_eval->Claim.bitwise_builtin_segment_start), mul(seq, M31_5)), M31_2);

        // 1. MemoryAddressToId lookup
        m31 addr_values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, and_address, and_id};
        cuda_evaluator1.add_to_relation<3>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, addr_values);

        // 2. MemoryIdToBig lookup
        // Verify memory cell and_id contains the AND result we computed (28 limbs)
        m31 id_big_values[30];
        id_big_values[0] = MEMORY_ID_TO_BIG_RELATION_ID;
        id_big_values[1] = and_id;
        for (int i = 0; i < 28; i++) {
            id_big_values[i + 2] = and_limbs[i];
        }
        cuda_evaluator1.add_to_relation<30>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, id_big_values);
    }

    // ===================== XOR Result Memory Verification =====================
    // Memory address: bitwise_builtin_segment_start + seq * 5 + 3
    // Verify writing XOR result to memory (XOR value already given in trace)
    {
        // Use proper M31 field arithmetic to avoid integer overflow
        m31 xor_address = add(add(m31(bitwise_eval->Claim.bitwise_builtin_segment_start), mul(seq, M31_5)), m31(3));

        // 1. MemoryAddressToId lookup
        m31 addr_values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, xor_address, xor_id};
        cuda_evaluator1.add_to_relation<3>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, addr_values);

        // 2. MemoryIdToBig lookup
        // Verify memory cell xor_id contains XOR result (28 limbs)
        m31 id_big_values[30];
        id_big_values[0] = MEMORY_ID_TO_BIG_RELATION_ID;
        id_big_values[1] = xor_id;
        for (int i = 0; i < 28; i++) {
            id_big_values[i + 2] = xor_limbs[i];
        }
        cuda_evaluator1.add_to_relation<30>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, id_big_values);
    }

    // ===================== OR Result Memory Verification =====================
    // Memory address: bitwise_builtin_segment_start + seq * 5 + 4
    // Verify writing OR result to memory, OR = AND + XOR
    {
        // Use proper M31 field arithmetic to avoid integer overflow
        m31 or_address = add(add(m31(bitwise_eval->Claim.bitwise_builtin_segment_start), mul(seq, M31_5)), m31(4));

        // 1. MemoryAddressToId lookup
        m31 addr_values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, or_address, or_id};
        cuda_evaluator1.add_to_relation<3>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, addr_values);

        // 2. MemoryIdToBig lookup
        // Verify memory cell or_id contains OR result
        // OR computed via algebraic relation: OR = AND + XOR
        m31 id_big_values[30];
        id_big_values[0] = MEMORY_ID_TO_BIG_RELATION_ID;
        id_big_values[1] = or_id;
        for (int i = 0; i < 28; i++) {
            id_big_values[i + 2] = add(and_limbs[i], xor_limbs[i]);
        }
        cuda_evaluator1.add_to_relation<30>(bitwise_eval->common_lookup_elements, qm31{M31_1, M31_0}, id_big_values);
    }

    // ===================== Complete Constraint Evaluation =====================
    // Save this row's constraint index
    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;

    // Debug: Print first row's values
    if (row == 0) {
        printf("[CUDA bitwise_builtin pre-kernel] row=0: constraint_index=%u, row_res=(%u + %ui) + (%u + %ui)u\n",
            cuda_evaluator1.constraint_index,
            cuda_evaluator1.row_res.a.a, cuda_evaluator1.row_res.a.b,
            cuda_evaluator1.row_res.b.a, cuda_evaluator1.row_res.b.b);
        printf("[CUDA bitwise_builtin pre-kernel] row=0: seq=%u, op0_id=%u, op1_id=%u\n",
            seq, op0_id, op1_id);
    }
}

// ============================================================================
// Bitwise Main Function: Orchestrates the Entire Evaluation Process
// ============================================================================
//
// This function coordinates the entire bitwise builtin evaluation process:
// 1. Prepare GPU memory and data
// 2. Call pre_kernel to evaluate constraints and relation lookups
// 3. Call post_kernel to complete logup accumulation
// 4. Call finalize_kernel to compute final quotients
// 5. Clean up GPU resources
extern "C"
void evaluate_bitwise_builtin(
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

    // ===================== GPU Memory Preparation =====================
    // Copy trace data to GPU
    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    // Allocate output buffer
    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    // Copy evaluator config to GPU
    BitwiseBuiltin_Eval *device_bitwise_eval = cuda_malloc<BitwiseBuiltin_Eval>(1);
    cuda_mem_copy_host_to_device<BitwiseBuiltin_Eval>((BitwiseBuiltin_Eval*)eval, device_bitwise_eval, 1);

    // Allocate intermediate buffer for logup fractions
    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);

    // Start timer
    timer global_timer;
    global_timer.start("evaluate_bitwise_builtin");

    // ===================== Kernel Configuration =====================
    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    // ===================== Pre-Kernel Execution =====================
    // Read trace columns, evaluate constraints, accumulate relation lookups
    if (use_assert_evaluator) {
        evaluate_bitwise_builtin_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_bitwise_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_bitwise_builtin_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_bitwise_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // ===================== Post-Kernel Execution =====================
    // Complete logup accumulation and constraint composition
    // Compute logup batching with optimized memory access
    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = logup_counts > 0 ? batching[logup_counts - 1] : 0;

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

    // ===================== Finalize-Kernel Execution =====================
    // Compute final quotients for FRI protocol
    generic_constraint_quotients_finalize_kernel<<<num_blocks, block_dim, 0, stream>>>(
        quotients_0,
        quotients_1,
        quotients_2,
        quotients_3,
        numerators,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size,
        g_should_accumulate_host // Global variable set by scheduler
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_bitwise_builtin");

    // ===================== Clean Up GPU Resources =====================
    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_bitwise_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}

