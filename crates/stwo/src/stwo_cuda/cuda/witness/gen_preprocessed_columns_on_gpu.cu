// CUDA kernels for generating preprocessed columns directly on GPU
// This avoids CPU->GPU transfer for deterministic columns:
// - Seq: Sequential numbers [0..2^n]
// - RangeCheck: Partitioned enumeration for range checks
// - BitwiseXor: XOR lookup tables

#include "fields.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include <cstdint>
#include <cstdio>

// ============================================================================
// Seq Column Generation
// ============================================================================
// Generates column with values [0, 1, 2, ..., 2^log_size - 1]

__global__ void gen_seq_column_kernel(
    m31* output,
    uint32_t n_elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    output[idx] = {idx};
}

extern "C" void gen_seq_column_on_gpu(
    m31* output,
    uint32_t log_size
) {
    const uint32_t BLOCK_SIZE = 256;
    uint32_t n_elements = 1u << log_size;
    uint32_t num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gen_seq_column_kernel<<<num_blocks, BLOCK_SIZE>>>(output, n_elements);
}

// ============================================================================
// RangeCheck Column Generation
// ============================================================================
// Generates partitioned enumeration for range check columns
// Example: bits_per_segment = [4, 3] generates:
//   column 0: [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,...]  (4-bit values)
//   column 1: [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,...]  (3-bit values)

__global__ void gen_range_check_columns_kernel(
    m31** output_columns,
    uint32_t n_columns,
    const uint32_t* bits_per_segment,
    uint32_t n_segments,
    uint32_t n_elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    // Partition idx into segments
    uint32_t value = idx;
    for (int seg = n_segments - 1; seg >= 0; seg--) {
        uint32_t bits = bits_per_segment[seg];
        uint32_t mask = (1u << bits) - 1;
        uint32_t segment_value = value & mask;
        output_columns[seg][idx] = {segment_value};
        value >>= bits;
    }
}

extern "C" void gen_range_check_columns_on_gpu(
    m31** output_columns,
    uint32_t n_columns,
    const uint32_t* bits_per_segment,
    uint32_t n_segments
) {
    const uint32_t BLOCK_SIZE = 256;

    // Calculate total bits to determine number of elements
    uint32_t total_bits = 0;
    for (uint32_t i = 0; i < n_segments; i++) {
        total_bits += bits_per_segment[i];
    }
    uint32_t n_elements = 1u << total_bits;
    uint32_t num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Copy bits_per_segment to device (const_cast needed as clone_to_device requires non-const)
    uint32_t* d_bits_per_segment = clone_to_device<uint32_t>(
        const_cast<uint32_t*>(bits_per_segment), n_segments);

    // Copy column pointers to device
    m31** d_columns = clone_to_device<m31*>(output_columns, n_columns);

    gen_range_check_columns_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_columns, n_columns, d_bits_per_segment, n_segments, n_elements
    );

    cuda_free_memory(d_bits_per_segment);
    cuda_free_memory(d_columns);
}

// ============================================================================
// BitwiseXor Column Generation
// ============================================================================
// Generates XOR lookup table with 3 columns:
// - Column 0: a values (0 to 2^n_bits - 1, each repeated 2^n_bits times)
// - Column 1: b values (0 to 2^n_bits - 1, cycling)
// - Column 2: a ^ b

__global__ void gen_bitwise_xor_columns_kernel(
    m31** output_columns,
    uint32_t n_bits,
    uint32_t n_elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    uint32_t elements_per_a = 1u << n_bits;
    uint32_t a = idx / elements_per_a;
    uint32_t b = idx % elements_per_a;

    output_columns[0][idx] = {a};
    output_columns[1][idx] = {b};
    output_columns[2][idx] = {a ^ b};
}

extern "C" void gen_bitwise_xor_columns_on_gpu(
    m31** output_columns,
    uint32_t n_bits
) {
    const uint32_t BLOCK_SIZE = 256;

    // Total elements = (2^n_bits)^2 = 2^(2*n_bits)
    uint32_t n_elements = 1u << (2 * n_bits);
    uint32_t num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Copy column pointers to device
    m31** d_columns = clone_to_device<m31*>(output_columns, 3);

    gen_bitwise_xor_columns_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_columns, n_bits, n_elements
    );

    cuda_free_memory(d_columns);
}

// ============================================================================
// PedersenPoints Column Generation (alternative to full table generation)
// ============================================================================
// This is handled by gen_pedersen_table_on_gpu.cu for the full table
// Individual column generation can use the same output format
