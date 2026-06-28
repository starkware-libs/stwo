cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Auto-detect GPU compute capability using nvidia-smi
        execute_process(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE GPU_COMPUTE_CAP
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE NVIDIA_SMI_RESULT
        )

        if (NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
                # Convert compute capability (e.g., "8.9") to architecture (e.g., "89")
                string(REGEX MATCH "^[0-9]+\\.[0-9]+" FIRST_GPU_CAP "${GPU_COMPUTE_CAP}")
                string(REPLACE "." "" GPU_ARCH "${FIRST_GPU_CAP}")
                set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCH}")
                message(STATUS "Auto-detected GPU architecture: ${GPU_ARCH} (compute capability: ${FIRST_GPU_CAP})")
        else()
                # Fallback to common architectures if detection fails
                # Support multiple GPU architectures:
                # 89 = Ada Lovelace (RTX 4090, etc.)
                # 100 = Blackwell (RTX 5090, etc.) - requires CUDA 12.8+
                # 120 = Blackwell B200, etc. - requires CUDA 13.0+
                set(CMAKE_CUDA_ARCHITECTURES "89;100;120")
                message(STATUS "Could not detect GPU, using default architectures: 89;100;120")
        endif()
endif ()

project(stwo_cuda LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CUH_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/*.cuh")
set(CUH_DIRS "")
foreach(file_path ${CUH_FILES})
    get_filename_component(dir ${file_path} DIRECTORY)
    list(APPEND CUH_DIRS ${dir})
endforeach()
list(REMOVE_DUPLICATES CUH_DIRS)

message(STATUS "Found cuh directories: ${CUH_DIRS}")

add_library(
        stwo_cuda STATIC
        # Core CUDA utilities
        batch_inverse.cu
        barycentric.cu
        utils.cu
        bit_reverse.cu
        eval_at_point.cu
        ifft.cu
        rfft.cu
        twiddles.cu
        poly_utils.cu
        fold_line.cu
        fold_circle_into_line.cu
        fri_utils.cu
        accumulate.cu
        fields.cu
        blake2s.cu
        grind_blake2s.cu
        poseidon252_starknet.cu
        quotients.cu
        gkr.cu
        mle.cu
        timer.cu
        evaluate_constraints.cu
        evaluate_common.cu
        evaluate_poseidon_constraint.cu
        evaluate_wide_fibonacci.cu
        prefix_sum.cu
        cuda_mem_pool.cu
        # stwo-cairo AIR components
        constraints/opcodes/evaluate_blake_compress_opcode.cu
        # Range checks
        constraints/range_check/evaluate_range_check_3_3_3_3_3.cu
        constraints/range_check/evaluate_range_check_3_6_6_3.cu
        constraints/range_check/evaluate_range_check_4_3.cu
        constraints/range_check/evaluate_range_check_4_4.cu
        constraints/range_check/evaluate_range_check_4_4_4_4.cu
        constraints/range_check/evaluate_range_check_6.cu
        constraints/range_check/evaluate_range_check_8.cu
        constraints/range_check/evaluate_range_check_9_9.cu
        constraints/range_check/evaluate_range_check_9_9_b.cu
        constraints/range_check/evaluate_range_check_9_9_c.cu
        constraints/range_check/evaluate_range_check_9_9_d.cu
        constraints/range_check/evaluate_range_check_9_9_e.cu
        constraints/range_check/evaluate_range_check_9_9_f.cu
        constraints/range_check/evaluate_range_check_9_9_g.cu
        constraints/range_check/evaluate_range_check_9_9_h.cu
        constraints/range_check/evaluate_range_check_11.cu
        constraints/range_check/evaluate_range_check_12.cu
        constraints/range_check/evaluate_range_check_18.cu
        constraints/range_check/evaluate_range_check_18_b.cu
        constraints/range_check/evaluate_range_check_19.cu
        constraints/range_check/evaluate_range_check_19_b.cu
        constraints/range_check/evaluate_range_check_19_c.cu
        constraints/range_check/evaluate_range_check_19_d.cu
        constraints/range_check/evaluate_range_check_19_e.cu
        constraints/range_check/evaluate_range_check_19_f.cu
        constraints/range_check/evaluate_range_check_19_g.cu
        constraints/range_check/evaluate_range_check_19_h.cu
        constraints/range_check/evaluate_range_check_20.cu
        constraints/range_check/evaluate_range_check_felt_252_width_27.cu
        constraints/range_check/evaluate_range_check_7_2_5.cu
        # Range check builtins
        constraints/builtin/evaluate_range_check_builtin_bits_96.cu
        constraints/builtin/evaluate_range_check_builtin_bits_128.cu
        constraints/evaluate_blake_g.cu
        # Builtins
        constraints/builtin/evaluate_add_mod_builtin.cu
        constraints/builtin/evaluate_bitwise_builtin.cu
        constraints/builtin/evaluate_mul_mod_builtin.cu
        constraints/builtin/evaluate_cube_252.cu
        constraints/builtin/evaluate_pedersen_builtin.cu
        constraints/builtin/evaluate_pedersen_builtin_narrow_windows.cu
        constraints/builtin/evaluate_poseidon_builtin.cu
        #pedersen
        constraints/pedersen/evaluate_pedersen_points_table.cu
        constraints/pedersen/evaluate_partial_ec_mul.cu
        constraints/pedersen/evaluate_pedersen_aggregator.cu
        constraints/pedersen/evaluate_partial_ec_mul_window_bits_18.cu
        constraints/pedersen/evaluate_pedersen_points_table_window_bits_18.cu
        constraints/pedersen/evaluate_pedersen_aggregator_window_bits_9.cu
        constraints/pedersen/evaluate_partial_ec_mul_window_bits_9.cu
        constraints/pedersen/evaluate_pedersen_points_table_window_bits_9.cu
        pedersen_table_init.cu
        pedersen_table_small_init.cu
        #poseidon
        constraints/poseidon/evaluate_poseidon_round_keys.cu
        constraints/poseidon/evaluate_poseidon_full_round_chain.cu
        constraints/poseidon/evaluate_poseidon_aggregator.cu
        constraints/poseidon/evaluate_poseidon_3_partial_rounds_chain.cu
        # Verify / bitwise XOR components
        constraints/verify/evaluate_verify_bitwise_xor_4.cu
        constraints/verify/evaluate_verify_bitwise_xor_7.cu
        constraints/verify/evaluate_verify_bitwise_xor_8.cu
        constraints/verify/evaluate_verify_bitwise_xor_8_b.cu
        constraints/verify/evaluate_verify_bitwise_xor_9.cu
        constraints/verify/evaluate_verify_bitwise_xor_12.cu
        constraints/verify/evaluate_verify_instruction.cu
        constraints/evaluate_triple_xor_32.cu
        constraints/evaluate_blake_round_sigma.cu
        constraints/evaluate_memory_address_to_id.cu
        constraints/evaluate_memory_id_to_big.cu
        constraints/evaluate_blake_round.cu
        #opcode
        constraints/opcodes/evaluate_add_opcode.cu
        constraints/opcodes/evaluate_add_ap_opcode.cu
        constraints/opcodes/evaluate_add_opcode_small.cu
        constraints/opcodes/evaluate_assert_eq_opcode.cu
        constraints/opcodes/evaluate_assert_eq_opcode_double_deref.cu
        constraints/opcodes/evaluate_assert_eq_opcode_imm.cu
        constraints/opcodes/evaluate_call_opcode.cu
        constraints/opcodes/evaluate_call_opcode_rel_imm.cu
        constraints/opcodes/evaluate_ret_opcode.cu
        constraints/opcodes/evaluate_jump_opcode.cu
        constraints/opcodes/evaluate_jump_opcode_double_deref.cu
        constraints/opcodes/evaluate_jump_opcode_rel_imm.cu
        constraints/opcodes/evaluate_jump_opcode_rel.cu
        constraints/opcodes/evaluate_jnz_opcode.cu
        constraints/opcodes/evaluate_jnz_opcode_non_taken.cu
        constraints/opcodes/evaluate_jnz_opcode_taken.cu
        constraints/opcodes/evaluate_mul_opcode.cu
        constraints/opcodes/evaluate_mul_opcode_small.cu
        constraints/opcodes/evaluate_generic_opcode.cu
        constraints/opcodes/evaluate_qm_31_add_mul_opcode.cu
        witness/gen_blake_g_trace.cu
        witness/gen_add_opcode_trace.cu
        witness/gen_add_opcode_small_trace.cu
        witness/gen_add_ap_opcode_trace.cu
        witness/gen_assert_eq_opcode_trace.cu
        witness/gen_assert_eq_opcode_double_deref_trace.cu
        witness/gen_assert_eq_opcode_imm_trace.cu
        witness/gen_triple_xor_32_trace.cu
        witness/gen_verify_bitwise_xor_trace.cu
        witness/gen_blake_round_trace.cu
        witness/gen_memory_address_to_id_trace.cu
        witness/gen_blake_round_sigma_trace.cu
        witness/gen_memory_id_to_big_trace.cu
        witness/gen_range_check_vector.cu
        witness/gen_blake_compress_opcode_trace.cu
        witness/gen_call_opcode_trace.cu
        witness/gen_call_opcode_rel_imm_trace.cu
        witness/gen_generic_opcode_trace.cu
        witness/gen_jnz_opcode_trace.cu
        witness/gen_jnz_opcode_taken_trace.cu
        witness/gen_jump_opcode_trace.cu
        witness/gen_jump_opcode_double_deref_trace.cu
        witness/gen_jump_opcode_rel_trace.cu
        witness/gen_jump_opcode_rel_imm_trace.cu
        witness/gen_mul_opcode_trace.cu
        witness/gen_mul_opcode_small_trace.cu
        witness/gen_qm_31_add_mul_opcode_trace.cu
        witness/gen_ret_opcode_trace.cu
        witness/gen_add_mod_builtin_trace.cu
        witness/gen_mul_mod_builtin_trace.cu
        witness/gen_bitwise_builtin_trace.cu
        witness/gen_range_check_builtin_bits_96_trace.cu
        witness/gen_range_check_builtin_bits_128_trace.cu
        witness/gen_poseidon_builtin_trace.cu
        witness/gen_poseidon_aggregator_trace.cu
        witness/gen_verify_instruction_trace.cu
        witness/gen_cube_252_trace.cu
        witness/gen_poseidon_full_round_chain_trace.cu
        witness/gen_poseidon_3_partial_rounds_chain_trace.cu
        witness/gen_range_check_felt_252_width_27_trace.cu
        witness/gen_partial_ec_mul_trace.cu
        witness/gen_partial_ec_mul_wb18_trace.cu
        witness/gen_pedersen_points_table_trace.cu
        witness/gen_pedersen_builtin_trace.cu
        witness/gen_pedersen_builtin_narrow_trace.cu
        witness/gen_poseidon_builtin_split_trace.cu
        witness/gen_pedersen_aggregator_wb18_trace.cu
        witness/gen_partial_ec_mul_wb9_trace.cu
        witness/gen_pedersen_aggregator_wb9_trace.cu
        witness/gen_pedersen_points_table_wb9_trace.cu
        # NOTE: gen_pedersen_table_on_gpu.cu is excluded due to multiple definition conflicts
        # with ec_ops.cuh and pedersen_table.cuh. The GPU table generation feature is experimental
        # and requires header refactoring to work properly.
        witness/gen_preprocessed_columns_on_gpu.cu
)
set_target_properties(stwo_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stwo_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(stwo_cuda PRIVATE -fPIC)
target_link_libraries(stwo_cuda CUDA::curand)
target_compile_options(stwo_cuda PRIVATE --ptxas-options=-v)
target_link_libraries(stwo_cuda CUDA::curand)
target_include_directories(stwo_cuda PRIVATE ${CUH_DIRS})
install(TARGETS stwo_cuda DESTINATION .)
