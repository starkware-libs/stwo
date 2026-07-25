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
        quotients.cu
        gkr.cu
        mle.cu
        timer.cu
        evaluate_common.cu
        prefix_sum.cu
        cuda_mem_pool.cu
)
set_target_properties(stwo_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stwo_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(stwo_cuda PRIVATE -fPIC)
target_link_libraries(stwo_cuda CUDA::curand)
target_compile_options(stwo_cuda PRIVATE --ptxas-options=-v)
target_link_libraries(stwo_cuda CUDA::curand)
target_include_directories(stwo_cuda PRIVATE ${CUH_DIRS})
install(TARGETS stwo_cuda DESTINATION .)
