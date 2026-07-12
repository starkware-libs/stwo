#include "fold_line.cuh"
#include "fri_utils.cuh"
#include "utils.cuh"

__global__ void fold_line_kernel(
    const m31 *domain,
    const uint32_t twiddle_offset,
    const uint32_t n,
    const qm31 alpha,
    m31 **eval_values,
    m31 **folded_values
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (n >> 1)) {
        const uint32_t x_inverse = domain[i + twiddle_offset];

        const uint32_t index_left = 2 * i;
        const uint32_t index_right = index_left + 1;

        const qm31 f_x = getEvaluation(eval_values, index_left);
        const qm31 f_x_minus = getEvaluation(eval_values, index_right);

        const qm31 f_0 = add(f_x, f_x_minus);
        const qm31 f_1 = mul_by_scalar(sub(f_x, f_x_minus), x_inverse);

        const qm31 f_prime = add(f_0, mul(alpha, f_1));

        folded_values[0][i] = f_prime.a.a;
        folded_values[1][i] = f_prime.a.b;
        folded_values[2][i] = f_prime.b.a;
        folded_values[3][i] = f_prime.b.b;
    }
}

// 1b buffer/pointer-array reuse: the caller (CudaBackend::fold_line) reuses a fixed set of
// device coordinate buffers across all alpha-steps of a layer, so the device 4-pointer arrays
// only need to be built once per buffer (not per step). `fold_line_alloc_coord_ptrs` uploads one
// host `m31*[4]` coordinate-pointer array to the device; `fold_line_launch` launches the kernel
// against already-uploaded device pointer arrays (no per-call clone/free). Both are freed once by
// the caller via `cuda_free_memory`. The kernel and launch geometry are byte-for-byte identical to
// the previous per-step `fold_line`.
extern "C"
m31 **fold_line_alloc_coord_ptrs(m31 **coord_ptrs) {
    return clone_to_device<m31*>(coord_ptrs, 4);
}

extern "C"
void fold_line_launch(
    m31 *gpu_domain,
    uint32_t twiddle_offset,
    uint32_t n,
    qm31 alpha,
    m31 **eval_values_device,
    m31 **folded_values_device
) {
    int block_dim = 1024;
    int num_blocks = (n / 2 + block_dim - 1) / block_dim;
    fold_line_kernel<<<num_blocks, block_dim>>>(
        gpu_domain,
        twiddle_offset,
        n,
        alpha,
        eval_values_device,
        folded_values_device
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: stream ordering ensures the kernel completes before the caller's async frees.
}

// =====================================================================================
// Improvement 1a: fused batched line-fold.
//
// Replaces the k sequential single-step `fold_line_launch` calls (k = fold_step, typically <= 4)
// of one FRI layer with ONE kernel that performs all k line-fold steps block-resident in shared
// memory. This turns k full global read+write round-trips into 1 read + 1 write.
//
// Math is byte-for-byte identical to `fold_line_kernel` applied k times: each step computes
//   out[g] = f0 + alpha_r * f1,  f0 = f(x)+f(-x),  f1 = (f(x)-f(-x)) * x_inv,
// pairing adjacent (2g, 2g+1) in bit-reversed order, x_inv = domain[twiddle_offset_r + g].
// The alphas are applied in the same order (alphas[0] first) and the exact same qm31 op sequence
// (add / mul_by_scalar / sub / mul(alpha,.) / add) is reused, so no reduction/order changes.
//
// ------------------------------------------------------------------------------------
// TILING (why a full 1024<<k tile is NOT used):
//
// Output element g after k steps depends only on the contiguous input block [g<<k, (g+1)<<k)
// (pairing is always adjacent 2i,2i+1), so a block producing outputs [B, B+T) needs exactly the
// contiguous inputs [B<<k, (B+T)<<k) -- a tile of (T<<k) qm31 values. A full T=1024 tile would be
// 1024<<k qm31 = up to 256 KiB for k=4, far beyond the 48 KiB static shared-memory budget. So we
// fix the *tile capacity in elements-per-coord* (TILE_CAP) and derive outputs-per-block
//   T = TILE_CAP >> k
// The shared tile holds exactly TILE_CAP qm31 (32 KiB), independent of k, and each block owns
// TILE_CAP contiguous inputs producing T final outputs.
//
// ------------------------------------------------------------------------------------
// THE TWIDDLE-INDEX MAP (the #1 fingerprint-break risk) -- derived exactly:
//
// Global layer length is n0 = n at entry; log_n0 = n0.ilog2(). Step r (0-based) has input length
// n_r = n0 >> r and (matching the single-step reference)
//   twiddle_offset_r = twiddles_size - (1 << (log_n0 - r)).
// These k offsets are precomputed host-side and passed in `twiddle_offsets`.
//
// A block owns final outputs [B, B+T) with B = blockIdx.x * T. Trace its live region per step:
//   step r INPUT indices (global):  [B << (k-r),     (B+T) << (k-r)    )   (length T << (k-r))
//   step r OUTPUT indices (global): [B << (k-r-1),   (B+T) << (k-r-1)  )   (length T << (k-r-1))
// A local output slot j_local in [0, T<<(k-r-1)) therefore maps to the GLOBAL step-r output index
//   g = (B << (k-r-1)) + j_local
// and uses x_inv = domain[twiddle_offset_r + g].
//   - r = 0     : outputs global [B<<(k-1), (B+T)<<(k-1)), count T<<(k-1).            (first step)
//   - r = k-1   : outputs global [B, B+T), count T.                                   (final step)
// Inside shared memory the block is self-contained: local input slot s in [0, T<<(k-r)) is global
// input index (B<<(k-r)) + s, and step r reduces local slot pair (2*j_local, 2*j_local+1) -> slot
// j_local. The (B<<(k-r-1)) base term is exactly what ties the local reduction back to the correct
// global twiddle.
// =====================================================================================

// Tile capacity in elements per coordinate. 2048 qm31 = 2048 * 16 B = 32 KiB of static shared
// memory (under the 48 KiB default per-block limit, so no opt-in attribute needed). Must be a
// power of two and >= (1 << MAX_SUPPORTED_K) so that T = TILE_CAP >> k >= 1.
#define FOLD_LINE_TILE_CAP 2048u
#define FOLD_LINE_BLOCK_DIM 1024u
// Max k the fused path supports (T >= 1 requires (1<<k) <= TILE_CAP).
#define FOLD_LINE_MAX_K 8u

// One block folds one contiguous tile of `tile_size` (= T << k) inputs into T outputs, running all
// k steps resident in shared memory. `tile_size`, `outputs_per_block` (= T) and `k` are launch
// parameters (derived host-side from TILE_CAP and k). `n0` is the full layer length at entry.
__global__ void fold_line_batch_kernel(
    const m31 *domain,
    const uint32_t *twiddle_offsets, // k offsets, twiddle_offsets[r] for step r
    const uint32_t n0,               // full layer length at entry (== eval length)
    const uint32_t k,                // number of fold steps (== alphas.len())
    const uint32_t tile_size,        // == outputs_per_block << k, <= FOLD_LINE_TILE_CAP
    const uint32_t outputs_per_block,// T
    const qm31 *alphas,              // k alphas, applied alphas[0] first
    m31 **eval_values,               // input coord-ptr array (4 distinct coords)
    m31 **folded_values              // output coord-ptr array (4 distinct coords)
) {
    __shared__ qm31 tile[FOLD_LINE_TILE_CAP];

    const uint32_t B = blockIdx.x * outputs_per_block; // first output index owned by this block
    const uint32_t input_base = B << k;                // first input index owned by this block

    // Cooperative coalesced load of the block's contiguous input tile [input_base, input_base+tile)
    // Adjacent threads read adjacent global elements -> coalesced. Guarded by n0 (tail blocks).
    for (uint32_t s = threadIdx.x; s < tile_size; s += blockDim.x) {
        const uint32_t g = input_base + s;
        if (g < n0) {
            tile[s] = getEvaluation(eval_values, g);
        }
    }
    __syncthreads();

    // Run the k reduction steps in shared memory. `live` = number of live local slots (inputs to
    // the current step). Step r reduces `live` slots -> `live/2` slots.
    //
    // In-place hazard note: output slot j reads input slots 2j and 2j+1; since 2j can be ANOTHER
    // thread's own output slot, an in-place "read-then-write" without a barrier between them would
    // race (thread 2j could overwrite tile[2j] before thread j reads it). We therefore split each
    // step into (a) every thread reads its pair into a register and computes f_prime, then a
    // __syncthreads(), then (b) every thread writes its result. This makes the reduction
    // read-before-write safe while staying fully in shared memory (no extra global traffic). The
    // per-thread f_prime register survives the barrier (barriers do not clobber registers).
    uint32_t live = tile_size; // == T << k
    for (uint32_t r = 0; r < k; ++r) {
        const uint32_t out_count = live >> 1;              // local outputs produced this step
        const uint32_t twiddle_offset_r = twiddle_offsets[r];
        // Base term tying local output slot -> global step-r output index (see derivation above):
        //   g = (B << (k-r-1)) + j_local
        const uint32_t global_out_base = B << (k - r - 1);
        const qm31 alpha = alphas[r];

        // Phase (a): read pairs + compute, buffering results in registers. A thread may own more
        // than one output slot (out_count > blockDim.x when tile_size > 2*blockDim.x), so buffer a
        // small fixed array indexed by the grid-stride iteration. FOLD_LINE_TILE_CAP / blockDim
        // bounds the iteration count; with TILE_CAP=2048, blockDim=1024 the max is 1 (tile_size<=
        // 2048 => out_count <= 1024 == blockDim), but size the buffer for the general bound.
        qm31 results[FOLD_LINE_TILE_CAP / FOLD_LINE_BLOCK_DIM + 1u];
        uint32_t nres = 0;
        for (uint32_t j = threadIdx.x; j < out_count; j += blockDim.x) {
            const uint32_t x_inverse = domain[twiddle_offset_r + global_out_base + j];

            const qm31 f_x = tile[2 * j];
            const qm31 f_x_minus = tile[2 * j + 1];

            const qm31 f_0 = add(f_x, f_x_minus);
            const qm31 f_1 = mul_by_scalar(sub(f_x, f_x_minus), x_inverse);

            results[nres++] = add(f_0, mul(alpha, f_1));
        }
        // Barrier: all reads of this step's inputs are complete before any write.
        __syncthreads();
        // Phase (b): write buffered results back into the front of the tile.
        nres = 0;
        for (uint32_t j = threadIdx.x; j < out_count; j += blockDim.x) {
            tile[j] = results[nres++];
        }
        __syncthreads();
        live = out_count;
    }

    // Write the T final outputs. tile[0..T) hold outputs for global indices [B, B+T).
    for (uint32_t j = threadIdx.x; j < outputs_per_block; j += blockDim.x) {
        const uint32_t g = B + j;
        if (g < (n0 >> k)) {
            const qm31 v = tile[j];
            folded_values[0][g] = v.a.a;
            folded_values[1][g] = v.a.b;
            folded_values[2][g] = v.b.a;
            folded_values[3][g] = v.b.b;
        }
    }
}

// Host wrapper for the fused batched fold. Returns true if the fused path launched, false if the
// layer is too small / k too large to tile (caller must then use the k-launch single-step path).
//
// Parameters:
//   gpu_domain            : device itwiddle/domain array base ptr.
//   twiddle_offsets       : host array of k offsets (offset_r = twiddles_size - (1<<(log_n0-r))).
//   n0                    : full layer length at entry (== eval length, power of two).
//   k                     : number of fold steps (== alphas.len()).
//   alphas                : host array of k qm31 alphas (alphas[0] applied first).
//   eval_values_device    : device input coord-ptr array (from fold_line_alloc_coord_ptrs).
//   folded_values_device  : device output coord-ptr array (from fold_line_alloc_coord_ptrs).
extern "C"
bool fold_line_batch(
    m31 *gpu_domain,
    const uint32_t *twiddle_offsets,
    uint32_t n0,
    uint32_t k,
    const qm31 *alphas,
    m31 **eval_values_device,
    m31 **folded_values_device
) {
    // Fallback conditions: keep the single-step path for anything the tiling can't cover.
    //  - k == 0                : nothing to do (caller guarantees k >= 1, defensive).
    //  - k > FOLD_LINE_MAX_K   : (1<<k) would exceed the tile capacity => T < 1.
    //  - (1u << k) > TILE_CAP  : tile can't hold even one output's inputs => T < 1.
    //  - n0 < (1u << k)        : layer folds below length 1 (caller also asserts this away).
    //  - n0 % TILE_CAP != 0    : layer smaller than one full tile OR not tile-aligned; the tiny
    //                            last FRI layers hit this. Single-step path handles them exactly.
    if (k == 0u || k > FOLD_LINE_MAX_K) {
        return false;
    }
    if ((1u << k) > FOLD_LINE_TILE_CAP) {
        return false;
    }
    if (n0 < (1u << k)) {
        return false;
    }
    const uint32_t tile_size = FOLD_LINE_TILE_CAP;      // elements per block (== T << k)
    const uint32_t outputs_per_block = tile_size >> k;  // T
    if (n0 % tile_size != 0u) {
        return false;
    }

    // Upload the small per-step arrays (k offsets, k alphas) to the device.
    uint32_t *twiddle_offsets_device = clone_to_device<uint32_t>(
        const_cast<uint32_t *>(twiddle_offsets), k);
    qm31 *alphas_device = clone_to_device<qm31>(const_cast<qm31 *>(alphas), k);

    const uint32_t num_blocks = n0 / tile_size; // == (n0 >> k) / outputs_per_block
    const uint32_t block_dim = FOLD_LINE_BLOCK_DIM;

    fold_line_batch_kernel<<<num_blocks, block_dim>>>(
        gpu_domain,
        twiddle_offsets_device,
        n0,
        k,
        tile_size,
        outputs_per_block,
        alphas_device,
        eval_values_device,
        folded_values_device
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: stream ordering ensures the kernel completes before the caller's async frees.
    cuda_free_memory(twiddle_offsets_device);
    cuda_free_memory(alphas_device);
    return true;
}
