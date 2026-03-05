---
name: performance-optimization
description: >
  STWO-specific performance patterns: SIMD backends, FFT hot paths, memory
  pooling, benchmarking, parallel proving, and regression detection. Use when
  benchmarking, optimizing hot paths, working on SIMD backends, analyzing
  regressions, or reviewing parallel proving code.
---

# Performance Optimization

## Architecture Overview

### Backend Hierarchy

```
prover/backend/
  cpu/       Reference implementation (slow, readable)
  simd/      Production implementation (fast, unsafe-heavy)
```

The SIMD backend is the primary production path. The CPU backend is:
- Reference for correctness verification
- Fallback for unsupported architectures
- Baseline for benchmark comparisons

### SIMD Targets

| Target | Feature | Lanes | Status |
|--------|---------|-------|--------|
| x86_64 AVX512 | `avx512f` | 16 x u32 | Primary production target |
| x86_64 AVX2 | `avx2` | 2 x 8 x u32 | Supported |
| AArch64 NEON | `neon` | 4 x 4 x u32 | Supported |
| WASM SIMD128 | `simd128` | 4 x 4 x u32 | Supported |

**PackedM31**: 16 M31 elements processed in parallel regardless of platform.
On platforms with < 512-bit SIMD, this is emulated with multiple operations.

### Memory Optimization

**Memory Pool** (`prover/mempool.rs`):
- `BaseColumnPool` reuses Vec allocations across proving rounds
- Uses `DashMap` for thread-safe concurrent access
- Avoids repeated large allocation + deallocation during polynomial evaluation

**Uninitialized Allocation**:
- `uninit_vec()` in `core/utils.rs` skips zero-initialization
- All callers MUST write before read — this is enforced by code review, not the type system
- Used in FFT buffers, quotient computation, trace generation

## Benchmarking

### Running Benchmarks

```bash
# All benchmarks (single-threaded)
cargo bench --features prover

# With parallelism
cargo bench --features "prover,parallel"

# Specific benchmark
cargo bench --features prover --bench fft
cargo bench --features prover --bench field
cargo bench --features prover --bench fri

# With AVX512 (on supported hardware)
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f" cargo bench --features prover
```

### Benchmark Suites

| Benchmark | Domain | Key Metrics |
|-----------|--------|-------------|
| `field` | M31/CM31/QM31 + SIMD throughput | ops/sec |
| `fft` | SIMD ifft, rfft, ifft_parts | elements/sec by log_size |
| `fri` | FRI fold throughput | elements/sec |
| `merkle` | Blake2s Merkle commit | hashes/sec |
| `pcs` | Full PCS prove/verify | proof time |
| `eval_at_point` | Circle poly evaluation | points/sec |
| `barycentric_eval_at_point` | Barycentric eval | points/sec |
| `bit_rev` | Bit-reversal permutation | elements/sec |
| `lookups` | GKR grand product + logup | layers/sec |
| `prefix_sum` | SIMD prefix sum | elements/sec |
| `poseidon` (examples) | Poseidon2 proof generation | proofs/sec |

### CI Regression Detection

The CI runs `run-avx512-bench` and `run-avx512-bench-parallel` on dedicated
`stwo-avx` runners. The `benchmark-action/github-action-benchmark@v1` action:
- Compares against previous benchmark data
- Fails on alert (regression detected)
- Comments on PRs with performance changes
- Alerts `@gilbens-starkware` on regressions

## Hot Paths

### FFT (Dominant Cost)

The SIMD FFT in `prover/backend/simd/fft/` is the single most
performance-critical code in STWO. It uses:

- Raw pointer arithmetic for zero-overhead array access
- Platform-specific SIMD intrinsics
- In-place computation to minimize memory traffic
- Twiddle factor precomputation (`prover/poly/twiddles.rs`)
- Bit-reversal permutation fused with FFT layers

**Files**: `simd/fft/rfft.rs` (forward), `simd/fft/ifft.rs` (inverse)

### Merkle Tree Hashing

Blake2s hashing for Merkle commitments is the second hot path:
- `prover/backend/simd/blake2s.rs` — SIMD Blake2s (16 parallel hashes)
- `prover/backend/simd/blake2s_lifted.rs` — Lifted Merkle variant

### Quotient Computation

DEEP quotient evaluation at query points:
- `prover/backend/simd/quotients.rs` — SIMD quotient computation
- Uses `from_simd_unchecked` to skip field reduction when values are known valid

## Parallelism

Feature: `parallel` (enables `rayon`)

Key parallel sections:
- Trace generation (row-parallel)
- Constraint evaluation (domain-parallel)
- Merkle tree construction (subtree-parallel)
- FRI fold (element-parallel)

**Thread safety**: `UnsafeMut`/`UnsafeConst` wrappers in `simd/utils.rs`
enable passing raw pointers across rayon threads. This requires that
no two threads write to the same memory location.

## Profile Configuration

```toml
# Cargo.toml [profile.bench]
codegen-units = 1    # Single codegen unit for maximum optimization
lto = true           # Link-time optimization
```

## Optimization Rules

1. **Never sacrifice correctness for performance.** A fast wrong answer is worse
   than a slow right answer.
2. **Benchmark before and after.** Every optimization must be measured.
3. **Regression = blocking issue.** Performance regressions are not tech debt.
4. **SIMD must match scalar.** Every SIMD optimization must produce identical
   results to the CPU reference.
5. **Document unsafe.** Every unsafe block in the hot path needs a safety comment.

## Review Checklist

Before approving performance changes:
- [ ] Benchmark shows measurable improvement
- [ ] No regression on other benchmarks
- [ ] SIMD implementation matches CPU reference
- [ ] Unsafe code has documented safety invariants
- [ ] Memory allocation patterns don't leak
- [ ] Thread safety is maintained under `parallel` feature

## Forbidden Actions

In this domain, agents must NEVER:
- Optimize by removing verification checks
- Add `unsafe` without a clear performance justification AND safety proof
- Change algorithmic correctness for speed (e.g., approximate reduction)
- Skip benchmarking when claiming an optimization
- Modify the bench profile without understanding CI implications
