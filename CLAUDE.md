# STWO — Circle STARK Proving System

STWO is StarkWare's production Circle STARK proving system.
Soundness bugs are catastrophic and irreversible — an accepted invalid proof or rejected valid proof is a
protocol failure with no recovery path.

## Priority Contract

All agent behavior is governed by this hierarchy. No exception, no override.

1. **SOUNDNESS & SECURITY** — Cryptographic correctness is absolute.
2. **PRODUCTION QUALITY** — All tests pass. All gates hold. No guardrail bypassed.
3. **PERFORMANCE** — Prover performance is a correctness property. Regressions block.

## Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Language | Rust nightly-2025-07-14 | Stable features except SIMD. See `rust-toolchain.toml` |
| Field | Mersenne31 (M31) | CM31, QM31 extension tower. p = 2^31 - 1 |
| Proof system | Circle STARKs | FRI-based, circle group C(F_p) |
| Test framework | Rust built-in + criterion | No proptest (coverage gap) |
| Benchmarks | Criterion 0.5.1 | 11 benchmark suites + Poseidon example |
| CI | GitHub Actions | See `.github/workflows/ci.yaml` |
| Hash functions | Blake2s, Blake3, Poseidon252 | Blake2s primary for proofs |

## Workspace Structure

```
crates/
  stwo/                    Core prover + verifier library
    src/core/              [SOUNDNESS-CRITICAL] Verifier-side (no_std compatible)
      fields/              [SOUNDNESS-CRITICAL] M31, CM31, QM31 field arithmetic
      fri.rs               [SOUNDNESS-CRITICAL] FRI verifier
      verifier.rs          [SOUNDNESS-CRITICAL] Top-level STARK verifier
      pcs/                 [SOUNDNESS-CRITICAL] Polynomial commitment scheme
      channel/             [SECURITY-CRITICAL] Fiat-Shamir channel
      circle.rs            [SOUNDNESS-CRITICAL] Circle group, cosets, domains
      constraints.rs       [SOUNDNESS-CRITICAL] Vanishing polynomials
      vcs/                 [SECURITY-CRITICAL] Merkle tree (original)
      vcs_lifted/          [SECURITY-CRITICAL] Lifted Merkle tree
      poly/                Circle polynomials, line polynomials, domains
      fft.rs               Butterfly operations
      proof.rs             Proof serialization
      proof_of_work.rs     [SECURITY-CRITICAL] PoW verification
    src/prover/            [PERFORMANCE-CRITICAL] Prover-side (requires "prover" feature)
      backend/cpu/         CPU reference implementation
      backend/simd/        [PERFORMANCE-CRITICAL] SIMD-optimized implementation
        fft/               [PERFORMANCE-CRITICAL] SIMD FFT (heavy unsafe)
        m31.rs             PackedM31 SIMD field ops
      fri.rs               FRI prover
      pcs/                 PCS prover + quotient ops
      lookups/             GKR + LogUp + sumcheck
      mempool.rs           Memory pool for allocation reuse
    benches/               Criterion benchmarks
  constraint-framework/    Framework for defining AIR constraints
    src/logup.rs           [SOUNDNESS-CRITICAL] LogUp interaction constraints
    src/prover/            Constraint evaluation (SIMD + CPU)
  air-utils/               Trace generation utilities
  air-utils-derive/        Proc macros for AIR utilities
  examples/                Example implementations (Blake, Poseidon, Fibonacci, etc.)
  std-shims/               no_std compatibility shims
ensure-verifier-no_std/    CI gate: verifier compiles without std
```

## Commands

### Build
```bash
cargo build --release                          # Release build
cargo build --release --features prover        # With prover
cargo build --release --features "prover,parallel"  # With parallelism
```

### Test
```bash
cargo test --features "prover"                 # Standard tests
cargo test --features "prover,parallel"        # With parallelism
cargo test --no-default-features --package stwo  # Verifier-only (no prover)
cargo test --release --features "slow-tests,prover"  # Slow tests
cargo test --no-default-features --package stwo-constraint-framework  # Framework
```

### Lint & Format
```bash
scripts/clippy.sh                              # Clippy (all crates, all features)
scripts/rust_fmt.sh --check                    # Format check
scripts/rust_fmt.sh                            # Auto-format
```

### Benchmarks
```bash
cargo bench --features prover                  # All benchmarks
./poseidon_benchmark.sh                        # Poseidon proof benchmark
./scripts/bench.sh                             # CI benchmark script
```

### No-Std Verification
```bash
cd ensure-verifier-no_std && cargo build -r    # Must compile
```

## Mathematical Context

**CRITICAL**: Before modifying any soundness-critical component, load the
corresponding skill AND read the referenced paper section. Do not proceed
on mathematical intuition alone.

Known paper-implementation divergences: `.claude/skills/paper-implementation-divergence-log.md`
This document is authoritative. Undocumented divergences found during work
MUST be added before proceeding.

## Operation Boundaries

### Forbidden (NEVER do these)

- Modify constraint definitions, field arithmetic, FRI parameters, Fiat-Shamir
  channel, verifier logic, or commitment scheme without human approval
- Disable, bypass, or weaken any test, lint, or CI gate
- Add `unsafe` blocks in soundness-critical paths without documented justification
- Accept a paper-code divergence as "probably fine" without documentation and review
- Generate code that cannot be traced to a mathematical definition in the papers
- Reduce security parameters (blowup factor, query count, grinding bits)

### Supervised (state intent, get approval)

Before modifying ANY [SOUNDNESS-CRITICAL] or [SECURITY-CRITICAL] file:

1. Load the corresponding skill from `.claude/skills/`
2. Identify the paper section governing the logic
3. State the mathematical invariant the change must preserve
4. State how the change preserves it
5. Identify the test that will verify this
6. Get explicit human approval
7. After the change: verify all tests pass, update divergence log if needed

### Autonomous (proceed freely)

- Read any file
- Add/modify/remove tests
- Fix lint, clippy, formatting
- Add documentation
- Add benchmarks
- Refactor within a module without changing external interface or math behavior

## Key Architectural Decisions

1. **Feature-gated prover**: The `prover` feature flag separates prover code from
   verifier. The verifier is no_std compatible for on-chain deployment.
2. **SIMD-first performance**: The SIMD backend is the primary path. CPU backend
   exists as reference. SIMD uses extensive `unsafe` for performance.
3. **Lifted Merkle trees**: `vcs_lifted/` commits multiple polynomial sizes in a
   single tree by lifting smaller polynomials to the largest domain.
4. **QM31 decomposition**: Secure-field polynomials are decomposed into 4 base-field
   coordinate polynomials for commitment and FRI.
5. **Memory pooling**: `prover/mempool.rs` reuses allocations to avoid repeated
   large allocations during proving.
