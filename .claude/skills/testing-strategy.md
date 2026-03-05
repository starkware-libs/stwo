---
name: testing-strategy
description: >
  Test taxonomy, coverage strategy, and test patterns for STWO. Covers unit
  test locations, coverage gaps, property testing opportunities, CI matrix,
  and patterns for field, prove-verify, and constraint tests. Use when adding
  tests, reviewing coverage, debugging failures, or assessing test adequacy.
---

# Testing Strategy

## Test Taxonomy

### Unit Tests (Inline)

All tests are inline `#[cfg(test)]` modules — no separate `tests/` directories.

**Well-tested domains** (run `cargo test --features prover -- --list` to get current counts):

| Domain | File(s) | Coverage |
|--------|---------|----------|
| FRI verifier | `core/fri.rs` | Excellent — includes rejection tests |
| Field arithmetic | `core/fields/m31.rs`, `cm31.rs`, `qm31.rs` | Basic ops + inverse |
| Batch inverse | `core/fields/mod.rs` | Including edge cases |
| Circle group | `core/circle.rs` | Generator, coset, random point |
| Vanishing polys | `core/constraints.rs` | Coset/point vanishing + degree |
| Blake2s channel | `core/channel/blake2s.rs` | Draw/mix operations |
| Merkle trees | `core/vcs/blake2_merkle.rs` | Including decommit failures |
| SIMD M31 | `prover/backend/simd/m31.rs` | Arithmetic + SIMD load/store |
| SIMD FFT | `prover/backend/simd/fft/` | Butterfly through full FFT |
| SIMD circle | `prover/backend/simd/circle.rs` | Eval, interpolate, extend |
| GKR/lookups | `prover/lookups/` | Sumcheck, GKR, grand product |
| PCS | `core/pcs/mod.rs` | Security bits only — **sparse** |

### E2E Tests (via Examples)

The `crates/examples/` crate contains prove-then-verify tests:
- Wide Fibonacci with preprocessed columns
- Poseidon2 hash proof
- Blake2s proof (round, scheduler, XOR table)
- State machine
- Plonk-style
- XOR with GKR lookups

These test the full proving pipeline but only on the **happy path**.

### Benchmarks as Tests

Criterion benchmarks in `crates/stwo/benches/` exercise hot paths.
They serve as regression detectors in CI.

## Coverage Gaps (CRITICAL)

### Soundness-Critical Files With Zero Unit Tests

| File | Risk | What's Missing |
|------|------|----------------|
| `core/verifier.rs` | **CRITICAL** | No rejection tests for malformed proofs |
| `core/pcs/verifier.rs` | **CRITICAL** | No unit tests for PCS verification |
| `core/pcs/quotients.rs` | **CRITICAL** | No tests for DEEP quotient construction |
| `prover/lookups/gkr_prover.rs` | HIGH | Tested only transitively |
| `prover/lookups/mle.rs` | HIGH | Empty `#[cfg(test)]` block |
| `core/proof_of_work.rs` | MEDIUM | PoW verifier side untested |

### Missing Test Categories

1. **No property-based tests**: No proptest, quickcheck, or `Arbitrary` anywhere.
   This is a significant gap for field arithmetic and polynomial operations.

2. **No adversarial verifier tests**: The verifier is only tested on valid proofs.
   There are no tests for:
   - Proofs with corrupted Merkle commitments
   - Proofs with wrong OODS evaluations
   - Proofs with invalid composition polynomial
   - Proofs with wrong PoW nonce

3. **No fuzzing targets**: No cargo-fuzz or honggfuzz configurations.

## How to Run Tests

```bash
# Quick: standard tests
cargo test --features prover

# Full: all features
cargo test --features "prover,parallel,tracing"

# Slow tests (release mode)
cargo test --release --features "slow-tests,prover"

# Specific crate
cargo test --package stwo --features prover
cargo test --package stwo-constraint-framework --features prover

# Specific test
cargo test --features prover -- test_name

# No-std check
cd ensure-verifier-no_std && cargo build -r
```

## Test Patterns for STWO

### Field Arithmetic Tests

```rust
#[test]
fn test_field_op() {
    let a = M31::from(42);
    let b = M31::from(17);
    // Verify algebraic identities
    assert_eq!(a + b - b, a);           // Additive inverse
    assert_eq!(a * b * b.inverse(), a); // Multiplicative inverse
    assert_eq!(a * M31::one(), a);      // Multiplicative identity
    assert_eq!(a + M31::zero(), a);     // Additive identity
}
```

### Prove-Verify Round Trip

```rust
#[test]
fn test_prove_verify() {
    // 1. Generate trace
    let trace = generate_trace(&input);
    // 2. Set up commitment scheme
    let config = PcsConfig { ... };
    // 3. Prove
    let proof = prove::<SimdBackend, Blake2sMerkleChannel>(&components, &mut channel, scheme);
    // 4. Verify
    verify::<Blake2sMerkleChannel>(&components, &mut channel, &mut verifier, proof).unwrap();
}
```

### Constraint Assertion

```rust
#[test]
fn test_constraints_hold() {
    let trace = generate_trace(&input);
    // Use AssertEvaluator to check constraints row-by-row
    assert_constraints_on_trace(&component, &trace);
}
```

### FRI Rejection Tests (Reference Pattern)

From `core/fri.rs` — the best-tested module. Pattern to follow:

```rust
#[test]
fn test_fri_proof_invalid_X() {
    let (proof, config) = generate_valid_proof();
    let mut corrupted = proof;
    // Corrupt specific element
    corrupted.X = invalid_value;
    // Verify rejection
    let result = verify(corrupted);
    assert!(matches!(result, Err(FriVerificationError::XInvalid)));
}
```

## Adding Tests: Priority Order

When adding tests, prioritize by risk:

1. **Verifier rejection tests** — Ensure malformed proofs are rejected
2. **Property tests for field arithmetic** — All algebraic identities hold
3. **DEEP quotient correctness** — Quotients are actually low-degree
4. **PCS verifier unit tests** — Commitment verification logic
5. **GKR prover correctness** — Prove-verify round trips
6. **Adversarial proof tests** — Proofs with targeted corruptions

## Test Configuration

### Feature Flags

| Flag | Effect |
|------|--------|
| `prover` | Enables prover code (required for most tests) |
| `parallel` | Enables rayon parallelism |
| `slow-tests` | Enables long-running tests (release mode only) |
| `tracing` | Enables tracing instrumentation |

### CI Matrix

Tests run on multiple configurations:
- nightly + prover (primary)
- stable + prover (compatibility)
- stable + no-default-features (verifier-only)
- nightly + parallel (threading)
- nightly + slow-tests + release
- AVX2, AVX512, NEON, WASM targets

## Forbidden Actions

In this domain, agents must NEVER:
- Remove or weaken any existing test
- Mark a test as `#[ignore]` without documented justification
- Reduce test coverage in soundness-critical code
- Use `#[should_panic]` to mask a real failure
- Skip running tests before submitting changes to proof-system code
