---
name: security-review-checklist
description: >
  Security review checklist for STWO. Separate from soundness review.
  Covers: side-channel surface, input validation, unsafe code audit,
  dependency security, API misuse patterns, and proof malleability.
  Run this for any change affecting the public API, proof format,
  hash functions, or memory-safety-critical code.
---

# Security Review Checklist

## When to Run

Run this checklist for changes touching:
- Public API surface (`prove()`, `verify()`, `CommitmentSchemeProver/Verifier`)
- Hash function implementations (Blake2s, Poseidon252)
- Merkle tree operations (`vcs/`, `vcs_lifted/`)
- Proof serialization/deserialization (`core/proof.rs`)
- Memory allocation with `unsafe` (see unsafe audit section)
- Dependency updates
- SIMD/FFT implementation
- Channel/Fiat-Shamir implementation

## 1. Input Validation

- [ ] **Proof format**: Malformed proofs are rejected gracefully (no panic, no UB)
- [ ] **Size bounds**: All deserialized vectors have bounded length
- [ ] **Domain validation**: FRI domains are constructed with validated parameters
- [ ] **Parameter ranges**: FriConfig fields are within documented ranges

## 2. Proof Malleability

- [ ] **Unique encoding**: The proof format does not allow multiple valid
      encodings for the same proof (which could enable replay attacks)
- [ ] **Binding commitments**: All proof elements that should be bound by
      Fiat-Shamir ARE mixed into the channel
- [ ] **Non-malleable nonce**: PoW nonce is bound to the transcript state

## 3. Side-Channel Resistance

Note: STWO is a prover, not a signing algorithm. Side-channel resistance
is less critical but still relevant for:

- [ ] **Timing**: Field operations are constant-time where feasible
      (inverse via Fermat's little theorem has data-dependent squaring chain
      but this is standard and acceptable for a prover)
- [ ] **Memory access patterns**: No secret-dependent branching in hot paths
      (SIMD paths are naturally constant-time)

## 4. Unsafe Code Audit

For any new or modified `unsafe` block:

- [ ] **Safety comment**: Block has a `// SAFETY:` comment explaining the invariant
- [ ] **Bounds check**: Array accesses are proven in-bounds
- [ ] **Initialization**: Memory written before read (no use of uninitialized memory)
- [ ] **Alignment**: Pointer casts respect alignment requirements
- [ ] **Lifetime**: No dangling references or use-after-free
- [ ] **Send/Sync**: If `unsafe impl Send/Sync`, type is actually thread-safe

### Known Unsafe Patterns in STWO

| Pattern | Files | Risk Level |
|---------|-------|------------|
| `uninit_vec` / `set_len` | Throughout prover | LOW — write-before-read guaranteed by fill loops |
| `from_simd_unchecked` | SIMD backends | MEDIUM — caller must prove values < P |
| SIMD FFT raw pointers | `simd/fft/rfft.rs`, `ifft.rs` | HIGH — complex pointer arithmetic |
| `UnsafeMut`/`UnsafeConst` | `simd/utils.rs` | MEDIUM — enables parallel FFT, requires no aliasing |
| `transmute` for SIMD | `simd/m31.rs`, `simd/blake2s.rs` | MEDIUM — layout compatibility required |
| `mem::zeroed` | `simd/blake2s.rs` | LOW — only for [u8] buffers |

## 5. Dependency Security

- [ ] **No new crypto deps**: New cryptographic dependencies require explicit approval
- [ ] **Version pinning**: Crypto dependencies are version-pinned (not floating ranges)
- [ ] **Feature flags**: No unintended features enabled on dependencies
- [ ] **Supply chain**: Dependencies are from established maintainers

### Current Crypto Dependencies

Check `Cargo.toml` for current versions. Key crates:

| Crate | Purpose |
|-------|---------|
| blake2 | Blake2s hash (verifier-compatible) |
| blake3 | Blake3 hash (optional) |
| starknet-crypto | Poseidon252 hash |
| starknet-ff | Felt252 field for Poseidon |

## 6. API Misuse Prevention

- [ ] **Type safety**: Security-critical parameters are typed (not raw integers)
- [ ] **Builder pattern**: Complex configurations use builders with validation
- [ ] **Default safety**: Default configurations are clearly marked as test-only
      (see PcsConfig::default() — 13 bits of security)

## 7. No-Std Compatibility

- [ ] **Verifier no_std**: Changes do not break `ensure-verifier-no_std` compilation
- [ ] **Feature gating**: Prover-only code is behind `#[cfg(feature = "prover")]`
- [ ] **No std leaks**: No `std::` usage without `#[cfg(feature = "std")]` guard

## 8. Denial of Service

- [ ] **Proof size**: Proof verification time is bounded by proof size
- [ ] **No quadratic**: No O(n^2) or worse operations in the verifier
- [ ] **Memory bounds**: Verifier memory usage is bounded

## Escalation Protocol

If ANY of the following are true, escalate to human review:

1. New `unsafe` code in the verifier (core/) path
2. Modification to hash function usage
3. New cryptographic dependency
4. Change to proof serialization format
5. Change that could affect no_std verifier compilation

**Format**:
```
SECURITY-ESCALATION:
  File: [path]
  Change: [description]
  Attack surface: [what could be exploited]
  Mitigation: [what protects against it]
  Confidence: [percentage]
```
