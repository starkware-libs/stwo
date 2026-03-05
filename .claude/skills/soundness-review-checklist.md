---
name: soundness-review-checklist
description: >
  Structured checklist for reviewing soundness-critical code changes in STWO.
  Run this checklist before approving ANY modification to: constraint logic,
  FRI protocol, verifier, field arithmetic, polynomial commitment scheme,
  Fiat-Shamir channel, proof serialization, or security parameters.
---

# Soundness Review Checklist

## Canonical Theory Sources

- `.agents/papers/llm/INDEX.llm.md` — first stop to map concepts and notation
- `.agents/papers/llm/Circle_STARKs.llm.md` — circle-FFT/FRI/AIR math anchors
- `.agents/papers/llm/Stwo_Whitepaper.llm.md` — STWO protocol/soundness/parameter anchors

## When to Run

Run this checklist for ANY change touching:
- `crates/stwo/src/core/fri.rs` or `crates/stwo/src/prover/fri.rs`
- `crates/stwo/src/core/verifier.rs`
- `crates/stwo/src/core/pcs/`
- `crates/stwo/src/core/fields/`
- `crates/stwo/src/core/constraints.rs`
- `crates/stwo/src/core/channel/`
- `crates/stwo/src/core/proof.rs`
- `crates/stwo/src/prover/mod.rs` (prove functions)
- `crates/constraint-framework/src/logup.rs`
- Any file tagged [SOUNDNESS-CRITICAL]

## Pre-Review: Context Loading

- [ ] Read the divergence log: `.claude/skills/paper-implementation-divergence-log.md`
- [ ] Identify which distilled-file anchor governs the modified code
- [ ] Load the relevant mathematical skill (circle-stark-mathematics, finite-field-arithmetic, etc.)

## 1. Mathematical Correctness

- [ ] **Invariant identification**: What mathematical invariant does this code maintain?
      State it explicitly.
- [ ] **Theory grounding**: Can the modified logic be traced to a specific definition,
      theorem, or algorithm in `Circle_STARKs.llm.md` or `Stwo_Whitepaper.llm.md`?
- [ ] **Divergence check**: Does this change introduce a new paper-implementation divergence?
      If yes, document in the divergence log before proceeding.

## 2. Constraint System Integrity

- [ ] **No constraint removal**: No constraint has been removed or weakened
- [ ] **Degree preservation**: Constraint degree bounds are unchanged or correctly updated
- [ ] **LogUp balance**: If logup interactions are modified, verify they still balance
- [ ] **Completeness**: Valid witnesses still satisfy all constraints
- [ ] **Soundness**: Invalid witnesses are still rejected (no new under-constraint)

## 3. FRI Protocol Correctness

- [ ] **Folding chain**: FRI folding operations are mathematically correct
      (challenge mixing, domain halving, polynomial splitting)
- [ ] **Layer verification**: All FRI layers are verified (none skipped)
- [ ] **Last layer check**: Last layer polynomial degree bound is enforced
- [ ] **Query consistency**: Query positions are consistently mapped across layers
- [ ] **Domain chain**: Each folded domain is correctly derived from the previous one

## 4. Fiat-Shamir Transcript

- [ ] **Ordering preserved**: Commitments are mixed BEFORE challenges are drawn
- [ ] **No reordering**: The sequence of mix/draw operations is identical in prover and verifier
- [ ] **Complete binding**: All proof elements are mixed into the transcript
      (no unbound values that an adversary could vary)
- [ ] **Channel state**: Channel state is deterministic given the same inputs

## 5. Field Arithmetic

- [ ] **Reduction correctness**: All arithmetic results are properly reduced
- [ ] **No overflow**: Intermediate computations do not overflow their integer types
- [ ] **Extension field**: Extension field operations use correct irreducible polynomials
- [ ] **SIMD parity**: SIMD implementations match scalar reference behavior

## 6. Verifier Completeness

- [ ] **All checks present**: The verifier performs ALL required verification steps
- [ ] **Error propagation**: Verification failures are properly propagated (not swallowed)
- [ ] **OODS evaluation**: Composition polynomial OODS eval is correctly extracted
- [ ] **Proof of work**: PoW nonce is verified
- [ ] **Merkle verification**: Decommitments are verified against commitments

## 7. Security Parameters

- [ ] **No parameter weakening**: log_blowup_factor, n_queries, pow_bits are not reduced
- [ ] **Parameter validation**: FriConfig::new() range checks are preserved
- [ ] **Security bits**: Total security_bits() >= target (document what target is)

## 8. Test Coverage

- [ ] **Existing tests pass**: All tests in the modified module still pass
- [ ] **New test for change**: A test specifically exercises the modified behavior
- [ ] **Negative tests**: There are tests that verify rejection of invalid inputs
- [ ] **Edge cases**: Boundary conditions are tested (zero, maximum, single-element)

## 9. Unsafe Code (if applicable)

- [ ] **Justified**: The unsafe block has a documented safety argument
- [ ] **Minimal scope**: The unsafe block is as small as possible
- [ ] **Invariant preserved**: The unsafe code does not violate any field/type invariants
- [ ] **No UB**: There is no undefined behavior under any valid input

## Post-Review Actions

- [ ] Update divergence log if new divergence found
- [ ] Flag any coverage gaps found during review
- [ ] If confidence < 90%: ESCALATE with SOUNDNESS-ESCALATION tag

## Escalation Protocol

If ANY of the following are true, escalate to human review:

1. The change modifies a mathematical identity and you cannot prove equivalence
2. The change affects security parameters
3. You find an undocumented divergence from the distilled references
4. A soundness-critical component has zero test coverage for the modified path
5. The change introduces or modifies `unsafe` code in a soundness-critical file
6. You are not confident the change preserves all invariants listed above

**Format**:
```
SOUNDNESS-ESCALATION:
  File: [path]
  Change: [description]
  Invariant at risk: [which invariant]
  Paper reference: [Circle_STARKs.llm.md anchor / Stwo_Whitepaper.llm.md anchor]
  Confidence: [percentage]
  Reason for escalation: [why]
```
